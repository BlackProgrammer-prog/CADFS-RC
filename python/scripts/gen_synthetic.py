"""Generate the synthetic CADFS dataset.

Families: random / maze / narrow-passage.
Splits are MAP-LEVEL (a map appears in exactly one split) to prevent leakage.
Structural training uses moderate maze/narrow parameters; shift_family reserves
hard corridor/passage/clutter parameters for final testing.

Outputs (relative to repo root):
  data/synthetic/<family>/<split>/<map_id>.map        MovingAI-format maps
  data/instances/<split>.csv                          instance table
CSV schema:
  map_id,map_path,start_x,start_y,goal_x,goal_y,family,density,width,height,split,optimal_cost

Reproducible: one master seed drives everything; per-map seeds are derived.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Prefer the paper-timing Release engine, while retaining compatibility with
# the historical build directory and an in-tree copied extension.
for candidate in reversed((ROOT / "cmake-build-release", ROOT / "build", ROOT)):
    sys.path.insert(0, str(candidate))
import cadfs_engine as eng  # noqa: E402

CONN = 8
CFG = dict(W=2.0, connectivity=CONN)


# ----------------------------- map generators ------------------------------

def gen_random(w: int, h: int, density: float, rng: random.Random) -> list[str]:
    return ["".join("@" if rng.random() < density else "." for _ in range(w))
            for _ in range(h)]


def gen_maze(w: int, h: int, corridor: int, rng: random.Random) -> list[str]:
    """Recursive-backtracker maze on a coarse lattice, carved at corridor width."""
    cell = corridor + 1                       # corridor + wall
    gw, gh = max(2, (w - 1) // cell), max(2, (h - 1) // cell)
    grid = [["@"] * w for _ in range(h)]

    def carve(cx: int, cy: int) -> None:
        x0, y0 = 1 + cx * cell, 1 + cy * cell
        for dy in range(corridor):
            for dx in range(corridor):
                if y0 + dy < h and x0 + dx < w:
                    grid[y0 + dy][x0 + dx] = "."

    def carve_between(cx, cy, nx, ny) -> None:
        x0, y0 = 1 + cx * cell, 1 + cy * cell
        x1, y1 = 1 + nx * cell, 1 + ny * cell
        for dy in range(min(y0, y1), max(y0, y1) + corridor):
            for dx in range(min(x0, x1), max(x0, x1) + corridor):
                if 0 <= dy < h and 0 <= dx < w:
                    grid[dy][dx] = "."

    seen = [[False] * gw for _ in range(gh)]
    stack = [(0, 0)]
    seen[0][0] = True
    carve(0, 0)
    while stack:
        cx, cy = stack[-1]
        nbrs = [(cx + dx, cy + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= cx + dx < gw and 0 <= cy + dy < gh and not seen[cy + dy][cx + dx]]
        if not nbrs:
            stack.pop()
            continue
        nx, ny = rng.choice(nbrs)
        seen[ny][nx] = True
        carve(nx, ny)
        carve_between(cx, cy, nx, ny)
        stack.append((nx, ny))
    # sparse loops so mazes are not strictly tree-like
    for _ in range(int(0.03 * gw * gh)):
        grid[rng.randrange(1, h - 1)][rng.randrange(1, w - 1)] = "."
    return ["".join(r) for r in grid]


def gen_narrow(w: int, h: int, passage: int, clutter: float,
               rng: random.Random) -> list[str]:
    """Rooms separated by walls pierced by narrow passages; light clutter inside."""
    grid = [["." if rng.random() > clutter else "@" for _ in range(w)]
            for _ in range(h)]
    xs = sorted(rng.sample(range(w // 5, 4 * w // 5), rng.randint(2, 4)))
    for wx in xs:
        for y in range(h):
            grid[y][wx] = "@"
        for _ in range(rng.randint(1, 2)):        # 1-2 passages per wall
            py = rng.randrange(1, h - passage - 1)
            for dy in range(passage):
                grid[py + dy][wx] = "."
    if rng.random() < 0.5:                        # optional horizontal wall
        wy = rng.randrange(h // 4, 3 * h // 4)
        for x in range(w):
            grid[wy][x] = "@"
        for _ in range(rng.randint(1, 2)):
            px = rng.randrange(1, w - passage - 1)
            for dx in range(passage):
                grid[wy][px + dx] = "."
    return ["".join(r) for r in grid]


# ------------------------------- queries -----------------------------------

def sample_queries(gm, rows: list[str], n_queries: int, rng: random.Random,
                   min_frac: float = 0.30,
                   max_goal_tries: int | None = None,
                   starts_per_goal: int = 2) -> list[tuple]:
    """Generate valid queries with exact C* using batched reverse Dijkstra.

    The previous implementation ran a complete A* for every random pair,
    including rejected pairs.  Dense maps therefore spent most of their time
    proving that unsuitable pairs were disconnected or too close.  One reverse
    Dijkstra gives the exact cost from every reachable start to a selected goal,
    so several well-separated starts can be sampled from the same pass.
    """
    w, h = gm.width, gm.height
    free = [(x, y) for y in range(h) for x in range(w) if rows[y][x] == "."]
    if len(free) < 2:
        return []
    diag = (w * w + h * h) ** 0.5
    max_goal_tries = max_goal_tries or max(16, n_queries * 4)
    out: list[tuple] = []
    used: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    goals = rng.sample(free, min(len(free), max_goal_tries))

    for g in goals:
        distances = eng.dijkstra_all(gm, g[0], g[1], CONN)
        # Reservoir-sample only the starts that will be used.  Materializing
        # every eligible node costs hundreds of MB per worker on 1024/2048
        # maps and is unnecessary for a uniform random sample.
        wanted = min(starts_per_goal, n_queries - len(out))
        candidates: list[tuple[tuple[int, int], float]] = []
        eligible = 0
        for s in free:
            if s == g or (s, g) in used:
                continue
            cost = float(distances[s[1] * w + s[0]])
            if math.isfinite(cost) and cost >= min_frac * diag:
                eligible += 1
                if len(candidates) < wanted:
                    candidates.append((s, cost))
                else:
                    replacement = rng.randrange(eligible)
                    if replacement < wanted:
                        candidates[replacement] = (s, cost)

        # Reuse the reverse search without letting one goal dominate a map.
        for s, cost in candidates:
            used.add((s, g))
            out.append((s, g, cost))
        if len(out) >= n_queries:
            break
    return out


# ------------------------------- pipeline ----------------------------------

def make_map(family: str, level: dict, w: int, h: int,
             rng: random.Random) -> list[str]:
    if family == "random":
        return gen_random(w, h, level["density"], rng)
    if family == "maze":
        return gen_maze(w, h, level["corridor"], rng)
    return gen_narrow(w, h, level["passage"], level["clutter"], rng)


def density_of(rows: list[str]) -> float:
    tot = sum(len(r) for r in rows)
    return sum(r.count("@") for r in rows) / tot


def save_map(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"type octile\nheight {len(rows)}\nwidth {len(rows[0])}\nmap\n")
        f.write("\n".join(rows) + "\n")


def generate_map_task(task: dict) -> dict:
    """Worker entry point; all inputs/outputs are process-serializable."""
    retry_rng = random.Random(task["seed"])
    for attempt in range(1, task["max_map_attempts"] + 1):
        map_seed = retry_rng.randrange(2**31)
        rng = random.Random(map_seed)
        grid_rows = make_map(
            task["family"], task["level"], task["size"], task["size"], rng)
        gm = eng.GridMap.from_ascii(grid_rows)
        queries = sample_queries(
            gm, grid_rows, task["queries_per_map"],
            random.Random(map_seed + 1),
            starts_per_goal=task["starts_per_goal"])
        if len(queries) == task["queries_per_map"]:
            return {
                **task,
                "rows": grid_rows,
                "queries": queries,
                "density": round(density_of(grid_rows), 4),
                "attempts": attempt,
            }
    raise RuntimeError(
        f"could not generate {task['map_id']} after "
        f"{task['max_map_attempts']} map attempts")


def write_instances(split: str, records: list[dict]) -> Path:
    inst_dir = ROOT / "data" / "instances"
    inst_dir.mkdir(parents=True, exist_ok=True)
    fields = ["map_id", "map_path", "start_x", "start_y", "goal_x", "goal_y",
              "family", "density", "width", "height", "split", "optimal_cost"]
    path = inst_dir / f"{split}.csv"
    records.sort(key=lambda row: (
        row["map_id"], row["start_x"], row["start_y"],
        row["goal_x"], row["goal_y"]))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--maps-per-split", type=int, default=40)
    ap.add_argument("--queries-per-map", type=int, default=10)
    ap.add_argument(
        "--starts-per-goal", type=int, default=2,
        help="queries sharing one reverse-Dijkstra goal (lower = more goal diversity)")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--shift-size", type=int, default=128)
    ap.add_argument(
        "--splits", nargs="+",
        help="generate only selected splits (default: generate every split)")
    ap.add_argument(
        "--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)),
        help="parallel worker processes (use 1 to disable multiprocessing)")
    ap.add_argument(
        "--max-map-attempts", type=int, default=100,
        help="fail instead of rerolling a difficult map forever")
    args = ap.parse_args()

    if args.workers < 1:
        ap.error("--workers must be at least 1")
    if args.starts_per_goal < 1:
        ap.error("--starts-per-goal must be at least 1")
    if args.max_map_attempts < 1:
        ap.error("--max-map-attempts must be at least 1")

    master = random.Random(args.seed)
    S, SS, n = args.size, args.shift_size, args.maps_per_split
    train_lvls = [dict(density=d) for d in (0.10, 0.20, 0.30)]
    plan = [
        # in-distribution: random family, density 10-30%
        ("train", "random", train_lvls, S, n),
        # Structural training data.  The hardest corridor/passage settings are
        # deliberately reserved for shift_family below.
        ("train_structural", "maze",
         [dict(corridor=c) for c in (2, 3, 4)], S, max(8, n // 2)),
        ("train_structural", "narrow",
         [dict(passage=p, clutter=c) for p, c in
          ((2, .08), (3, .12), (4, .16))], S, max(8, n // 2)),
        ("val",   "random", train_lvls, S, max(8, n // 2)),
        ("val_structural", "maze",
         [dict(corridor=c) for c in (2, 3)], S, max(4, n // 4)),
        ("val_structural", "narrow",
         [dict(passage=p, clutter=c) for p, c in ((2, .10), (3, .14))],
         S, max(4, n // 4)),
        ("test",  "random", train_lvls, S, max(8, n // 2)),
        # shift 1: obstacle density
        ("shift_density", "random",
         [dict(density=d) for d in (0.35, 0.45, 0.50)], S, max(8, n // 2)),
        # shift 2: map size
        ("shift_size", "random",
         [dict(density=d) for d in (0.20, 0.30, 0.40)], SS, max(6, n // 3)),
        # shift 3: held-out HARD structural parameters.  Maze/narrow families
        # are seen during structural training, but corridor=1 / passage=1 and
        # the heavier clutter levels remain test-only.
        ("shift_family", "maze",
         [dict(corridor=1)], S, max(6, n // 3)),
        ("shift_family", "narrow",
         [dict(passage=1, clutter=c) for c in (.18, .22, .26)],
         S, max(6, n // 3)),
        # calibration proxy: MILD, disjoint-seed geometry shift used ONLY to
        # calibrate ensemble-variance -> confidence mapping (Section "variance
        # calibration"). Never touches training weights, never touches the
        # shift_family TEST set above -- different corridor/passage levels and
        # a different map_id range guarantee no overlap.
        ("val_shift", "maze",
         [dict(corridor=c) for c in (2, 3)], S, max(4, n // 5)),
        ("val_shift", "narrow",
         [dict(passage=p, clutter=c) for p, c in ((2, .10), (3, .12))],
         S, max(4, n // 5)),
    ]

    tasks_by_split: dict[str, list[dict]] = {}
    map_counter = 0
    for split, family, levels, size, n_maps in plan:
        for index in range(n_maps):
            map_id = f"{family}_{split}_{map_counter:04d}"
            map_counter += 1
            tasks_by_split.setdefault(split, []).append({
                "split": split,
                "family": family,
                "level": levels[index % len(levels)],
                "size": size,
                "map_id": map_id,
                "seed": master.randrange(2**31),
                "queries_per_map": args.queries_per_map,
                "starts_per_goal": args.starts_per_goal,
                "max_map_attempts": args.max_map_attempts,
            })

    if args.splits:
        known = set(tasks_by_split)
        unknown = sorted(set(args.splits) - known)
        if unknown:
            ap.error(f"unknown splits: {unknown}; choose from {sorted(known)}")
        selected = set(args.splits)
        tasks_by_split = {
            split: tasks for split, tasks in tasks_by_split.items()
            if split in selected
        }

    rows_out: dict[str, list[dict]] = {}
    print(f"workers: {args.workers}")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for split, tasks in tasks_by_split.items():
            records: list[dict] = []
            family_counts: dict[str, int] = {}
            futures = [executor.submit(generate_map_task, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                family = result["family"]
                family_counts[family] = family_counts.get(family, 0) + 1
                rel = (Path("data/synthetic") / family / split /
                       f"{result['map_id']}.map")
                save_map(ROOT / rel, result["rows"])
                for (sx, sy), (gx, gy), cstar in result["queries"]:
                    records.append(dict(
                        map_id=result["map_id"], map_path=str(rel),
                        start_x=sx, start_y=sy, goal_x=gx, goal_y=gy,
                        family=family, density=result["density"],
                        width=result["size"], height=result["size"], split=split,
                        optimal_cost=round(cstar, 6)))
                if completed == 1 or completed % 10 == 0 or completed == len(tasks):
                    print(f"[{split:13s}] {completed:4d}/{len(tasks):4d} maps",
                          flush=True)

            rows_out[split] = records
            path = write_instances(split, records)
            summary = ", ".join(
                f"{family}={count}" for family, count in sorted(family_counts.items()))
            print(f"wrote {path} ({len(records)} instances; {summary})",
                  flush=True)

    # sanity: map-level split disjointness (no leakage)
    ids = {s: {r["map_id"] for r in recs} for s, recs in rows_out.items()}
    keys = list(ids)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            assert not (ids[a] & ids[b]), f"map leakage between {a} and {b}"
    print("map-level split disjointness: OK")


if __name__ == "__main__":
    main()
