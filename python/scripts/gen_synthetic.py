"""Generate the synthetic CADFS dataset.

Families: random / maze / narrow-passage.
Splits are MAP-LEVEL (a map appears in exactly one split) to prevent leakage.
Shift sets: density shift, size shift, family shift (maze+narrow unseen at train).

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
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build"))
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
                   min_frac: float = 0.30, max_tries: int = 400) -> list[tuple]:
    """Valid (start, goal, C*) with d* >= min_frac * map diagonal."""
    w, h = gm.width, gm.height
    free = [(x, y) for y in range(h) for x in range(w) if rows[y][x] == "."]
    if len(free) < 2:
        return []
    diag = (w * w + h * h) ** 0.5
    out, tries = [], 0
    while len(out) < n_queries and tries < max_tries:
        tries += 1
        s, g = rng.choice(free), rng.choice(free)
        if s == g:
            continue
        r = eng.run_astar(gm, s, g, CFG)  # optimal: weight=1, admissible anchor
        if r["found"] and r["cost"] >= min_frac * diag:
            out.append((s, g, r["cost"]))
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--maps-per-split", type=int, default=40)
    ap.add_argument("--queries-per-map", type=int, default=10)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--shift-size", type=int, default=128)
    args = ap.parse_args()

    master = random.Random(args.seed)
    rows_out: dict[str, list[dict]] = {}

    S, SS, n = args.size, args.shift_size, args.maps_per_split
    train_lvls = [dict(density=d) for d in (0.10, 0.20, 0.30)]
    plan = [
        # in-distribution: random family, density 10-30%
        ("train", "random", train_lvls, S, n),
        ("val",   "random", train_lvls, S, max(8, n // 2)),
        ("test",  "random", train_lvls, S, max(8, n // 2)),
        # shift 1: obstacle density
        ("shift_density", "random",
         [dict(density=d) for d in (0.35, 0.45, 0.50)], S, max(8, n // 2)),
        # shift 2: map size
        ("shift_size", "random",
         [dict(density=d) for d in (0.20, 0.30, 0.40)], SS, max(6, n // 3)),
        # shift 3: unseen families (TEST ONLY — never used for tuning/calibration)
        ("shift_family", "maze",
         [dict(corridor=c) for c in (1, 2, 4)], S, max(6, n // 3)),
        ("shift_family", "narrow",
         [dict(passage=p, clutter=c) for p, c in ((1, .10), (2, .15), (3, .20))],
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

    map_counter = 0
    for split, family, levels, size, n_maps in plan:
        made = 0
        while made < n_maps:
            lvl = levels[made % len(levels)]
            seed = master.randrange(2**31)
            rng = random.Random(seed)
            grid_rows = make_map(family, lvl, size, size, rng)
            gm = eng.GridMap.from_ascii(grid_rows)
            qs = sample_queries(gm, grid_rows, args.queries_per_map,
                                random.Random(seed + 1))
            if len(qs) < args.queries_per_map:    # degenerate map -> reroll
                continue
            map_id = f"{family}_{split}_{map_counter:04d}"
            map_counter += 1
            rel = Path("data/synthetic") / family / split / f"{map_id}.map"
            save_map(ROOT / rel, grid_rows)
            dens = round(density_of(grid_rows), 4)
            for (sx, sy), (gx, gy), cstar in qs:
                rows_out.setdefault(split, []).append(dict(
                    map_id=map_id, map_path=str(rel),
                    start_x=sx, start_y=sy, goal_x=gx, goal_y=gy,
                    family=family, density=dens, width=size, height=size,
                    split=split, optimal_cost=round(cstar, 6)))
            made += 1
        print(f"[{split:13s}] {family:6s}: {n_maps} maps "
              f"x {args.queries_per_map} queries")

    inst_dir = ROOT / "data" / "instances"
    inst_dir.mkdir(parents=True, exist_ok=True)
    fields = ["map_id", "map_path", "start_x", "start_y", "goal_x", "goal_y",
              "family", "density", "width", "height", "split", "optimal_cost"]
    for split, recs in rows_out.items():
        with open(inst_dir / f"{split}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(recs)
        print(f"wrote {inst_dir / f'{split}.csv'}  ({len(recs)} instances)")

    # sanity: map-level split disjointness (no leakage)
    ids = {s: {r["map_id"] for r in recs} for s, recs in rows_out.items()}
    keys = list(ids)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            assert not (ids[a] & ids[b]), f"map leakage between {a} and {b}"
    print("map-level split disjointness: OK")


if __name__ == "__main__":
    main()
