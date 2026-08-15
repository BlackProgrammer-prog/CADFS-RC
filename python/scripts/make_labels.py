"""Build cost-to-go training labels with the C++ engine.

For every (map, goal) pair in an instance split, run dijkstra_all from the goal
(one pass gives d*(cell, goal) for ALL cells) and sample reachable cells.

Output: data/labels/<split>.npz with arrays
  patch  (N, 1, P, P) uint8     local occupancy window (out-of-bounds = 1)
  extra  (N, 10)      float32   goal geometry + multi-scale structure
  y      (N,)         float32   d*(cell, goal) / diag   (normalized cost-to-go)
  meta   (N, 3)       int32     [map_index, cell_id, goal_id]
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

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build"))
import cadfs_engine as eng  # noqa: E402

PATCH = 31
R = PATCH // 2
SQRT2 = math.sqrt(2.0)


def occupancy(gm) -> np.ndarray:
    w, h = gm.width, gm.height
    occ = np.ones((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            if gm.passable(x, y):
                occ[y, x] = 0.0
    return occ


def extract_patch(occ: np.ndarray, x: int, y: int) -> np.ndarray:
    h, w = occ.shape
    out = np.ones((PATCH, PATCH), dtype=np.uint8)  # OOB = obstacle
    x0, x1 = max(0, x - R), min(w, x + R + 1)
    y0, y1 = max(0, y - R), min(h, y + R + 1)
    out[y0 - (y - R):y1 - (y - R), x0 - (x - R):x1 - (x - R)] = occ[y0:y1, x0:x1]
    return out


def density_at(occ: np.ndarray, x: int, y: int, radius: int) -> float:
    h, w = occ.shape
    blocked = 0
    total = (2 * radius + 1) ** 2
    for yy in range(y - radius, y + radius + 1):
        for xx in range(x - radius, x + radius + 1):
            if not (0 <= xx < w and 0 <= yy < h) or occ[yy, xx] > 0.5:
                blocked += 1
    return blocked / total


def degree8(occ: np.ndarray, x: int, y: int) -> int:
    h, w = occ.shape

    def free(xx: int, yy: int) -> bool:
        return 0 <= xx < w and 0 <= yy < h and occ[yy, xx] < 0.5

    degree = sum(free(x + dx, y + dy)
                 for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        degree += free(x + dx, y + dy) and free(x + dx, y) and free(x, y + dy)
    return int(degree)


def line_obstacle_fraction(occ: np.ndarray, x0: int, y0: int,
                           x1: int, y1: int) -> float:
    """Bresenham occupancy fraction, including the passable endpoints."""
    dx, sx = abs(x1 - x0), 1 if x0 < x1 else -1
    dy, sy = -abs(y1 - y0), 1 if y0 < y1 else -1
    error = dx + dy
    blocked = total = 0
    while True:
        total += 1
        blocked += int(occ[y0, x0] > 0.5)
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy
    return blocked / max(1, total)


def extra_feats(occ: np.ndarray, x: int, y: int,
                gx: int, gy: int, diag: float) -> np.ndarray:
    dx, dy = gx - x, gy - y
    eu = math.hypot(dx, dy)
    adx, ady = abs(dx), abs(dy)
    octile = (adx + ady) + (SQRT2 - 2.0) * min(adx, ady)
    return np.array([
        dx / diag,
        dy / diag,
        eu / diag,
        octile / diag,
        line_obstacle_fraction(occ, x, y, gx, gy),
        density_at(occ, x, y, 3),
        density_at(occ, x, y, 7),
        density_at(occ, x, y, 15),
        degree8(occ, x, y) / 8.0,
        density_at(occ, gx, gy, 3),
    ], dtype=np.float32)


def build_map_labels(task: dict) -> tuple[int, np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray]:
    """Build every goal for one map in a worker process."""
    gm = eng.GridMap.load_movingai(str(ROOT / task["map_path"]))
    occ = occupancy(gm)
    w, h = gm.width, gm.height
    diag = math.hypot(w, h)
    patches, extras, targets, metadata = [], [], [], []

    for pair_index, gx, gy in task["goals"]:
        distances = np.asarray(eng.dijkstra_all(gm, gx, gy, 8))
        reachable = np.flatnonzero(np.isfinite(distances))
        if len(reachable) == 0:
            continue
        rng = random.Random(f"{task['seed']}:{task['split']}:{pair_index}")
        take = rng.sample(
            list(reachable), min(task["samples_per_goal"], len(reachable)))
        for cid in take:
            x, y = int(cid % w), int(cid // w)
            patches.append(extract_patch(occ, x, y))
            extras.append(extra_feats(occ, x, y, gx, gy, diag))
            targets.append(distances[cid] / diag)
            metadata.append([pair_index, cid, gy * w + gx])

    if not targets:
        return (task["map_order"],
                np.empty((0, 1, PATCH, PATCH), dtype=np.uint8),
                np.empty((0, 10), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0, 3), dtype=np.int32))
    return (
        task["map_order"],
        np.stack(patches)[:, None, :, :],
        np.stack(extras),
        np.asarray(targets, dtype=np.float32),
        np.asarray(metadata, dtype=np.int32),
    )


def build_split(split: str, samples_per_goal: int, seed: int,
                workers: int) -> None:
    rows = list(csv.DictReader(open(ROOT / "data/instances" / f"{split}.csv")))
    # unique (map, goal) pairs; one dijkstra_all per pair
    pairs: dict[tuple, dict] = {}
    for r in rows:
        pairs.setdefault((r["map_path"], int(r["goal_x"]), int(r["goal_y"])), r)

    grouped: dict[str, list[tuple[int, int, int]]] = {}
    for pair_index, ((map_path, gx, gy), _) in enumerate(sorted(pairs.items())):
        grouped.setdefault(map_path, []).append((pair_index, gx, gy))

    tasks = [
        {
            "map_order": map_order,
            "map_path": map_path,
            "goals": goals,
            "samples_per_goal": samples_per_goal,
            "seed": seed,
            "split": split,
        }
        for map_order, (map_path, goals) in enumerate(sorted(grouped.items()))
    ]
    chunks: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(build_map_labels, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            order, patch, extra, target, meta = future.result()
            chunks[order] = (patch, extra, target, meta)
            if completed == 1 or completed % 10 == 0 or completed == len(tasks):
                print(f"[{split:16s}] {completed:4d}/{len(tasks):4d} maps",
                      flush=True)

    ordered = [chunks[index] for index in sorted(chunks)]
    P = np.concatenate([chunk[0] for chunk in ordered], axis=0)
    X = np.concatenate([chunk[1] for chunk in ordered], axis=0)
    Y = np.concatenate([chunk[2] for chunk in ordered], axis=0)
    M = np.concatenate([chunk[3] for chunk in ordered], axis=0)

    out = ROOT / "data/labels"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{split}.npz",
        patch=P, extra=X, y=Y, meta=M,
        schema_version=np.asarray(2, dtype=np.int32),
        patch_size=np.asarray(PATCH, dtype=np.int32),
        target_name=np.asarray("normalized_cost_to_go", dtype="U32"))
    print(f"{split:14s}: {len(Y):7d} samples from {len(pairs)} (map,goal) pairs "
          f"-> {out / f'{split}.npz'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+",
                    default=["train", "train_structural", "val",
                             "val_structural", "val_shift", "test",
                             "shift_density", "shift_size", "shift_family"])
    ap.add_argument("--samples-per-goal", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--workers", type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers must be at least 1")
    if args.samples_per_goal < 1:
        ap.error("--samples-per-goal must be at least 1")
    for s in args.splits:
        build_split(s, args.samples_per_goal, args.seed, args.workers)
