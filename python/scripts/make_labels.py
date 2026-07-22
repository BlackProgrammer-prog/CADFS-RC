"""Build cost-to-go training labels with the C++ engine.

For every (map, goal) pair in an instance split, run dijkstra_all from the goal
(one pass gives d*(cell, goal) for ALL cells) and sample reachable cells.

Output: data/labels/<split>.npz with arrays
  patch  (N, 1, P, P) float32   local occupancy window (out-of-bounds = 1)
  extra  (N, 4)       float32   [dx/diag, dy/diag, euclid/diag, octile/diag]
  y      (N,)         float32   d*(cell, goal) / diag   (normalized cost-to-go)
  meta   (N, 3)       int32     [map_index, cell_id, goal_id]
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build"))
import cadfs_engine as eng  # noqa: E402

PATCH = 15
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
    out = np.ones((PATCH, PATCH), dtype=np.float32)  # OOB = obstacle
    x0, x1 = max(0, x - R), min(w, x + R + 1)
    y0, y1 = max(0, y - R), min(h, y + R + 1)
    out[y0 - (y - R):y1 - (y - R), x0 - (x - R):x1 - (x - R)] = occ[y0:y1, x0:x1]
    return out


def extra_feats(x, y, gx, gy, diag) -> np.ndarray:
    dx, dy = gx - x, gy - y
    eu = math.hypot(dx, dy)
    adx, ady = abs(dx), abs(dy)
    octile = (adx + ady) + (SQRT2 - 2.0) * min(adx, ady)
    return np.array([dx / diag, dy / diag, eu / diag, octile / diag],
                    dtype=np.float32)


def build_split(split: str, samples_per_goal: int, seed: int) -> None:
    rows = list(csv.DictReader(open(ROOT / "data/instances" / f"{split}.csv")))
    # unique (map, goal) pairs; one dijkstra_all per pair
    pairs: dict[tuple, dict] = {}
    for r in rows:
        pairs.setdefault((r["map_path"], int(r["goal_x"]), int(r["goal_y"])), r)

    rng = random.Random(seed)
    P, X, Y, M = [], [], [], []
    occ_cache: dict[str, tuple] = {}
    for mi, ((mp, gx, gy), _) in enumerate(sorted(pairs.items())):
        if mp not in occ_cache:
            gm = eng.GridMap.load_movingai(str(ROOT / mp))
            occ_cache[mp] = (gm, occupancy(gm))
        gm, occ = occ_cache[mp]
        w, h = gm.width, gm.height
        diag = math.hypot(w, h)
        d = np.asarray(eng.dijkstra_all(gm, gx, gy, 8))
        reachable = np.flatnonzero(np.isfinite(d))
        if len(reachable) == 0:
            continue
        take = rng.sample(list(reachable),
                          min(samples_per_goal, len(reachable)))
        for cid in take:
            x, y = int(cid % w), int(cid // w)
            P.append(extract_patch(occ, x, y))
            X.append(extra_feats(x, y, gx, gy, diag))
            Y.append(d[cid] / diag)
            M.append([mi, cid, gy * w + gx])

    out = ROOT / "data/labels"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{split}.npz",
        patch=np.stack(P)[:, None, :, :], extra=np.stack(X),
        y=np.asarray(Y, dtype=np.float32), meta=np.asarray(M, dtype=np.int32))
    print(f"{split:14s}: {len(Y):7d} samples from {len(pairs)} (map,goal) pairs "
          f"-> {out / f'{split}.npz'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+",
                    default=["train", "val", "test",
                             "shift_density", "shift_size", "shift_family"])
    ap.add_argument("--samples-per-goal", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    for s in args.splits:
        build_split(s, args.samples_per_goal, args.seed)
