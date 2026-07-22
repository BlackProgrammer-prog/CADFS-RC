"""Run the full benchmark: baselines + ablations on test and shift splits.

Per-instance log rows -> results/logs/bench.csv with columns:
  split,family,density,map_id,instance,method,found,cost,cstar,ratio,
  expansions,runtime_ms,fallback_rate,mean_w,min_w,max_w,mean_abs_dw,mean_C,mean_R

Methods:
  astar, wastar            (weight = W)
  focal_plain              fixed focal, no learning (secondary = h_a)
  learn_focal_W            learning-guided fixed focal, width W
  learn_focal_wstar        learning-guided fixed focal, tuned width w*  <- key control
  cadfs                    full CADFS (multiplicative controller)
  cadfs_linear             linear controller baseline
  cadfs_norisk             risk_mode = off
  cadfs_randomrisk         risk_mode = random
  cadfs_permutedrisk       risk_mode = permuted
  cadfs_nofallback         fallback disabled
  cadfs_noconf             confidence forced to 1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build"))
import cadfs_engine as eng  # noqa: E402

TUNED = json.load(open(ROOT / "results/models/tuned.json"))
BASE = dict(TUNED["base"])
BASE["theta_c"] = TUNED["theta_c"]
TAU = TUNED["tau_c"]
W_STAR = TUNED["w_star"]
W = BASE["W"]


def methods(ens):
    def cadfs_variant(**over):
        def run(m, s, g, cfg):
            c = dict(cfg, **over)
            return eng.run_cadfs(m, s, g, c, ens, TAU)
        return run
    return {
        "astar":            lambda m, s, g, cfg: eng.run_astar(m, s, g, cfg, 1.0),
        "wastar":           lambda m, s, g, cfg: eng.run_astar(m, s, g, cfg, W),
        "focal_plain":      lambda m, s, g, cfg: eng.run_focal(m, s, g, cfg, None, W),
        "learn_focal_W":    lambda m, s, g, cfg: eng.run_focal(m, s, g, cfg, ens, W),
        "learn_focal_wstar":lambda m, s, g, cfg: eng.run_focal(m, s, g, cfg, ens, W_STAR),
        "cadfs":            cadfs_variant(),
        "cadfs_linear":     cadfs_variant(controller="linear",
                                          lin_a=1/3, lin_b=1/3, lin_c=1/3),
        "cadfs_norisk":     cadfs_variant(risk_mode="off"),
        "cadfs_randomrisk": cadfs_variant(risk_mode="random", risk_seed=11),
        "cadfs_permutedrisk": cadfs_variant(risk_mode="permuted"),
        "cadfs_nofallback": cadfs_variant(fallback_enabled=False),
        "cadfs_noconf":     cadfs_variant(confidence_enabled=False),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+",
                    default=["test", "shift_density", "shift_size", "shift_family"])
    ap.add_argument("--per-split", type=int, default=40)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="results/logs/bench.csv")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    ens = eng.EnsembleGuidance(str(ROOT / "results/models/ensemble.txt"))
    meths = methods(ens)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append and out_path.exists() else "w"
    fields = ["split", "family", "density", "map_id", "instance", "method",
              "found", "cost", "cstar", "ratio", "expansions", "runtime_ms",
              "fallback_rate", "mean_w", "min_w", "max_w", "mean_abs_dw",
              "mean_C", "mean_R"]
    fout = open(out_path, mode, newline="")
    wcsv = csv.DictWriter(fout, fieldnames=fields)
    if mode == "w":
        wcsv.writeheader()

    for split in args.splits:
        rows = list(csv.DictReader(open(ROOT / "data/instances" / f"{split}.csv")))
        random.Random(args.seed).shuffle(rows)
        rows = rows[:args.per_split]
        t0 = time.time()
        for i, r in enumerate(rows):
            m = eng.GridMap.load_movingai(str(ROOT / r["map_path"]))
            s = (int(r["start_x"]), int(r["start_y"]))
            g = (int(r["goal_x"]), int(r["goal_y"]))
            cstar = float(r["optimal_cost"])
            cfg = dict(BASE, h_max=math.hypot(m.width, m.height))
            for name, fn in meths.items():
                res = fn(m, s, g, cfg)
                assert (not res["found"]) or res["cost"] <= W * cstar + 1e-9, \
                    f"BOUND VIOLATION {name} {r['map_id']}"
                wcsv.writerow(dict(
                    split=split, family=r["family"], density=r["density"],
                    map_id=r["map_id"], instance=i, method=name,
                    found=int(res["found"]), cost=round(res["cost"], 6),
                    cstar=cstar, ratio=round(res["cost"] / cstar, 6),
                    expansions=res["expansions"],
                    runtime_ms=round(res["runtime_ms"], 3),
                    fallback_rate=round(res["fallback_rate"], 4),
                    mean_w=round(res["mean_w"], 4), min_w=round(res["min_w"], 4),
                    max_w=round(res["max_w"], 4),
                    mean_abs_dw=round(res["mean_abs_dw"], 5),
                    mean_C=round(res["mean_C"], 4), mean_R=round(res["mean_R"], 4)))
            fout.flush()
        print(f"[{split}] {len(rows)} instances x {len(meths)} methods "
              f"in {time.time() - t0:.0f}s")
    fout.close()
    print(f"log -> {out_path}")


if __name__ == "__main__":
    main()
