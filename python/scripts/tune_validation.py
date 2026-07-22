"""Select hyperparameters on the VALIDATION split only (paper Sec. 11.6).

Tunes:
  w*        best fixed width for the learning-guided focal baseline
  tau_c     confidence temperature
  theta_c   fallback confidence threshold
Fixed by convention (stated in the paper): lambda_* = 1/3 each, alpha=0.7,
beta=0.3, theta_r=0.8, theta_dev=0.5, L=16, K=50, W=2.
Normalization rule: h_min=0, h_max = map diagonal (per map, rule fixed here).

Output: results/models/tuned.json
"""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build"))
import cadfs_engine as eng  # noqa: E402

BASE = dict(W=2.0, L=16, K=50, connectivity=8, alpha=1.0, beta=0.0,
            lambda_obs=1 / 3, lambda_mob=1 / 3, lambda_dev=1 / 3,
            theta_r=0.8, theta_dev=0.5, h_min=0.0)


def load_val(n: int, seed: int = 0):
    rows = list(csv.DictReader(open(ROOT / "data/instances/val.csv")))
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows[:n]:
        m = eng.GridMap.load_movingai(str(ROOT / r["map_path"]))
        out.append((m, (int(r["start_x"]), int(r["start_y"])),
                    (int(r["goal_x"]), int(r["goal_y"])),
                    float(r["optimal_cost"]),
                    math.hypot(m.width, m.height)))
    return out


def main() -> None:
    ens = eng.EnsembleGuidance(str(ROOT / "results/models/ensemble.txt"))
    val = load_val(40)

    # ---- 1) w* for the tuned fixed-width learning-guided baseline ----
    best_w, best_e = None, float("inf")
    for w in (1.1, 1.25, 1.5, 1.75, 2.0):
        exps = []
        for m, s, g, _, diag in val:
            cfg = dict(BASE, h_max=diag)
            exps.append(eng.run_focal(m, s, g, cfg, ens, w)["expansions"])
        mu = statistics.mean(exps)
        print(f"  w*={w:4.2f}  mean expansions {mu:8.1f}")
        if mu < best_e:
            best_e, best_w = mu, w

    # ---- 2) (tau_c, theta_c) for CADFS ----
    best = None
    for tau in (1e-4, 5e-4, 2e-3):
        for th_c in (0.2, 0.35, 0.5):
            exps = []
            for m, s, g, _, diag in val:
                cfg = dict(BASE, h_max=diag, theta_c=th_c)
                exps.append(eng.run_cadfs(m, s, g, cfg, ens, tau)["expansions"])
            mu = statistics.mean(exps)
            print(f"  tau_c={tau:.0e} theta_c={th_c:.2f}  mean exp {mu:8.1f}")
            if best is None or mu < best[0]:
                best = (mu, tau, th_c)

    tuned = dict(w_star=best_w, tau_c=best[1], theta_c=best[2],
                 base=BASE, h_max_rule="map_diagonal",
                 val_mean_exp_wstar=best_e, val_mean_exp_cadfs=best[0])
    out = ROOT / "results/models/tuned.json"
    json.dump(tuned, open(out, "w"), indent=2)
    print(f"\nselected: w*={best_w}, tau_c={best[1]:.0e}, theta_c={best[2]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
