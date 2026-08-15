"""Fast bound/smoke check for CADFS Next on validation instances."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from cadfs_py import load_engine  # noqa: E402
from cadfs_py.experiments import (  # noqa: E402
    build_methods,
    load_settings,
    select_methods,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument(
        "--methods", nargs="+",
        default=["learn_focal_wstar", "cadfs", "cadfs_next"])
    parser.add_argument(
        "--guidance", choices=["auto", "fast", "cnn", "cnn-adaptive"],
        default="auto")
    parser.add_argument("--early-exit-members", type=int, default=2)
    parser.add_argument("--early-exit-variance", type=float, default=0.01)
    args = parser.parse_args()

    engine = load_engine(required=("run_cadfs_next",))
    tuned_path = ROOT / "results/models/tuned_next.json"
    legacy, next_settings = load_settings(ROOT, tuned_path)
    if not tuned_path.exists():
        print("[smoke] tuned_next.json not found; using documented defaults")
    fast_path = ROOT / "results/models/fast_ensemble.txt"
    guidance_name = args.guidance
    if guidance_name == "auto":
        guidance_name = "fast" if fast_path.exists() else "cnn"
    if guidance_name == "fast":
        if not fast_path.exists():
            raise FileNotFoundError(
                f"{fast_path} is missing; run train_student.py")
        ensemble = engine.FastEnsembleGuidance(str(fast_path))
    elif guidance_name == "cnn-adaptive":
        ensemble = engine.EnsembleGuidance(
            str(ROOT / "results/models/ensemble.txt"),
            args.early_exit_members, args.early_exit_variance)
    else:
        ensemble = engine.EnsembleGuidance(
            str(ROOT / "results/models/ensemble.txt"))
    available = build_methods(engine, ensemble, legacy, next_settings)
    methods = select_methods(available, "main", args.methods)

    with (ROOT / "data/instances" / f"{args.split}.csv").open(
            newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))[:args.instances]

    print(f"engine: {engine.__file__}")
    print(f"{'instance':30s} {'method':24s} {'exp':>8s} "
          f"{'runtime':>10s} {'model':>10s} {'ratio':>8s} {'mean_w':>8s}")
    for row in rows:
        map_ = engine.GridMap.load_movingai(str(ROOT / row["map_path"]))
        start = (int(row["start_x"]), int(row["start_y"]))
        goal = (int(row["goal_x"]), int(row["goal_y"]))
        optimal = float(row["optimal_cost"])
        config = dict(
            legacy["base"], theta_c=legacy["theta_c"],
            h_max=math.hypot(map_.width, map_.height))
        for name, method in methods.items():
            result = method(map_, start, goal, config)
            if not result["found"]:
                raise AssertionError(f"{name} failed on {row['map_id']}")
            ratio = result["cost"] / optimal
            if ratio > float(config["W"]) + 1e-9:
                raise AssertionError(
                    f"bound violation: {name} {row['map_id']} ratio={ratio}")
            print(f"{row['map_id']:30s} {name:24s} "
                  f"{result['expansions']:8d} {result['runtime_ms']:10.3f} "
                  f"{result['model_eval_time_ms']:10.3f} "
                  f"{ratio:8.4f} {result['mean_w']:8.4f}")


if __name__ == "__main__":
    main()
