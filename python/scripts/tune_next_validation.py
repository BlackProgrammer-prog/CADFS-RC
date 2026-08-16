"""Tune CADFS-Next in stages using validation data only.

The selected expert fusion, confidence estimator, and controller are written
to ``results/models/tuned_next.json`` together with every audited trial.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from itertools import product
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from cadfs_py import load_engine  # noqa: E402
from cadfs_py.experiments import DEFAULT_NEXT, load_json, next_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits", nargs="+",
        default=["val", "val_structural", "val_shift"])
    parser.add_argument("--per-split", type=int, default=20)
    parser.add_argument(
        "--workers", type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
        help="parallel C++ searches per candidate (use 1 for serial timing)")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", default="results/models/tuned_next.json")
    parser.add_argument(
        "--out-conservative",
        default="results/models/tuned_next_conservative.json")
    parser.add_argument(
        "--out-tail",
        help="optional third profile selected for split-wise tail robustness")
    parser.add_argument(
        "--guidance", choices=["auto", "fast", "cnn", "cnn-adaptive"],
        default="auto")
    parser.add_argument(
        "--guidance-model",
        help="versioned guidance export; defaults to the standard model path")
    parser.add_argument(
        "--guidance-region-radius", type=int, default=0,
        help="0 keeps exact per-node inference; r>0 enables regional residual reuse")
    parser.add_argument("--early-exit-members", type=int, default=2)
    parser.add_argument("--early-exit-variance", type=float, default=0.01)
    parser.add_argument(
        "--max-candidates", type=int, default=0,
        help="deterministic development cap; 0 evaluates the full joint grid")
    parser.add_argument(
        "--conservative-max-ratio", type=float, default=1.30,
        help="validation worst-case quality gate for the conservative profile")
    return parser.parse_args()


def load_validation(engine, splits: list[str], per_split: int, seed: int):
    instances = []
    for split in splits:
        path = ROOT / "data/instances" / f"{split}.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        random.Random(f"{seed}:{split}").shuffle(rows)
        for row in rows[:per_split]:
            map_ = engine.GridMap.load_movingai(str(ROOT / row["map_path"]))
            instances.append({
                "split": split,
                "map": map_,
                "start": (int(row["start_x"]), int(row["start_y"])),
                "goal": (int(row["goal_x"]), int(row["goal_y"])),
                "optimal": float(row["optimal_cost"]),
                "diagonal": math.hypot(map_.width, map_.height),
            })
    return instances


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_ratios(values: list[float]) -> dict:
    threshold = percentile(values, .95)
    tail = [value for value in values if value >= threshold]
    return {
        "mean_ratio": statistics.mean(values),
        "p95_ratio": threshold,
        "p99_ratio": percentile(values, .99),
        "cvar95_ratio": statistics.mean(tail),
        "max_ratio": max(values),
    }


def evaluate(engine, ensemble, instances, base: dict, settings: dict,
             workers: int) -> dict:
    expansions: list[int] = []
    runtimes: list[float] = []
    ratios: list[float] = []
    fallback_rates: list[float] = []
    mean_widths: list[float] = []
    model_times: list[float] = []
    split_values: dict[str, dict[str, list[float]]] = {}
    maximum_width = float(base["W"])

    def run_one(item):
        config = next_config(
            dict(base, h_max=item["diagonal"]), settings)
        result = engine.run_cadfs_next(
            item["map"], item["start"], item["goal"], config, ensemble)
        if not result["found"]:
            raise AssertionError(f"search failed on validation split {item['split']}")
        ratio = result["cost"] / item["optimal"]
        if ratio > maximum_width + 1e-9:
            raise AssertionError(f"validation bound violation: ratio={ratio}")
        return (
            item["split"], result["expansions"], result["runtime_ms"], ratio,
            result["fallback_rate"], result["mean_w"],
            result["model_eval_time_ms"])

    started = time.perf_counter()
    if workers == 1:
        results = map(run_one, instances)
        for (split, expanded, runtime, ratio, fallback, width,
             model_time) in results:
            expansions.append(expanded)
            runtimes.append(runtime)
            ratios.append(ratio)
            fallback_rates.append(fallback)
            mean_widths.append(width)
            model_times.append(model_time)
            values = split_values.setdefault(
                split, {"ratios": [], "expansions": [], "runtimes": []})
            values["ratios"].append(ratio)
            values["expansions"].append(expanded)
            values["runtimes"].append(runtime)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for (split, expanded, runtime, ratio, fallback, width,
                 model_time) in executor.map(run_one, instances):
                expansions.append(expanded)
                runtimes.append(runtime)
                ratios.append(ratio)
                fallback_rates.append(fallback)
                mean_widths.append(width)
                model_times.append(model_time)
                values = split_values.setdefault(
                    split, {"ratios": [], "expansions": [], "runtimes": []})
                values["ratios"].append(ratio)
                values["expansions"].append(expanded)
                values["runtimes"].append(runtime)
    wall_seconds = time.perf_counter() - started

    return {
        "mean_expansions": statistics.mean(expansions),
        "median_expansions": statistics.median(expansions),
        "mean_runtime_ms": statistics.mean(runtimes),
        "mean_ratio": statistics.mean(ratios),
        "max_ratio": max(ratios),
        "mean_fallback_rate": statistics.mean(fallback_rates),
        "mean_focal_width": statistics.mean(mean_widths),
        "mean_model_eval_time_ms": statistics.mean(model_times),
        "wall_seconds": wall_seconds,
        "workers": workers,
        "split_metrics": {
            split: {
                **summarize_ratios(values["ratios"]),
                "mean_expansions": statistics.mean(values["expansions"]),
                "mean_runtime_ms": statistics.mean(values["runtimes"]),
            }
            for split, values in sorted(split_values.items())
        },
    }


def evaluate_wastar_tail(engine, instances, base: dict, workers: int) -> dict:
    width = float(base["W"])

    def run_one(item):
        config = dict(base, h_max=item["diagonal"])
        result = engine.run_astar(
            item["map"], item["start"], item["goal"], config, width)
        if not result["found"]:
            raise AssertionError(
                f"Weighted A* failed on validation split {item['split']}")
        return item["split"], result["cost"] / item["optimal"]

    grouped: dict[str, list[float]] = {}
    if workers == 1:
        results = map(run_one, instances)
        for split, ratio in results:
            grouped.setdefault(split, []).append(ratio)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for split, ratio in executor.map(run_one, instances):
                grouped.setdefault(split, []).append(ratio)
    return {
        split: summarize_ratios(values)
        for split, values in sorted(grouped.items())
    }


def select_stage(stage: str, candidates: list[tuple[str, dict]], engine,
                 ensemble, instances, base: dict, audit: list[dict],
                 workers: int) -> dict:
    best_settings: dict | None = None
    best_score: tuple[float, float] | None = None
    for name, settings in candidates:
        metrics = evaluate(
            engine, ensemble, instances, base, settings, workers)
        audit.append({"stage": stage, "name": name,
                      "settings": settings, "metrics": metrics})
        score = (metrics["mean_expansions"], metrics["mean_runtime_ms"])
        print(f"[{stage:10s}] {name:24s} exp={score[0]:8.2f} "
              f"runtime={score[1]:9.2f}ms max-ratio={metrics['max_ratio']:.4f}")
        print(f"{'':13s} wall={metrics['wall_seconds']:.1f}s "
              f"workers={workers}", flush=True)
        if best_score is None or score < best_score:
            best_score = score
            best_settings = copy.deepcopy(settings)
    assert best_settings is not None
    return best_settings


def joint_candidates(legacy: dict) -> list[tuple[str, dict]]:
    experts = {
        "geometry": [1.0, 0.0, 0.0],
        "geometry_topology": [0.5, 0.5, 0.0],
        "geometry_goal": [0.5, 0.0, 0.5],
        "uniform": [1 / 3, 1 / 3, 1 / 3],
    }
    confidences = [
        (intra, inter, temperature)
        for intra, inter in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
        for temperature in (0.002, 0.01, 0.05)
    ]
    controllers = [
        ("multiplicative", {"type": "multiplicative"}),
        ("linear", {"type": "linear", "lin_a": 1 / 3,
                    "lin_b": 1 / 3, "lin_c": 1 / 3}),
        ("threshold-low", {
            "type": "threshold",
            "controller_confidence_threshold": 0.2,
            "controller_disagreement_threshold": 0.1,
            "controller_conservative_width": 1.0,
        }),
        ("threshold-mid", {
            "type": "threshold",
            "controller_confidence_threshold": 0.35,
            "controller_disagreement_threshold": 0.25,
            "controller_conservative_width": 1.25,
        }),
        ("fixed-wstar", {
            "type": "fixed", "tuned_fixed_w": legacy["w_star"],
        }),
    ]
    candidates = []
    for ((expert_name, weights),
         (intra, inter, temperature),
         (controller_name, controller)) in product(
             experts.items(), confidences, controllers):
        settings = copy.deepcopy(DEFAULT_NEXT)
        settings["expert"] = {
            "name": expert_name, "weights": weights,
        }
        settings["confidence"].update({
            "intra_weight": intra,
            "inter_weight": inter,
            "temperature": temperature,
        })
        settings["controller"] = controller
        name = (
            f"{expert_name}|i={intra:g},e={inter:g},t={temperature:g}|"
            f"{controller_name}")
        candidates.append((name, settings))
    return candidates


def score_trial(metrics: dict, reference: dict, width_bound: float,
                profile: str) -> tuple[float, bool, float]:
    width_collapse = max(
        0.0, (1.02 - metrics["mean_focal_width"]) / 0.02)
    fallback_collapse = max(
        0.0, (metrics["mean_fallback_rate"] - 0.90) / 0.10)
    collapse = width_collapse + fallback_collapse
    rejected = (
        metrics["mean_focal_width"] <= 1.02 or
        metrics["mean_fallback_rate"] >= 0.90)
    expansion = (
        metrics["mean_expansions"] /
        max(1e-12, reference["mean_expansions"]))
    runtime = (
        metrics["mean_runtime_ms"] /
        max(1e-12, reference["mean_runtime_ms"]))
    suboptimality = max(
        0.0, (metrics["mean_ratio"] - 1.0) /
        max(1e-12, width_bound - 1.0))
    if profile == "conservative":
        score = (
            0.30 * expansion + 0.15 * runtime +
            0.15 * metrics["mean_fallback_rate"] +
            0.35 * suboptimality + 0.05 * collapse)
    else:
        score = (
            0.45 * expansion + 0.25 * runtime +
            0.15 * metrics["mean_fallback_rate"] +
            0.10 * suboptimality + 0.05 * collapse)
    return score, rejected, collapse


def main() -> None:
    args = parse_args()
    forbidden = {"test", "shift_density", "shift_size", "shift_family"}
    if (forbidden.intersection(args.splits) or
            any(split.startswith("final_") for split in args.splits)):
        raise ValueError("test/OOD-test splits cannot be used for tuning")
    if args.per_split < 1:
        raise ValueError("--per-split must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.guidance_region_radius < 0:
        raise ValueError("--guidance-region-radius must be non-negative")
    if not 1.0 <= args.conservative_max_ratio <= 2.0:
        raise ValueError("--conservative-max-ratio must be in [1, 2]")

    engine = load_engine(required=("run_cadfs_next",))
    print(f"engine: {getattr(engine, '__file__', 'unknown')}", flush=True)
    legacy = load_json(ROOT / "results/models/tuned.json")
    base = dict(legacy["base"])
    base["theta_c"] = legacy["theta_c"]
    standard_fast = ROOT / "results/models/fast_ensemble.txt"
    guidance_name = args.guidance
    if guidance_name == "auto":
        guidance_name = (
            "fast" if args.guidance_model or standard_fast.exists()
            else "cnn")
    default_model = (
        "results/models/fast_ensemble.txt"
        if guidance_name == "fast"
        else "results/models/ensemble.txt")
    model_path = Path(args.guidance_model or default_model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    fast_path = model_path
    if guidance_name == "fast":
        if not fast_path.exists():
            raise FileNotFoundError(
                f"{fast_path} is missing; run python/ml/train_student.py")
        ensemble = engine.FastEnsembleGuidance(str(model_path))
    elif guidance_name == "cnn-adaptive":
        ensemble = engine.EnsembleGuidance(
            str(model_path),
            args.early_exit_members, args.early_exit_variance)
    else:
        ensemble = engine.EnsembleGuidance(str(model_path))
    instances = load_validation(
        engine, args.splits, args.per_split, args.seed)
    base["guidance_region_radius"] = args.guidance_region_radius
    print(f"validation instances: {len(instances)}; workers: {args.workers}",
          flush=True)
    candidates = joint_candidates(legacy)
    if args.max_candidates:
        random.Random(args.seed).shuffle(candidates)
        candidates = candidates[:args.max_candidates]
    print(
        f"joint candidates: {len(candidates)}; guidance: {guidance_name}",
        flush=True)

    audit: list[dict] = []
    for index, (name, settings) in enumerate(candidates, start=1):
        metrics = evaluate(
            engine, ensemble, instances, base, settings, args.workers)
        audit.append({
            "stage": "joint",
            "name": name,
            "settings": settings,
            "metrics": metrics,
        })
        print(
            f"[{index:03d}/{len(candidates):03d}] {name} "
            f"exp={metrics['mean_expansions']:.1f} "
            f"runtime={metrics['mean_runtime_ms']:.2f}ms "
            f"width={metrics['mean_focal_width']:.3f} "
            f"fallback={metrics['mean_fallback_rate']:.3f} "
            f"ratio={metrics['mean_ratio']:.4f}",
            flush=True)

    reference = audit[0]["metrics"]
    width_bound = float(base["W"])
    selected: dict[str, dict] = {}
    quality_gate_satisfied: dict[str, bool] = {}
    for profile in ("balanced", "conservative"):
        eligible = []
        noncollapsed = []
        for trial in audit:
            score, rejected, collapse = score_trial(
                trial["metrics"], reference, width_bound, profile)
            trial[f"{profile}_score"] = score
            trial["collapse_penalty"] = collapse
            trial["rejected_collapse"] = rejected
            quality_rejected = (
                profile == "conservative" and
                trial["metrics"]["max_ratio"] >
                args.conservative_max_ratio)
            trial[f"{profile}_quality_rejected"] = quality_rejected
            if not rejected:
                noncollapsed.append((score, trial))
            if not rejected and not quality_rejected:
                eligible.append((score, trial))
        if not eligible:
            if not noncollapsed:
                raise RuntimeError(
                    "all candidates collapsed; expand the joint grid or inspect "
                    "confidence calibration")
            if profile == "conservative":
                print(
                    "warning: no non-collapsed candidate satisfies "
                    f"max_ratio<={args.conservative_max_ratio:g}; "
                    "saving the best non-collapsed candidate with "
                    "quality_gate_satisfied=false",
                    flush=True)
            eligible = noncollapsed
            quality_gate_satisfied[profile] = False
        else:
            quality_gate_satisfied[profile] = True
        selected[profile] = min(eligible, key=lambda item: item[0])[1]

    tail_reference = None
    if args.out_tail:
        print("evaluating split-wise Weighted A* tail reference", flush=True)
        tail_reference = evaluate_wastar_tail(
            engine, instances, base, args.workers)
        eligible_tail = []
        noncollapsed_tail = []
        for trial in audit:
            split_metrics = trial["metrics"]["split_metrics"]
            max_margins = [
                split_metrics[split]["max_ratio"] - values["max_ratio"]
                for split, values in tail_reference.items()
            ]
            cvar_margins = [
                split_metrics[split]["cvar95_ratio"] - values["cvar95_ratio"]
                for split, values in tail_reference.items()
            ]
            p95_margins = [
                split_metrics[split]["p95_ratio"] - values["p95_ratio"]
                for split, values in tail_reference.items()
            ]
            score = (
                max(max_margins), max(cvar_margins), max(p95_margins),
                trial["metrics"]["mean_expansions"] /
                max(1e-12, reference["mean_expansions"]),
                trial["metrics"]["mean_runtime_ms"] /
                max(1e-12, reference["mean_runtime_ms"]),
            )
            gate_pass = all(margin < 0.0 for margin in max_margins)
            trial["tail_score"] = score
            trial["tail_max_margins_vs_wastar"] = {
                split: split_metrics[split]["max_ratio"] - values["max_ratio"]
                for split, values in tail_reference.items()
            }
            trial["tail_gate_pass"] = gate_pass
            if not trial["rejected_collapse"]:
                noncollapsed_tail.append((score, trial))
                if gate_pass:
                    eligible_tail.append((score, trial))
        if not noncollapsed_tail:
            raise RuntimeError("all tail candidates collapsed")
        quality_gate_satisfied["tail"] = bool(eligible_tail)
        if not eligible_tail:
            print(
                "warning: no non-collapsed candidate beats the validation "
                "Weighted A* maximum on every split; saving the closest "
                "candidate with quality_gate_satisfied=false",
                flush=True)
            eligible_tail = noncollapsed_tail
        selected["tail"] = min(eligible_tail, key=lambda item: item[0])[1]

    created_at = datetime.now(timezone.utc).isoformat()

    def write_profile(profile: str, value: str) -> Path:
        trial = selected[profile]
        output = {
            "schema_version": 2,
            "algorithm_version": "cadfs-next-v2",
            "profile": profile,
            "created_at": created_at,
            "selection_splits": args.splits,
            "per_split": args.per_split,
            "seed": args.seed,
            "workers": args.workers,
            "guidance_backend": guidance_name,
            "guidance_model": str(model_path),
            "guidance_region_radius": args.guidance_region_radius,
            "conservative_max_ratio": args.conservative_max_ratio,
            "quality_gate_satisfied": quality_gate_satisfied[profile],
            "wastar_split_tail_reference": (
                tail_reference if profile == "tail" else None),
            "selected_name": trial["name"],
            "selected_metrics": trial["metrics"],
            **trial["settings"],
            "trials": audit,
        }
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            json.dump(output, stream, indent=2)
        return path

    balanced_path = write_profile("balanced", args.out)
    conservative_path = write_profile(
        "conservative", args.out_conservative)
    print(f"balanced     -> {balanced_path}")
    print(f"conservative -> {conservative_path}")
    if args.out_tail:
        tail_path = write_profile("tail", args.out_tail)
        print(f"tail-robust  -> {tail_path}")


if __name__ == "__main__":
    main()
