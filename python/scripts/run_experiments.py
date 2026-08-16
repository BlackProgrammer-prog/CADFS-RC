"""Run paired Legacy/CADFS-Next benchmarks without overwriting old results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from cadfs_py import load_engine  # noqa: E402
from cadfs_py.experiments import (  # noqa: E402
    METHOD_SUITES,
    build_methods,
    load_settings,
    select_methods,
)

FIELDS = [
    "problem_id", "split", "family", "density", "map_id", "instance",
    "seed", "algorithm_version", "method", "found", "cost", "cstar",
    "ratio", "expansions", "generated", "runtime_ms", "fallback_rate",
    "mean_w", "min_w", "max_w", "mean_abs_dw", "mean_C", "mean_R",
    "model_eval_count", "model_member_evals", "model_cache_hits",
    "model_cache_hit_rate", "model_eval_time_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits", nargs="+",
        default=["test", "shift_density", "shift_size", "shift_family"])
    parser.add_argument("--per-split", type=int, default=40)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument(
        "--suite", choices=["main", "next", "legacy", "full"], default="main")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--list-methods", action="store_true")
    parser.add_argument("--tuned-next", default="results/models/tuned_next.json")
    parser.add_argument(
        "--tuned-next-metric",
        help="optional separately tuned metric-region JSON")
    parser.add_argument("--mlp-model", default="results/models/controller_mlp.json")
    parser.add_argument(
        "--guidance", choices=["auto", "fast", "cnn", "cnn-adaptive"],
        default="auto")
    parser.add_argument(
        "--guidance-model",
        help="versioned guidance export; defaults to the standard model path")
    parser.add_argument("--early-exit-members", type=int, default=2)
    parser.add_argument("--early-exit-variance", type=float, default=0.01)
    parser.add_argument(
        "--warmup-runs", type=int, default=0,
        help="discarded warm-up queries per method and split")
    parser.add_argument(
        "--repetitions", type=int, default=1,
        help="timed repetitions per method/query; median timing is recorded")
    parser.add_argument(
        "--method-order", choices=["fixed", "rotate", "shuffle"],
        default="fixed", help="counterbalance timing order across queries")
    parser.add_argument("--out", default="results/logs/bench_next.csv")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--append", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_instances(split: str, limit: int, seed: int) -> list[dict[str, str]]:
    path = ROOT / "data/instances" / f"{split}.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    random.Random(seed).shuffle(rows)
    return rows[:limit]


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(
                encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def package_versions() -> dict[str, str]:
    versions = {}
    for name in ("numpy", "pandas", "scipy", "torch", "pybind11"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def median_timed_result(runs: list[dict]) -> dict:
    reference = runs[0]
    for result in runs[1:]:
        if (result["found"] != reference["found"] or
                result["expansions"] != reference["expansions"] or
                not math.isclose(result["cost"], reference["cost"],
                                 rel_tol=1e-12, abs_tol=1e-12)):
            raise AssertionError("non-deterministic search result across repetitions")
    result = dict(reference)
    result["runtime_ms"] = statistics.median(
        item["runtime_ms"] for item in runs)
    result["model_eval_time_ms"] = statistics.median(
        item["model_eval_time_ms"] for item in runs)
    return result


def main() -> None:
    args = parse_args()
    if args.warmup_runs < 0 or args.repetitions < 1:
        raise ValueError("warmup-runs must be >= 0 and repetitions must be >= 1")
    engine = load_engine(required=("run_cadfs_next",))
    legacy, next_settings = load_settings(
        ROOT, resolve(ROOT, args.tuned_next))
    metric_tuned_path = (
        resolve(ROOT, args.tuned_next_metric)
        if args.tuned_next_metric else None)
    metric_next_settings = (
        json.loads(metric_tuned_path.read_text(encoding="utf-8"))
        if metric_tuned_path and metric_tuned_path.exists() else None)
    if metric_tuned_path and metric_next_settings is None:
        raise FileNotFoundError(metric_tuned_path)
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
    guidance_path = resolve(ROOT, args.guidance_model or default_model)
    fast_path = guidance_path
    if guidance_name == "fast":
        if not fast_path.exists():
            raise FileNotFoundError(
                f"{fast_path} is missing; run python/ml/train_student.py")
        ensemble = engine.FastEnsembleGuidance(str(fast_path))
        guidance_path = fast_path
    elif guidance_name == "cnn-adaptive":
        ensemble = engine.EnsembleGuidance(
            str(guidance_path),
            args.early_exit_members, args.early_exit_variance)
    else:
        ensemble = engine.EnsembleGuidance(str(guidance_path))

    available = build_methods(
        engine, ensemble, legacy, next_settings,
        resolve(ROOT, args.mlp_model),
        metric_next_settings=metric_next_settings)
    if args.list_methods:
        print(*available, sep="\n")
        return
    methods = select_methods(available, args.suite, args.methods)

    out_path = resolve(ROOT, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not (args.append or args.overwrite):
        raise FileExistsError(
            f"{out_path} exists; use --append, --overwrite, or another --out")
    mode = "a" if args.append and out_path.exists() else "w"

    if mode == "a":
        with out_path.open(newline="", encoding="utf-8") as existing:
            existing_fields = next(csv.reader(existing), [])
        if existing_fields != FIELDS:
            raise ValueError(
                "cannot append: existing CSV schema differs from CADFS Next schema")

    created_at = datetime.now(timezone.utc)
    tuned_path = resolve(ROOT, args.tuned_next)
    split_hashes = {
        split: sha256_file(ROOT / "data/instances" / f"{split}.csv")
        for split in args.splits
    }
    git_status = git_value("status", "--porcelain")
    manifest = {
        "created_at": created_at.isoformat(),
        "engine": str(getattr(engine, "__file__", "unknown")),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_model": cpu_model(),
            "logical_cpus": os.cpu_count(),
            "python": sys.version,
            "packages": package_versions(),
            "engine_build": (
                engine.build_info() if hasattr(engine, "build_info") else {}),
        },
        "algorithm_version": "cadfs-next-v2",
        "command": [sys.executable, *sys.argv],
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "dirty": bool(git_status),
        },
        "artifacts": {
            "guidance_model": {
                "path": str(guidance_path),
                "sha256": sha256_file(guidance_path),
            },
            "legacy_tuning_sha256": sha256_file(
                ROOT / "results/models/tuned.json"),
            "next_tuning_sha256": sha256_file(tuned_path),
            "metric_next_tuning_sha256": (
                sha256_file(metric_tuned_path)
                if metric_tuned_path else None),
            "split_csv_sha256": split_hashes,
        },
        "timing_protocol": {
            "warmup_runs_per_method_split": args.warmup_runs,
            "repetitions_per_query": args.repetitions,
            "recorded_statistic": "median",
            "method_order": args.method_order,
        },
        "guidance_model": {
            "backend": guidance_name,
            "members": getattr(ensemble, "members",
                               getattr(ensemble, "heads", 0)),
            "format_version": getattr(ensemble, "format_version", 1),
            "patch_size": getattr(ensemble, "patch_size", 15),
            "variance_scale": getattr(ensemble, "variance_scale", 1.0),
            "variance_floor": getattr(ensemble, "variance_floor", 0.0),
            "early_exit_members": getattr(
                ensemble, "early_exit_members", 0),
            "early_exit_variance": getattr(
                ensemble, "early_exit_variance", 0.0),
        },
        "seed": args.seed,
        "splits": args.splits,
        "per_split": args.per_split,
        "methods": list(methods),
        "legacy_tuning": legacy,
        "next_tuning": next_settings,
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    if mode == "a" and manifest_path.exists():
        stamp = created_at.strftime("%Y%m%dT%H%M%SZ")
        manifest_path = out_path.with_name(
            f"{out_path.stem}.append-{stamp}.manifest.json")
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)

    base = dict(legacy["base"])
    base["theta_c"] = legacy["theta_c"]
    maximum_width = float(base["W"])

    with out_path.open(mode, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if mode == "w":
            writer.writeheader()

        for split in args.splits:
            rows = load_instances(split, args.per_split, args.seed)
            started = time.perf_counter()
            if rows and args.warmup_runs:
                warm = rows[0]
                warm_map = engine.GridMap.load_movingai(
                    str(ROOT / warm["map_path"]))
                warm_start = (int(warm["start_x"]), int(warm["start_y"]))
                warm_goal = (int(warm["goal_x"]), int(warm["goal_y"]))
                warm_config = dict(
                    base, h_max=math.hypot(
                        warm_map.width, warm_map.height))
                for _, method in methods.items():
                    for _ in range(args.warmup_runs):
                        method(warm_map, warm_start, warm_goal, warm_config)
            for index, row in enumerate(rows):
                map_ = engine.GridMap.load_movingai(str(ROOT / row["map_path"]))
                start = (int(row["start_x"]), int(row["start_y"]))
                goal = (int(row["goal_x"]), int(row["goal_y"]))
                optimal = float(row["optimal_cost"])
                if optimal <= 0.0:
                    raise ValueError(f"non-positive C* for {row['map_id']}")
                config = dict(base, h_max=math.hypot(map_.width, map_.height))
                problem_id = (
                    f"{row['map_id']}:{start[0]},{start[1]}:"
                    f"{goal[0]},{goal[1]}")

                method_items = list(methods.items())
                if args.method_order == "rotate" and method_items:
                    offset = index % len(method_items)
                    method_items = method_items[offset:] + method_items[:offset]
                elif args.method_order == "shuffle":
                    random.Random(
                        f"{args.seed}:{split}:{problem_id}").shuffle(method_items)

                for name, method in method_items:
                    result = median_timed_result([
                        method(map_, start, goal, config)
                        for _ in range(args.repetitions)
                    ])
                    if result["found"]:
                        tolerance = max(1e-9, 1e-9 * maximum_width * optimal)
                        if result["cost"] > maximum_width * optimal + tolerance:
                            raise AssertionError(
                                f"BOUND VIOLATION {name} {problem_id}: "
                                f"{result['cost']} > {maximum_width * optimal}")

                    ratio = result["cost"] / optimal if result["found"] else math.inf
                    writer.writerow({
                        "problem_id": problem_id,
                        "split": split, "family": row["family"],
                        "density": row["density"], "map_id": row["map_id"],
                        "instance": index, "seed": args.seed,
                        "algorithm_version": (
                            ("cadfs-next-metric-v1"
                             if name.startswith("cadfs_next_metric")
                             else "cadfs-next-v2") if name.startswith("cadfs_next")
                            else "legacy-v1"),
                        "method": name, "found": int(result["found"]),
                        "cost": round(result["cost"], 6), "cstar": optimal,
                        "ratio": round(ratio, 6),
                        "expansions": result["expansions"],
                        "generated": result["generated"],
                        "runtime_ms": round(result["runtime_ms"], 3),
                        "fallback_rate": round(result["fallback_rate"], 6),
                        "mean_w": round(result["mean_w"], 6),
                        "min_w": round(result["min_w"], 6),
                        "max_w": round(result["max_w"], 6),
                        "mean_abs_dw": round(result["mean_abs_dw"], 6),
                        "mean_C": round(result["mean_C"], 6),
                        "mean_R": round(result["mean_R"], 6),
                        "model_eval_count": result["model_eval_count"],
                        "model_member_evals": result["model_member_evals"],
                        "model_cache_hits": result["model_cache_hits"],
                        "model_cache_hit_rate": round(
                            result["model_cache_hit_rate"], 6),
                        "model_eval_time_ms": round(
                            result["model_eval_time_ms"], 3),
                    })
                stream.flush()
            elapsed = time.perf_counter() - started
            print(f"[{split}] {len(rows)} instances x {len(methods)} methods "
                  f"in {elapsed:.1f}s")

    print(f"log -> {out_path}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
