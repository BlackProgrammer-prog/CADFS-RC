"""Generate predeclared, leakage-free final/OOD/scaling benchmark splits.

This generator is intentionally separate from the development dataset.  It
refuses to overwrite any target CSV or map and emits SHA-256 manifests.  Run it
only after creating FROZEN_PROTOCOL.json; do not tune after observing results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "scripts"))
from gen_synthetic import generate_map_task, save_map  # noqa: E402
from freeze_final_protocol import verify as verify_frozen_protocol  # noqa: E402

VERSION = "v1"
MASTER_SEED = 84673129
FIELDS = [
    "map_id", "map_path", "start_x", "start_y", "goal_x", "goal_y",
    "family", "density", "width", "height", "split", "optimal_cost",
]

# Core: 1,100 queries.  Scaling: 96 paired queries at each size.  The 2048
# tier is isolated because it is a stress test rather than a powered estimate.
GROUP_SPECS = {
    "core": [
        dict(split="final_test_v1", family="random", size=64, maps=30,
             queries=10, levels=[dict(density=d) for d in (.10, .20, .30)]),
        dict(split="final_ood_density_v1", family="random", size=64, maps=30,
             queries=10, levels=[dict(density=d) for d in (.35, .40, .45)]),
        dict(split="final_ood_structure_v1", family="maze", size=64, maps=15,
             queries=10, levels=[dict(corridor=1)]),
        dict(split="final_ood_structure_v1", family="narrow", size=64, maps=15,
             queries=10,
             levels=[dict(passage=1, clutter=c) for c in (.18, .22, .26)]),
        dict(split="final_ood_size_v1", family="random", size=128, maps=20,
             queries=10, levels=[dict(density=d) for d in (.20, .30, .40)]),
    ],
    "scaling": [
        dict(split=f"final_scale_{size}_v1", family="random", size=size,
             maps=12, queries=8,
             levels=[dict(density=d) for d in (.10, .20, .30)])
        for size in (256, 512, 1024)
    ],
    "stress": [
        dict(split="final_scale_2048_v1", family="random", size=2048,
             maps=4, queries=4,
             levels=[dict(density=d) for d in (.10, .20)]),
    ],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--groups", nargs="+", choices=sorted(GROUP_SPECS),
        default=["core", "scaling"],
        help="stress (2048) is opt-in and reported separately")
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument(
        "--workers", type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--starts-per-goal", type=int, default=2)
    parser.add_argument("--max-map-attempts", type=int, default=200)
    parser.add_argument(
        "--lock", default="results/final_v1/FROZEN_PROTOCOL.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def derive_tasks(groups: list[str], seed: int, starts_per_goal: int,
                 max_map_attempts: int) -> dict[str, list[dict]]:
    # Generate seeds for the full canonical plan first.  Selecting a group can
    # therefore never change any map in another group.
    master = random.Random(seed)
    selected = set(groups)
    tasks: dict[str, list[dict]] = {}
    counter = 0
    for group in ("core", "scaling", "stress"):
        for spec in GROUP_SPECS[group]:
            for index in range(spec["maps"]):
                map_seed = master.randrange(2**31)
                map_id = (
                    f"final_{spec['family']}_{spec['size']}_{counter:05d}_{VERSION}")
                counter += 1
                if group not in selected:
                    continue
                tasks.setdefault(spec["split"], []).append({
                    "group": group,
                    "split": spec["split"],
                    "family": spec["family"],
                    "level": spec["levels"][index % len(spec["levels"])],
                    "size": spec["size"],
                    "map_id": map_id,
                    "seed": map_seed,
                    "queries_per_map": spec["queries"],
                    "starts_per_goal": starts_per_goal,
                    "max_map_attempts": max_map_attempts,
                })
    return tasks


def target_map(task: dict) -> Path:
    return (ROOT / "data" / "synthetic" / task["family"] /
            task["split"] / f"{task['map_id']}.map")


def preflight(tasks: dict[str, list[dict]], groups: list[str], lock: Path) -> None:
    if not lock.is_file():
        raise FileNotFoundError(
            f"missing protocol lock: {lock}; freeze code/model/tuning first")
    verify_frozen_protocol(lock, require_paper_final=True)
    collisions = []
    for split, split_tasks in tasks.items():
        csv_path = ROOT / "data" / "instances" / f"{split}.csv"
        if csv_path.exists():
            collisions.append(csv_path)
        collisions.extend(path for path in map(target_map, split_tasks) if path.exists())
    for group in groups:
        manifest = ROOT / "data" / "instances" / f"final_{group}_{VERSION}.manifest.json"
        if manifest.exists():
            collisions.append(manifest)
    if collisions:
        shown = "\n  ".join(str(path) for path in collisions[:12])
        raise FileExistsError(
            "final generator never overwrites existing outputs:\n  " + shown)


def write_csv(split: str, records: list[dict]) -> Path:
    path = ROOT / "data" / "instances" / f"{split}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    records.sort(key=lambda row: (
        row["map_id"], row["start_x"], row["start_y"],
        row["goal_x"], row["goal_y"]))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return path


def group_manifest(group: str, split_outputs: dict[str, dict], lock: Path,
                   args: argparse.Namespace) -> Path:
    selected = {
        split: value for split, value in split_outputs.items()
        if value["group"] == group
    }
    payload = {
        "schema_version": 1,
        "dataset_version": VERSION,
        "group": group,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "master_seed": args.seed,
        "generator_sha256": sha256_file(Path(__file__)),
        "protocol_lock": lock.relative_to(ROOT).as_posix(),
        "protocol_lock_sha256": sha256_file(lock),
        "git_commit": git_commit(),
        "starts_per_goal": args.starts_per_goal,
        "splits": selected,
    }
    path = ROOT / "data" / "instances" / f"final_{group}_{VERSION}.manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.starts_per_goal < 1 or args.max_map_attempts < 1:
        raise ValueError("workers, starts-per-goal and max-map-attempts must be >= 1")
    groups = list(dict.fromkeys(args.groups))
    lock = Path(args.lock)
    if not lock.is_absolute():
        lock = ROOT / lock
    tasks = derive_tasks(
        groups, args.seed, args.starts_per_goal, args.max_map_attempts)

    plan = {
        split: {
            "maps": len(items),
            "queries": sum(item["queries_per_map"] for item in items),
            "size": sorted({item["size"] for item in items}),
            "families": sorted({item["family"] for item in items}),
        }
        for split, items in tasks.items()
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return
    preflight(tasks, groups, lock)

    # No final file is written until every map in all requested groups has
    # generated successfully.  This avoids a half-created benchmark on failure.
    generated: dict[str, list[dict]] = {split: [] for split in tasks}
    all_tasks = [(split, task) for split, items in tasks.items() for task in items]
    print(f"generating {len(all_tasks)} maps with {args.workers} workers")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(generate_map_task, task): split
            for split, task in all_tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            split = futures[future]
            generated[split].append(future.result())
            if completed == 1 or completed % 10 == 0 or completed == len(futures):
                print(f"[{completed:4d}/{len(futures):4d}] maps ready", flush=True)

    split_outputs: dict[str, dict] = {}
    for split, results in generated.items():
        records: list[dict] = []
        map_entries: dict[str, str] = {}
        for result in sorted(results, key=lambda item: item["map_id"]):
            path = target_map(result)
            if path.exists():
                raise FileExistsError(path)
            save_map(path, result["rows"])
            rel = path.relative_to(ROOT).as_posix()
            map_entries[rel] = sha256_file(path)
            for (sx, sy), (gx, gy), cstar in result["queries"]:
                records.append({
                    "map_id": result["map_id"], "map_path": rel,
                    "start_x": sx, "start_y": sy,
                    "goal_x": gx, "goal_y": gy,
                    "family": result["family"],
                    "density": result["density"],
                    "width": result["size"], "height": result["size"],
                    "split": split, "optimal_cost": round(cstar, 6),
                })
        csv_path = write_csv(split, records)
        group = results[0]["group"]
        split_outputs[split] = {
            "group": group,
            "instances": len(records),
            "maps": len(results),
            "csv": csv_path.relative_to(ROOT).as_posix(),
            "csv_sha256": sha256_file(csv_path),
            "map_sha256": map_entries,
        }
        print(f"wrote {csv_path} ({len(records)} instances)")

    # Map IDs are globally unique across the canonical plan by construction.
    ids = [result["map_id"] for values in generated.values() for result in values]
    if len(ids) != len(set(ids)):
        raise AssertionError("map-level leakage inside final benchmark")
    for group in groups:
        print(f"manifest -> {group_manifest(group, split_outputs, lock, args)}")


if __name__ == "__main__":
    main()
