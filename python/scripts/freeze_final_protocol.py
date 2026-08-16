"""Freeze or verify the exact implementation used for a final benchmark.

The lock deliberately hashes source files instead of the whole worktree so
that subsequently generated data/results do not invalidate it.  Creation is
refused when benchmark-relevant paths are dirty by default; this makes the
recorded commit a real, recoverable snapshot rather than a decorative ID.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = [
    "results/models/fast_ensemble.txt",
    "results/models/tuned.json",
    "results/models/tuned_next.json",
    "results/models/metric_r1/tuned_next_conservative.json",
]
SOURCE_ROOTS = ["cpp", "python", "configs"]
SOURCE_FILES = [
    "CMakeLists.txt", "requirements.txt", "requirements-cpu.txt",
    "requirements-gpu.txt",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True).strip()


def source_paths() -> list[Path]:
    paths: list[Path] = []
    for name in SOURCE_ROOTS:
        base = ROOT / name
        paths.extend(
            path for path in base.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    paths.extend(ROOT / name for name in SOURCE_FILES if (ROOT / name).is_file())
    return sorted(set(paths), key=lambda path: path.relative_to(ROOT).as_posix())


def inventory(paths: list[Path]) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen input(s): " + ", ".join(missing))
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in paths
    }


def combined_sha256(entries: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(entries.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["create", "verify"])
    parser.add_argument(
        "--lock", default="results/final_v1/FROZEN_PROTOCOL.json")
    parser.add_argument("--artifacts", nargs="+", default=DEFAULT_ARTIFACTS)
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="record a dirty tree (diagnostic only; not paper-final)")
    return parser.parse_args()


def create(lock_path: Path, artifact_names: list[str], allow_dirty: bool) -> None:
    if lock_path.exists():
        raise FileExistsError(f"lock already exists: {lock_path}")
    relevant_scope = [*SOURCE_ROOTS, *SOURCE_FILES, *artifact_names]
    status = git("status", "--porcelain", "--", *relevant_scope)
    overall_status = git("status", "--porcelain")
    if status and not allow_dirty:
        raise RuntimeError(
            "Benchmark-relevant source/artifacts are dirty. Commit the "
            "intended implementation before creating a paper-final lock. "
            "Unrelated files outside cpp/python/configs do not block freeze. "
            "Use --allow-dirty only for a diagnostic rehearsal.\n" + status)

    sources = inventory(source_paths())
    artifacts = inventory([ROOT / name for name in artifact_names])
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_final": not bool(status),
        "git_commit": git("rev-parse", "HEAD"),
        "relevant_git_dirty_at_freeze": bool(status),
        "overall_git_dirty_at_freeze": bool(overall_status),
        "source_tree_sha256": combined_sha256(sources),
        "artifact_set_sha256": combined_sha256(artifacts),
        "sources": sources,
        "artifacts": artifacts,
        "predeclared_primary_claim": {
            "name": "empirical tail-quality replication",
            "candidate": "cadfs_next_metric_tuned",
            "baseline": "wastar",
            "required_on_every_predeclared_split": [
                "success_rate == 1",
                "bound_violations == 0",
                "observed_max_ratio(candidate) < observed_max_ratio(baseline)",
            ],
            "note": (
                "This is an empirical observed-maximum claim, not a tighter "
                "theoretical worst-case bound; both methods retain W=2."
            ),
        },
        "predeclared_secondary_tail_metrics": [
            "ratio_p95", "ratio_p99", "ratio_cvar95",
        ],
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"created immutable protocol lock -> {lock_path}")
    print(f"source tree sha256  -> {payload['source_tree_sha256']}")
    print(f"artifact set sha256 -> {payload['artifact_set_sha256']}")


def verify(lock_path: Path, require_paper_final: bool = False) -> None:
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if require_paper_final and not payload.get("paper_final", False):
        raise RuntimeError(
            "runner refuses a rehearsal lock (paper_final=false); create the "
            "lock from committed benchmark-relevant sources")
    expected_sources = payload["sources"]
    expected_artifacts = payload["artifacts"]
    actual_sources = inventory([ROOT / name for name in expected_sources])
    actual_artifacts = inventory([ROOT / name for name in expected_artifacts])

    changed = []
    for kind, expected, actual in (
            ("source", expected_sources, actual_sources),
            ("artifact", expected_artifacts, actual_artifacts)):
        for name in sorted(set(expected) | set(actual)):
            if expected.get(name) != actual.get(name):
                changed.append(f"{kind}: {name}")
    if changed:
        raise RuntimeError("frozen protocol mismatch:\n  " + "\n  ".join(changed))
    if combined_sha256(actual_sources) != payload["source_tree_sha256"]:
        raise RuntimeError("combined source hash mismatch")
    if combined_sha256(actual_artifacts) != payload["artifact_set_sha256"]:
        raise RuntimeError("combined artifact hash mismatch")
    print(f"FROZEN PROTOCOL VERIFIED: {lock_path}")
    print(f"git snapshot: {payload['git_commit']}")
    print(f"paper_final: {payload['paper_final']}")


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    if not lock_path.is_absolute():
        lock_path = ROOT / lock_path
    try:
        if args.action == "create":
            create(lock_path, args.artifacts, args.allow_dirty)
        else:
            verify(lock_path)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from None


if __name__ == "__main__":
    main()
