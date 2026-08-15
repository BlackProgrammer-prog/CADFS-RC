"""Runtime helpers for loading the CADFS C++ extension.

The project is commonly built either in ``build`` or a CLion-generated
``cmake-build-*`` directory.  Keeping discovery here prevents experiment
scripts from silently importing an older extension copied to the repo root.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]


def _engine_directories() -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("CADFS_ENGINE_DIR")
    if configured:
        candidates.append(Path(configured).expanduser())
    # Experiment and benchmark scripts should prefer optimized inference when
    # no explicit directory is configured.  CADFS_ENGINE_DIR still has the
    # highest priority and can intentionally select a debug build.
    candidates.extend((
        ROOT / "cmake-build-release",
        ROOT / "cmake-build-relwithdebinfo",
        ROOT / "cmake-build-debug",
        ROOT / "build",
    ))
    candidates.extend(sorted(ROOT.glob("cmake-build-*")))
    candidates.append(ROOT)

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def load_engine(required: Iterable[str] = ()) -> ModuleType:
    """Load ``cadfs_engine`` and require the requested exported symbols."""
    required = tuple(required)
    failures: list[str] = []

    for directory in _engine_directories():
        if not directory.exists() or not any(directory.glob("cadfs_engine*.so")):
            continue

        sys.modules.pop("cadfs_engine", None)
        sys.path.insert(0, str(directory))
        try:
            module = importlib.import_module("cadfs_engine")
            missing = [name for name in required if not hasattr(module, name)]
            if not missing:
                return module
            failures.append(f"{directory}: missing {missing}")
        except ImportError as exc:
            failures.append(f"{directory}: {exc}")
        finally:
            try:
                sys.path.remove(str(directory))
            except ValueError:
                pass

    detail = "\n  ".join(failures) if failures else "no extension was found"
    raise ImportError(
        "Could not load a compatible cadfs_engine. Build the C++ module or set "
        f"CADFS_ENGINE_DIR. Attempts:\n  {detail}"
    )


__all__ = ["load_engine"]
