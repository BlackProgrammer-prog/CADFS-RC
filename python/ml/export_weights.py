"""Export the trained PyTorch ensemble to the engine's plain-text weight format.

Format (dependency-free to parse in C++):
  CADFS_ENSEMBLE 2
  K <k> PATCH <p> EXTRA <e> HIDDEN <h> TARGET LOG1P
    VARIANCE_SCALE <s> VARIANCE_FLOOR <b>
  MEMBER <i>
  CONV <cin> <cout>   then cout*cin*3*3 floats (w) + cout floats (b)
  CONV <cin> <cout>   ...
  FC <in> <out>       then out*in floats (row-major) + out floats (b)
  FC <in> <out>       ...

Output: results/models/ensemble.txt
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from ml.model import MODEL_SCHEMA_VERSION, TARGET_TRANSFORM, CostToGoNet  # noqa: E402

MOD = ROOT / "results/models"


def dump(f, t: torch.Tensor) -> None:
    f.write(" ".join(f"{v:.9g}" for v in t.flatten().tolist()) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="results/models")
    parser.add_argument("--out", default="results/models/ensemble.txt")
    args = parser.parse_args()
    models_dir = Path(args.models_dir)
    if not models_dir.is_absolute():
        models_dir = ROOT / models_dir
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out

    members = sorted(models_dir.glob("member_*.pt"))
    assert members, "no checkpoints found; run train_ensemble.py first"
    loaded = [torch.load(path, map_location="cpu", weights_only=False)
              for path in members]
    for path, checkpoint in zip(members, loaded):
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError(
                f"{path} is a legacy checkpoint; retrain with model schema "
                f"{MODEL_SCHEMA_VERSION}")
        if checkpoint.get("schema_version") != MODEL_SCHEMA_VERSION:
            raise ValueError(f"unsupported checkpoint schema in {path}")
        if checkpoint.get("target_transform") != TARGET_TRANSFORM:
            raise ValueError(f"target transform mismatch in {path}")

    model_config = loaded[0]["model_config"]
    for path, checkpoint in zip(members[1:], loaded[1:]):
        if checkpoint["model_config"] != model_config:
            raise ValueError(f"ensemble architecture mismatch in {path}")

    calibration_path = models_dir / "calibration.json"
    if not calibration_path.exists():
        raise FileNotFoundError(
            f"missing {calibration_path}; train_ensemble.py must calibrate first")
    import json
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if "variance_scale" not in calibration or "variance_floor" not in calibration:
        raise ValueError(
            f"{calibration_path} uses legacy scale-only calibration; retrain "
            "the ensemble to produce affine calibration")
    variance_scale = float(calibration["variance_scale"])
    variance_floor = float(calibration["variance_floor"])
    if (not math.isfinite(variance_scale) or variance_scale < 0 or
            not math.isfinite(variance_floor) or variance_floor < 0):
        raise ValueError(
            "calibrated variance coefficients must be finite and non-negative")

    patch = int(model_config["patch"])
    extra = int(model_config["extra"])
    hidden = int(model_config["hidden"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"CADFS_ENSEMBLE {MODEL_SCHEMA_VERSION}\n")
        f.write(
            f"K {len(members)} PATCH {patch} EXTRA {extra} HIDDEN {hidden} "
            f"TARGET {TARGET_TRANSFORM} VARIANCE_SCALE {variance_scale:.17g} "
            f"VARIANCE_FLOOR {variance_floor:.17g}\n")
        for i, checkpoint in enumerate(loaded):
            net = CostToGoNet(patch=patch, extra=extra, hidden=hidden)
            net.load_state_dict(checkpoint["state_dict"])
            net.eval()
            f.write(f"MEMBER {i}\n")
            for name in ("conv1", "conv2"):
                c = getattr(net, name)
                f.write(f"CONV {c.in_channels} {c.out_channels}\n")
                dump(f, c.weight.data)
                dump(f, c.bias.data)
            for name in ("fc1", "fc2"):
                fc = getattr(net, name)
                f.write(f"FC {fc.in_features} {fc.out_features}\n")
                dump(f, fc.weight.data)
                dump(f, fc.bias.data)
    print(f"exported {len(members)} members -> {out} "
          f"({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
