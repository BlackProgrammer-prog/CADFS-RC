"""Export the trained PyTorch ensemble to the engine's plain-text weight format.

Format (dependency-free to parse in C++):
  CADFS_ENSEMBLE 1
  K <k> PATCH <p> EXTRA <e> HIDDEN <h>
  MEMBER <i>
  CONV <cin> <cout>   then cout*cin*3*3 floats (w) + cout floats (b)
  CONV <cin> <cout>   ...
  FC <in> <out>       then out*in floats (row-major) + out floats (b)
  FC <in> <out>       ...

Output: results/models/ensemble.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from ml.model import CostToGoNet  # noqa: E402

MOD = ROOT / "results/models"


def dump(f, t: torch.Tensor) -> None:
    f.write(" ".join(f"{v:.9g}" for v in t.flatten().tolist()) + "\n")


def main() -> None:
    members = sorted(MOD.glob("member_*.pt"))
    assert members, "no checkpoints found; run train_ensemble.py first"
    out = MOD / "ensemble.txt"
    with open(out, "w") as f:
        f.write("CADFS_ENSEMBLE 1\n")
        f.write(f"K {len(members)} PATCH 15 EXTRA 4 HIDDEN 64\n")
        for i, ckpt in enumerate(members):
            net = CostToGoNet()
            net.load_state_dict(torch.load(ckpt, map_location="cpu"))
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
