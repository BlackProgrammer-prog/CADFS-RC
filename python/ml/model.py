"""C++-portable multi-scale cost-to-go regressor.

Architecture (deliberately restricted to ops that are trivial to mirror in the
C++ inference engine — Conv3x3/pad1, ReLU, MaxPool2, Linear):

  patch (1,31,31) -> Conv(1->8,3x3,pad1) -> ReLU -> MaxPool2
                   -> Conv(8->16,3x3,pad1) -> ReLU -> MaxPool2
                   -> flatten
  concat extra(10) -> Linear(...->96) -> ReLU -> Linear(96->1) -> Softplus

The network predicts log1p(d*(n,goal)/diag).  This target is non-negative and
unbounded, so long maze detours are representable without destabilizing the
loss.  Search converts each prediction z to the bounded, monotone priority
1-exp(-z), which is exactly y/(1+y) for a perfect prediction.
"""
from __future__ import annotations

import torch
import torch.nn as nn

MODEL_SCHEMA_VERSION = 2
DEFAULT_PATCH = 31
DEFAULT_EXTRA = 10
DEFAULT_HIDDEN = 96
DEFAULT_STUDENT_HIDDEN = (64, 32)
DEFAULT_STUDENT_HEADS = 3
TARGET_TRANSFORM = "LOG1P"


def encode_target(y: torch.Tensor) -> torch.Tensor:
    """Raw normalized cost-to-go -> stable non-negative training target."""
    return torch.log1p(torch.clamp_min(y, 0.0))


def decode_target(z: torch.Tensor) -> torch.Tensor:
    """Network target -> raw normalized cost-to-go for reported metrics."""
    return torch.expm1(torch.clamp(z, min=0.0, max=20.0))


def target_to_priority(z: torch.Tensor) -> torch.Tensor:
    """Network target -> bounded monotone search priority in [0,1)."""
    return -torch.expm1(-torch.clamp_min(z, 0.0))


class CostToGoNet(nn.Module):
    def __init__(self, patch: int = DEFAULT_PATCH, extra: int = DEFAULT_EXTRA,
                 hidden: int = DEFAULT_HIDDEN):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        p2 = patch // 2 // 2                      # 31 -> 15 -> 7
        self.flat_dim = 16 * p2 * p2              # 784 for a 31x31 patch
        self.fc1 = nn.Linear(self.flat_dim + extra, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.act = nn.ReLU()

    def forward(self, patch: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        z = self.pool(self.act(self.conv1(patch)))
        z = self.pool(self.act(self.conv2(z)))
        z = torch.cat([z.flatten(1), extra], dim=1)
        z = self.act(self.fc1(z))
        return nn.functional.softplus(self.fc2(z).squeeze(-1))


class FastMultiHeadStudent(nn.Module):
    """C++-portable shared MLP with cheap epistemic heads.

    Flattening the complete local patch preserves spatial information while
    avoiding seven repeated convolutional backbones during search.
    """

    def __init__(
        self,
        patch: int = DEFAULT_PATCH,
        extra: int = DEFAULT_EXTRA,
        hidden: tuple[int, int] = DEFAULT_STUDENT_HIDDEN,
        heads: int = DEFAULT_STUDENT_HEADS,
    ):
        super().__init__()
        if heads < 2:
            raise ValueError("at least two heads are required for uncertainty")
        self.patch_size = patch
        self.extra_features = extra
        self.hidden = hidden
        self.head_count = heads
        self.fc1 = nn.Linear(patch * patch + extra, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.heads = nn.ModuleList(
            nn.Linear(hidden[1], 1) for _ in range(heads))

    def forward(self, patch: torch.Tensor,
                extra: torch.Tensor) -> torch.Tensor:
        features = torch.cat([patch.flatten(1), extra], dim=1)
        features = torch.relu(self.fc1(features))
        features = torch.relu(self.fc2(features))
        return torch.stack([
            nn.functional.softplus(head(features).squeeze(-1))
            for head in self.heads
        ], dim=1)


def pick_device(requested: str = "auto", verbose: bool = True) -> torch.device:
    """Automatic hardware selection: CUDA -> Apple MPS -> CPU."""
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported device: {requested}")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if requested == "mps" and not (
            getattr(torch.backends, "mps", None) and
            torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is unavailable")
    if requested == "cpu":
        dev, name = torch.device("cpu"), "CPU"
    elif requested == "cuda" or (
            requested == "auto" and torch.cuda.is_available()):
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
    elif requested == "mps" or (
            requested == "auto" and getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()):
        dev, name = torch.device("mps"), "Apple MPS"
    else:
        dev, name = torch.device("cpu"), "CPU"
        torch.set_num_threads(max(1, torch.get_num_threads()))
    if verbose:
        print(f"[device] using {dev} ({name})")
    return dev
