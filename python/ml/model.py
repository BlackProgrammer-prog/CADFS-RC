"""Small CNN + MLP cost-to-go regressor.

Architecture (deliberately restricted to ops that are trivial to mirror in the
C++ inference engine — Conv3x3/pad1, ReLU, MaxPool2, Linear):

  patch (1,15,15) -> Conv(1->8,3x3,pad1) -> ReLU -> MaxPool2   # (8,7,7)
                  -> Conv(8->16,3x3,pad1) -> ReLU -> MaxPool2  # (16,3,3)
                  -> flatten (144)
  concat extra(4) -> Linear(148->64) -> ReLU -> Linear(64->1)

Output: normalized cost-to-go y_hat ~ d*(n, goal)/diag, clipped to [0,1] at
inference to give H_L(n). Ensemble of K models -> mean H_L and variance.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CostToGoNet(nn.Module):
    def __init__(self, patch: int = 15, extra: int = 4, hidden: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        p2 = patch // 2 // 2                      # 15 -> 7 -> 3
        self.flat_dim = 16 * p2 * p2              # 144
        self.fc1 = nn.Linear(self.flat_dim + extra, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.act = nn.ReLU()

    def forward(self, patch: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        z = self.pool(self.act(self.conv1(patch)))
        z = self.pool(self.act(self.conv2(z)))
        z = torch.cat([z.flatten(1), extra], dim=1)
        z = self.act(self.fc1(z))
        return self.fc2(z).squeeze(-1)


def pick_device(verbose: bool = True) -> torch.device:
    """Automatic hardware selection: CUDA -> Apple MPS -> CPU."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev, name = torch.device("mps"), "Apple MPS"
    else:
        dev, name = torch.device("cpu"), "CPU"
        torch.set_num_threads(max(1, torch.get_num_threads()))
    if verbose:
        print(f"[device] using {dev} ({name})")
    return dev
