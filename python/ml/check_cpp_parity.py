"""Verify that exported C++ inference matches the trained PyTorch members."""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from cadfs_py import load_engine  # noqa: E402
from ml.model import CostToGoNet, FastMultiHeadStudent  # noqa: E402


def patch_for(map_, x: int, y: int, size: int) -> np.ndarray:
    radius = size // 2
    patch = np.ones((size, size), dtype=np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            xx, yy = x + dx, y + dy
            if 0 <= xx < map_.width and 0 <= yy < map_.height:
                patch[dy + radius, dx + radius] = (
                    0.0 if map_.passable(xx, yy) else 1.0)
    return patch[None, None, :, :]


def extra_for(map_, x: int, y: int, gx: int, gy: int) -> np.ndarray:
    diagonal = math.hypot(map_.width, map_.height)
    dx, dy = gx - x, gy - y
    adx, ady = abs(dx), abs(dy)
    octile = adx + ady + (math.sqrt(2.0) - 2.0) * min(adx, ady)
    def density(cx: int, cy: int, radius: int) -> float:
        blocked = 0
        total = (2 * radius + 1) ** 2
        for yy in range(cy - radius, cy + radius + 1):
            for xx in range(cx - radius, cx + radius + 1):
                if not (0 <= xx < map_.width and 0 <= yy < map_.height) or not map_.passable(xx, yy):
                    blocked += 1
        return blocked / total

    x0, y0 = x, y
    line_dx, sx = abs(gx - x0), 1 if x0 < gx else -1
    line_dy, sy = -abs(gy - y0), 1 if y0 < gy else -1
    error = line_dx + line_dy
    blocked = total = 0
    while True:
        total += 1
        blocked += int(not map_.passable(x0, y0))
        if x0 == gx and y0 == gy:
            break
        twice = 2 * error
        if twice >= line_dy:
            error += line_dy
            x0 += sx
        if twice <= line_dx:
            error += line_dx
            y0 += sy

    return np.asarray([[
        dx / diagonal, dy / diagonal, math.hypot(dx, dy) / diagonal,
        octile / diagonal,
        blocked / max(1, total),
        density(x, y, 3),
        density(x, y, 7),
        density(x, y, 15),
        # The binding intentionally exposes only passability, so mirror the
        # engine's no-corner-cutting degree rule here.
        sum(
            1 for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if map_.passable(x + ddx, y + ddy)
        ) / 8.0 + sum(
            1 for ddx, ddy in ((1, 1), (1, -1), (-1, 1), (-1, -1))
            if map_.passable(x + ddx, y + ddy)
            and map_.passable(x + ddx, y)
            and map_.passable(x, y + ddy)
        ) / 8.0,
        density(gx, gy, 3),
    ]], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--backend", choices=["cnn", "fast"], default="cnn")
    parser.add_argument("--model")
    parser.add_argument("--checkpoints-dir", default="results/models")
    args = parser.parse_args()

    engine = load_engine(required=("run_cadfs_next",))
    model_path = ROOT / (
        args.model or (
            "results/models/fast_ensemble.txt"
            if args.backend == "fast"
            else "results/models/ensemble.txt"))
    ensemble = (
        engine.FastEnsembleGuidance(str(model_path))
        if args.backend == "fast"
        else engine.EnsembleGuidance(str(model_path)))
    checkpoints_dir = Path(args.checkpoints_dir)
    if not checkpoints_dir.is_absolute():
        checkpoints_dir = ROOT / checkpoints_dir
    if args.backend == "fast":
        checkpoint = torch.load(
            checkpoints_dir / "fast_student.pt",
            map_location="cpu", weights_only=False)
        config = checkpoint["model_config"]
        fast_net = FastMultiHeadStudent(
            patch=config["patch"], extra=config["extra"],
            hidden=tuple(config["hidden"]), heads=config["heads"])
        fast_net.load_state_dict(checkpoint["state_dict"])
        fast_net.eval()
        nets = [fast_net]
    else:
        checkpoints = sorted(checkpoints_dir.glob("member_*.pt"))
        if len(checkpoints) != ensemble.members:
            raise ValueError(
                f"checkpoint count {len(checkpoints)} != exported members "
                f"{ensemble.members}")
        nets = []
        for path in checkpoints:
            checkpoint = torch.load(
                path, map_location="cpu", weights_only=False)
            config = checkpoint["model_config"]
            net = CostToGoNet(
                patch=config["patch"], extra=config["extra"],
                hidden=config["hidden"])
            net.load_state_dict(checkpoint["state_dict"])
            net.eval()
            nets.append(net)

    with (ROOT / "data/instances" / f"{args.split}.csv").open(
            newline="", encoding="utf-8") as stream:
        instances = list(csv.DictReader(stream))
    rng = random.Random(args.seed)
    max_error = 0.0
    checked = 0
    for row in instances:
        map_ = engine.GridMap.load_movingai(str(ROOT / row["map_path"]))
        gx, gy = int(row["goal_x"]), int(row["goal_y"])
        free = [(x, y) for y in range(map_.height) for x in range(map_.width)
                if map_.passable(x, y)]
        rng.shuffle(free)
        for x, y in free[:min(4, len(free))]:
            patch = torch.from_numpy(patch_for(
                map_, x, y, ensemble.patch_size))
            extra = torch.from_numpy(extra_for(map_, x, y, gx, gy))
            with torch.no_grad():
                if args.backend == "fast":
                    expected = nets[0](patch, extra)[0].numpy().astype(
                        np.float64)
                else:
                    expected = np.asarray(
                        [net(patch, extra).item() for net in nets],
                        dtype=np.float64)
            actual = np.asarray(
                ensemble.raw_eval(map_, x, y, gx, gy), dtype=np.float64)
            error = float(np.max(np.abs(expected - actual)))
            max_error = max(max_error, error)
            checked += 1
            if error > args.atol:
                raise AssertionError(
                    f"Python/C++ parity error {error:.3e} at {(x, y)} "
                    f"on {row['map_id']} (atol={args.atol})")
            if checked >= args.samples:
                break
        if checked >= args.samples:
            break

    if checked == 0:
        raise RuntimeError("no passable samples were checked")
    print(f"parity OK: {checked} samples, max_abs_error={max_error:.3e}")
    print(f"engine: {engine.__file__}")
    count = getattr(ensemble, "members", getattr(ensemble, "heads", 0))
    print(
        f"model: backend={args.backend}, patch={ensemble.patch_size}, "
        f"members/heads={count}, "
        f"variance_scale={ensemble.variance_scale:.6g}")


if __name__ == "__main__":
    main()
