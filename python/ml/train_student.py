"""Train a low-latency shared-backbone student for C++ search.

The student is supervised by exact Dijkstra cost-to-go labels, optionally
distilled from the existing CNN ensemble, and optimized with a goal-consistent
pairwise ranking loss. CPU and CUDA/AMP training use the same checkpoints and
export format.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from ml.model import (  # noqa: E402
    DEFAULT_EXTRA,
    DEFAULT_PATCH,
    DEFAULT_STUDENT_HEADS,
    DEFAULT_STUDENT_HIDDEN,
    CostToGoNet,
    FastMultiHeadStudent,
    encode_target,
    pick_device,
    target_to_priority,
)

MODELS = ROOT / "results/models"
TRAIN_SPLITS = ("train", "train_structural")
VAL_SPLITS = ("val", "val_structural")
CALIBRATION_SPLIT = "val_shift"


class PairedGuidanceDataset(Dataset):
    def __init__(self, patch: torch.Tensor, extra: torch.Tensor,
                 target: torch.Tensor, group: torch.Tensor):
        self.patch = patch
        self.extra = extra
        self.target = target
        members: dict[int, list[int]] = {}
        for index, value in enumerate(group.tolist()):
            members.setdefault(int(value), []).append(index)
        partner = torch.arange(len(target))
        for indices in members.values():
            if len(indices) > 1:
                rotated = indices[1:] + indices[:1]
                partner[torch.tensor(indices)] = torch.tensor(rotated)
        self.partner = partner

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int):
        other = int(self.partner[index])
        return (
            self.patch[index], self.extra[index], self.target[index],
            self.patch[other], self.extra[other], self.target[other],
        )


def load_split(split: str, max_samples: int = 0, seed: int = 0):
    path = ROOT / "data/labels" / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run make_labels.py")
    with np.load(path) as archive:
        patch = torch.from_numpy(archive["patch"])
        extra = torch.from_numpy(archive["extra"])
        target = torch.from_numpy(archive["y"])
        group = torch.from_numpy(archive["meta"][:, 0].astype(np.int64))
    if patch.shape[1:] != (1, DEFAULT_PATCH, DEFAULT_PATCH):
        raise ValueError(f"{path} has an incompatible patch schema")
    if extra.shape[1] != DEFAULT_EXTRA:
        raise ValueError(f"{path} has an incompatible feature schema")
    if max_samples and len(target) > max_samples:
        generator = torch.Generator().manual_seed(seed)
        keep = torch.randperm(len(target), generator=generator)[:max_samples]
        patch, extra, target, group = (
            patch[keep], extra[keep], target[keep], group[keep])
    return patch, extra, target, group


def combine_training(max_samples: int, seed: int):
    chunks = [
        load_split(split, max_samples, seed + index)
        for index, split in enumerate(TRAIN_SPLITS)
    ]
    patches, extras, targets, groups, weights = [], [], [], [], []
    group_offset = 0
    for patch, extra, target, group in chunks:
        group = group - group.min() + group_offset
        group_offset = int(group.max()) + 1
        patches.append(patch)
        extras.append(extra)
        targets.append(target)
        groups.append(group)
        weights.append(torch.full(
            (len(target),), 1.0 / len(target), dtype=torch.double))
    return (
        PairedGuidanceDataset(
            torch.cat(patches), torch.cat(extras),
            torch.cat(targets), torch.cat(groups)),
        torch.cat(weights),
    )


def load_teacher(directory: Path, device: torch.device) -> list[CostToGoNet]:
    teachers = []
    for path in sorted(directory.glob("member_*.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["model_config"]
        model = CostToGoNet(
            patch=config["patch"], extra=config["extra"],
            hidden=config["hidden"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval().requires_grad_(False).to(device)
        teachers.append(model)
    if not teachers:
        raise FileNotFoundError(
            f"no teacher member checkpoints found in {directory}")
    return teachers


def autocast_context(device: torch.device, enabled: bool):
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(
        device_type=device.type, dtype=dtype,
        enabled=enabled and device.type in {"cpu", "cuda"})


@torch.no_grad()
def validation_loss(model, data, device: torch.device,
                    batch_size: int, amp: bool) -> float:
    patch, extra, target, _ = data
    loader = DataLoader(
        torch.utils.data.TensorDataset(patch, extra, target),
        batch_size=batch_size, shuffle=False,
        pin_memory=device.type == "cuda")
    total = 0.0
    count = 0
    model.eval()
    for patch_batch, extra_batch, target_batch in loader:
        patch_batch = patch_batch.to(
            device, dtype=torch.float32, non_blocking=True)
        extra_batch = extra_batch.to(device, non_blocking=True)
        encoded = encode_target(
            target_batch.to(device, non_blocking=True))
        with autocast_context(device, amp):
            prediction = model(patch_batch, extra_batch).mean(dim=1)
            loss = torch.nn.functional.smooth_l1_loss(
                prediction, encoded, reduction="sum")
        total += float(loss)
        count += len(target_batch)
    return total / max(1, count)


def ranking_loss(first: torch.Tensor, second: torch.Tensor,
                 first_target: torch.Tensor,
                 second_target: torch.Tensor) -> torch.Tensor:
    target_delta = second_target - first_target
    useful = target_delta.abs() > 1e-4
    if not torch.any(useful):
        return first.sum() * 0.0
    direction = target_delta[useful].sign()
    prediction_delta = second[useful] - first[useful]
    return torch.nn.functional.softplus(
        0.02 - direction * prediction_delta).mean()


def train(args, device: torch.device) -> FastMultiHeadStudent:
    dataset, sample_weights = combine_training(args.max_samples, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=device.type == "cuda")
    validation = {
        split: load_split(split, args.max_val_samples, args.seed + 100 + index)
        for index, split in enumerate(VAL_SPLITS)
    }

    model = FastMultiHeadStudent(
        hidden=(args.hidden1, args.hidden2), heads=args.heads).to(device)
    forward_model = (
        torch.compile(model, mode="reduce-overhead")
        if args.compile else model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp and device.type == "cuda")
    teachers = (
        load_teacher(Path(args.teacher_dir), device)
        if args.teacher_weight > 0 else [])

    best_score = math.inf
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    log: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        forward_model.train()
        running = 0.0
        seen = 0
        for batch in loader:
            p1, x1, y1, p2, x2, y2 = batch
            patch = torch.cat([p1, p2]).to(
                device, dtype=torch.float32, non_blocking=True)
            extra = torch.cat([x1, x2]).to(device, non_blocking=True)
            target = torch.cat([y1, y2]).to(device, non_blocking=True)
            encoded = encode_target(target)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.amp):
                prediction = forward_model(patch, extra)
                # Per-head bootstrap masks retain diversity with one backbone.
                mask = (
                    torch.rand(prediction.shape, device=device) <
                    args.bootstrap_probability).to(prediction.dtype)
                squared = torch.nn.functional.smooth_l1_loss(
                    prediction, encoded[:, None].expand_as(prediction),
                    reduction="none")
                supervised = (squared * mask).sum() / mask.sum().clamp_min(1)
                split = len(y1)
                rank = ranking_loss(
                    prediction[:split].mean(1),
                    prediction[split:].mean(1),
                    y1.to(device), y2.to(device))
                distillation = prediction.sum() * 0.0
                if teachers:
                    teacher_outputs = torch.stack([
                        teacher(patch, extra) for teacher in teachers
                    ]).mean(0)
                    distillation = torch.nn.functional.smooth_l1_loss(
                        prediction.mean(1), teacher_outputs)
                loss = (
                    args.supervised_weight * supervised +
                    args.rank_weight * rank +
                    args.teacher_weight * distillation)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach()) * len(target)
            seen += len(target)

        scores = {
            split: validation_loss(
                forward_model, data, device, args.batch_size * 2, args.amp)
            for split, data in validation.items()
        }
        selection = max(scores.values()) + 0.1 * sum(scores.values()) / len(scores)
        row = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            **{f"{key}_loss": value for key, value in scores.items()},
            "selection_score": selection,
        }
        log.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train={row['train_loss']:.6f} "
            f"val={scores['val']:.6f} "
            f"val-struct={scores['val_structural']:.6f} "
            f"score={selection:.6f}", flush=True)
        if selection < best_score - 1e-7:
            best_score = selection
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("student training did not produce a checkpoint")
    model.load_state_dict(best_state)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": 1,
        "architecture": "fast_shared_multihead",
        "model_config": {
            "patch": DEFAULT_PATCH,
            "extra": DEFAULT_EXTRA,
            "hidden": [args.hidden1, args.hidden2],
            "heads": args.heads,
        },
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "state_dict": best_state,
        "training": vars(args),
        "log": log,
    }, MODELS / "fast_student.pt")
    return model


@torch.no_grad()
def calibration(model: FastMultiHeadStudent, device: torch.device,
                batch_size: int, amp: bool,
                max_samples: int) -> dict:
    patch, extra, target, _ = load_split(
        CALIBRATION_SPLIT, max_samples, 991)
    predictions = []
    for start in range(0, len(target), batch_size):
        with autocast_context(device, amp):
            predictions.append(model(
                patch[start:start + batch_size].to(
                    device, dtype=torch.float32),
                extra[start:start + batch_size].to(device)).cpu())
    encoded = torch.cat(predictions)
    priority = target_to_priority(encoded).numpy()
    target_priority = (target / (1.0 + target)).numpy()
    mean = priority.mean(axis=1)
    raw_variance = priority.var(axis=1)
    squared_error = (mean - target_priority) ** 2
    design = np.column_stack([raw_variance, np.ones_like(raw_variance)])
    coefficients = np.linalg.lstsq(design, squared_error, rcond=None)[0]
    candidates: list[tuple[float, float]] = []
    if coefficients[0] >= 0 and coefficients[1] >= 0:
        candidates.append((float(coefficients[0]), float(coefficients[1])))
    candidates.append((0.0, float(squared_error.mean())))
    denominator = float(np.dot(raw_variance, raw_variance))
    scale_at_zero_floor = (
        max(0.0, float(np.dot(raw_variance, squared_error)) / denominator)
        if denominator > 0 else 0.0)
    candidates.append((scale_at_zero_floor, 0.0))
    scale, floor = min(
        candidates,
        key=lambda pair: float(np.mean(
            (pair[0] * raw_variance + pair[1] - squared_error) ** 2)))
    return {
        "schema_version": 1,
        "split": CALIBRATION_SPLIT,
        "n": len(target),
        "variance_scale": scale,
        "variance_floor": floor,
        "priority_mae": float(np.abs(mean - target_priority).mean()),
        "calibration_mse": float(np.mean(
            (scale * raw_variance + floor - squared_error) ** 2)),
    }


def dump_tensor(stream, tensor: torch.Tensor) -> None:
    stream.write(
        " ".join(f"{value:.9g}" for value in tensor.flatten().tolist()) +
        "\n")


def dump_fc(stream, layer: torch.nn.Linear) -> None:
    stream.write(f"FC {layer.in_features} {layer.out_features}\n")
    dump_tensor(stream, layer.weight.detach().cpu())
    dump_tensor(stream, layer.bias.detach().cpu())


def export(model: FastMultiHeadStudent, calibrated: dict, output: Path) -> None:
    model.eval().cpu()
    output.parent.mkdir(parents=True, exist_ok=True)
    h1, h2 = model.hidden
    with output.open("w", encoding="utf-8") as stream:
        stream.write("CADFS_FAST_ENSEMBLE 1\n")
        stream.write(
            f"PATCH {model.patch_size} EXTRA {model.extra_features} "
            f"HIDDEN1 {h1} HIDDEN2 {h2} HEADS {model.head_count} "
            f"TARGET LOG1P VARIANCE_SCALE {calibrated['variance_scale']:.17g} "
            f"VARIANCE_FLOOR {calibrated['variance_floor']:.17g}\n")
        dump_fc(stream, model.fc1)
        dump_fc(stream, model.fc2)
        for index, head in enumerate(model.heads):
            stream.write(f"HEAD {index}\n")
            dump_fc(stream, head)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"],
                        default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--heads", type=int, default=DEFAULT_STUDENT_HEADS)
    parser.add_argument("--hidden1", type=int,
                        default=DEFAULT_STUDENT_HIDDEN[0])
    parser.add_argument("--hidden2", type=int,
                        default=DEFAULT_STUDENT_HIDDEN[1])
    parser.add_argument("--bootstrap-probability", type=float, default=0.8)
    parser.add_argument("--supervised-weight", type=float, default=0.8)
    parser.add_argument("--rank-weight", type=float, default=0.2)
    parser.add_argument("--teacher-weight", type=float, default=0.0)
    parser.add_argument("--teacher-dir", default="results/models")
    parser.add_argument(
        "--artifacts-dir", default="results/models",
        help="checkpoint/calibration directory (use a temp dir for smoke runs)")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="per-training-split cap for smoke/development runs; 0 uses all")
    parser.add_argument(
        "--max-val-samples", type=int, default=0,
        help="per-validation-split cap; 0 uses all")
    parser.add_argument("--out", default="results/models/fast_ensemble.txt")
    args = parser.parse_args()
    if args.heads < 2:
        parser.error("--heads must be at least 2")
    if not 0.0 < args.bootstrap_probability <= 1.0:
        parser.error("--bootstrap-probability must be in (0,1]")
    for name in ("supervised_weight", "rank_weight", "teacher_weight"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    total_weight = (
        args.supervised_weight + args.rank_weight + args.teacher_weight)
    if total_weight <= 0:
        parser.error("at least one loss weight must be positive")
    if abs(total_weight - 1.0) > 1e-9:
        parser.error("loss weights must sum to 1")
    teacher_dir = Path(args.teacher_dir)
    args.teacher_dir = str(
        teacher_dir if teacher_dir.is_absolute() else ROOT / teacher_dir)
    return args


def main() -> None:
    global MODELS
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    MODELS = (
        artifacts_dir if artifacts_dir.is_absolute()
        else ROOT / artifacts_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    if args.amp and device.type not in {"cpu", "cuda"}:
        raise ValueError("--amp is supported only for CPU/CUDA profiles")
    print(
        f"profile: device={device.type} amp={args.amp} "
        f"compile={args.compile} teacher_weight={args.teacher_weight}",
        flush=True)
    model = train(args, device)
    model.eval()
    calibrated = calibration(
        model, device, args.batch_size * 2, args.amp,
        args.max_val_samples)
    with (MODELS / "calibration_fast.json").open(
            "w", encoding="utf-8") as stream:
        json.dump(calibrated, stream, indent=2)
    output = Path(args.out)
    if not output.is_absolute():
        output = ROOT / output
    export(model, calibrated, output)
    print(
        f"student -> {MODELS / 'fast_student.pt'}\n"
        f"export  -> {output}\n"
        f"calibration -> {MODELS / 'calibration_fast.json'}")


if __name__ == "__main__":
    main()
