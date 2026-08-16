"""Train and calibrate the CADFS deep ensemble without structural-test leakage.

Training domains:
  train             random grids
  train_structural  maze/narrow grids with moderate parameters

Model selection domains:
  val               random grids
  val_structural    disjoint moderate maze/narrow grids

Calibration domain:
  val_shift         disjoint mild structural shift

Final report-only domains:
  test, shift_density, shift_size, shift_family

The network predicts log1p(d*/diag). Search maps this value monotonically to
1-exp(-prediction), avoiding the old [0,1] target clipping failure on long
maze detours. Domain-balanced sampling and worst-domain early stopping prevent
the structural data from silently degrading random-grid accuracy.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    TensorDataset,
    WeightedRandomSampler,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from ml.model import (  # noqa: E402
    DEFAULT_EXTRA,
    DEFAULT_HIDDEN,
    DEFAULT_PATCH,
    MODEL_SCHEMA_VERSION,
    TARGET_TRANSFORM,
    CostToGoNet,
    decode_target,
    encode_target,
    pick_device,
    target_to_priority,
)

FIG = ROOT / "results/figures"
TAB = ROOT / "results/tables"
MOD = ROOT / "results/models"
for directory in (FIG, TAB, MOD):
    directory.mkdir(parents=True, exist_ok=True)

TRAIN_SPLITS = ("train", "train_structural")
VAL_SPLITS = ("val", "val_structural")
CALIB_SPLIT = "val_shift"
SPLITS_EVAL = ("test", "shift_density", "shift_size", "shift_family")


def load_split(split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    path = ROOT / "data/labels" / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; regenerate labels with make_labels.py")
    with np.load(path) as archive:
        patch = torch.from_numpy(archive["patch"])
        extra = torch.from_numpy(archive["extra"])
        target = torch.from_numpy(archive["y"])
    if patch.ndim != 4 or patch.shape[-2:] != (DEFAULT_PATCH, DEFAULT_PATCH):
        raise ValueError(
            f"{path} uses patch {tuple(patch.shape[-2:])}; expected "
            f"{DEFAULT_PATCH}x{DEFAULT_PATCH}. Regenerate labels.")
    if extra.ndim != 2 or extra.shape[1] != DEFAULT_EXTRA:
        raise ValueError(
            f"{path} has {extra.shape[1]} extra features; expected {DEFAULT_EXTRA}")
    if not torch.isfinite(target).all() or torch.any(target < 0):
        raise ValueError(f"{path} contains invalid cost-to-go targets")
    return patch, extra, target


def validation_loss(net: CostToGoNet, data, device: torch.device,
                    batch_size: int) -> float:
    loader = DataLoader(
        TensorDataset(*data), batch_size=batch_size, shuffle=False,
        pin_memory=(device.type == "cuda"))
    loss_sum = 0.0
    count = 0
    net.eval()
    with torch.no_grad():
        for patch, extra, target in loader:
            patch = patch.to(device, dtype=torch.float32, non_blocking=True)
            extra = extra.to(device, non_blocking=True)
            encoded = encode_target(target.to(device, non_blocking=True))
            prediction = net(patch, extra)
            loss_sum += torch.nn.functional.smooth_l1_loss(
                prediction, encoded, reduction="sum").item()
            count += len(target)
    return loss_sum / max(1, count)


def balanced_loader(train_domains: dict[str, tuple], member: int,
                    batch_size: int, structural_weight: float,
                    device: torch.device) -> DataLoader:
    datasets = [TensorDataset(*train_domains[name]) for name in TRAIN_SPLITS]
    random_mass = 1.0
    structural_mass = structural_weight
    masses = (random_mass, structural_mass)
    weights = torch.cat([
        torch.full((len(dataset),), mass / len(dataset), dtype=torch.double)
        for dataset, mass in zip(datasets, masses)
    ])
    generator = torch.Generator()
    generator.manual_seed(1000 + member)
    sampler = WeightedRandomSampler(
        weights, num_samples=sum(len(dataset) for dataset in datasets),
        replacement=True, generator=generator)
    return DataLoader(
        ConcatDataset(datasets), batch_size=batch_size, sampler=sampler,
        pin_memory=(device.type == "cuda"),
        num_workers=0)


def train_member(member: int, train_domains: dict, val_domains: dict,
                 device: torch.device, epochs: int, batch_size: int,
                 learning_rate: float, log_rows: list[dict], patience: int,
                 structural_weight: float, hidden: int, amp: bool,
                 compile_model: bool) -> CostToGoNet:
    torch.manual_seed(1000 + member)
    np.random.seed(1000 + member)

    loader = balanced_loader(
        train_domains, member, batch_size, structural_weight, device)
    net = CostToGoNet(
        patch=DEFAULT_PATCH, extra=DEFAULT_EXTRA, hidden=hidden).to(device)
    forward_net = (
        torch.compile(net, mode="reduce-overhead")
        if compile_model else net)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(2, patience // 3),
        min_lr=1e-6)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp and device.type == "cuda")
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    best_score = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        forward_net.train()
        loss_sum = 0.0
        count = 0
        for patch, extra, target in loader:
            patch = patch.to(device, dtype=torch.float32, non_blocking=True)
            extra = extra.to(device, non_blocking=True)
            encoded = encode_target(target.to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                    device_type=device.type, dtype=amp_dtype,
                    enabled=amp and device.type in {"cpu", "cuda"}):
                prediction = forward_net(patch, extra)
                loss = torch.nn.functional.smooth_l1_loss(
                    prediction, encoded)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(target)
            count += len(target)

        train_loss = loss_sum / max(1, count)
        val_random = validation_loss(
            forward_net, val_domains["val"], device, batch_size * 2)
        val_structural = validation_loss(
            forward_net, val_domains["val_structural"], device, batch_size * 2)
        # The max term protects the weaker domain.  The small mean term gives a
        # deterministic tie-break without allowing one domain to be ignored.
        selection_score = max(val_random, val_structural) + 0.1 * (
            val_random + val_structural) / 2.0
        scheduler.step(selection_score)
        current_lr = optimizer.param_groups[0]["lr"]
        log_rows.append({
            "member": member,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_random_loss": val_random,
            "val_structural_loss": val_structural,
            "selection_score": selection_score,
            "learning_rate": current_lr,
        })
        print(
            f"  member {member} epoch {epoch:2d}/{epochs} "
            f"train {train_loss:.6f} val-random {val_random:.6f} "
            f"val-struct {val_structural:.6f} score {selection_score:.6f} "
            f"lr {current_lr:.2e}", flush=True)

        if selection_score < best_score - 1e-7:
            best_score = selection_score
            best_state = copy.deepcopy(net.state_dict())
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            print(
                f"  early stopping member {member}: epoch={best_epoch} "
                f"best-score={best_score:.6f}", flush=True)
            break

    if best_state is None:
        raise RuntimeError(f"member {member} never produced a checkpoint")
    net.load_state_dict(best_state)
    checkpoint = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "target_transform": TARGET_TRANSFORM,
        "model_config": {
            "patch": DEFAULT_PATCH,
            "extra": DEFAULT_EXTRA,
            "hidden": hidden,
            "conv_channels": [8, 16],
        },
        "member": member,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "state_dict": best_state,
    }
    torch.save(checkpoint, MOD / f"member_{member}.pt")
    return net


@torch.no_grad()
def ensemble_outputs(nets: list[CostToGoNet], data, device: torch.device,
                     batch_size: int = 2048) -> dict[str, np.ndarray]:
    patch, extra, target = data
    member_outputs = []
    for net in nets:
        net.eval()
        outputs = []
        for start in range(0, len(patch), batch_size):
            outputs.append(net(
                patch[start:start + batch_size].to(
                    device, dtype=torch.float32),
                extra[start:start + batch_size].to(device)).cpu())
        member_outputs.append(torch.cat(outputs))

    encoded = torch.stack(member_outputs)
    decoded = decode_target(encoded)
    priority = target_to_priority(encoded)
    target_priority = target / (1.0 + target)
    return {
        "target": target.numpy(),
        "target_priority": target_priority.numpy(),
        "prediction": decoded.mean(0).numpy(),
        "prediction_priority": priority.mean(0).numpy(),
        "priority_variance": priority.var(0, unbiased=False).numpy(),
    }


def compute_calibration(nets: list[CostToGoNet], device: torch.device) -> dict:
    outputs = ensemble_outputs(nets, load_split(CALIB_SPLIT), device)
    residual = outputs["prediction_priority"] - outputs["target_priority"]
    mse = float(np.mean(residual ** 2))
    squared_error = residual ** 2
    member_variance = outputs["priority_variance"].astype(np.float64)
    # Non-negative least squares for E[error^2 | v] ~= scale*v + floor.
    # The floor captures shared ensemble bias; scale-only calibration otherwise
    # explodes when all members make the same OOD error.
    design = np.column_stack([member_variance, np.ones_like(member_variance)])
    unconstrained = np.linalg.lstsq(
        design, squared_error.astype(np.float64), rcond=None)[0]
    candidates: list[tuple[float, float]] = []
    if unconstrained[0] >= 0.0 and unconstrained[1] >= 0.0:
        candidates.append((float(unconstrained[0]), float(unconstrained[1])))
    candidates.append((0.0, float(np.mean(squared_error))))
    denominator = float(np.dot(member_variance, member_variance))
    scale_at_zero_floor = (
        max(0.0, float(np.dot(member_variance, squared_error)) / denominator)
        if denominator > 0.0 else 0.0)
    candidates.append((scale_at_zero_floor, 0.0))
    raw_scale, raw_floor = min(
        candidates,
        key=lambda pair: float(np.mean(
            (pair[0] * member_variance + pair[1] - squared_error) ** 2)))
    # Priority is bounded, hence both the variance and squared error are <= 1.
    variance_scale = float(np.clip(raw_scale, 0.0, 1e6))
    variance_floor = float(np.clip(raw_floor, 0.0, 1.0))
    calibrated_variance = variance_scale * member_variance + variance_floor
    return {
        "schema_version": 2,
        "split": CALIB_SPLIT,
        "n": len(residual),
        "priority_mse": mse,
        "mean_priority_variance": float(np.mean(member_variance)),
        "raw_variance_scale": raw_scale,
        "raw_variance_floor": raw_floor,
        "variance_scale": variance_scale,
        "variance_floor": variance_floor,
        "calibration_mse": float(np.mean(
            (calibrated_variance - squared_error) ** 2)),
        "decoded_mae": float(np.mean(np.abs(
            outputs["prediction"] - outputs["target"]))),
        "priority_mae": float(np.mean(np.abs(residual))),
    }


def make_figures(nets: list[CostToGoNet], device: torch.device,
                 calibration: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    log = pd.read_csv(MOD / "train_log.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5), sharey=True)
    for axis, column, title in zip(
            axes,
            ("train_loss", "val_random_loss", "val_structural_loss"),
            ("balanced training", "random validation", "structural validation")):
        sns.lineplot(log, x="epoch", y=column, hue="member", ax=axis,
                     legend=(axis is axes[-1]), palette="viridis")
        axis.set(title=title, ylabel="Smooth L1 on log1p target")
    fig.tight_layout()
    fig.savefig(FIG / "ml_training_curves.png", dpi=200)
    plt.close(fig)

    rows = []
    frames = []
    scale = calibration["variance_scale"]
    floor = calibration["variance_floor"]
    for split in SPLITS_EVAL:
        outputs = ensemble_outputs(nets, load_split(split), device)
        raw_error = np.abs(outputs["prediction"] - outputs["target"])
        priority_error = np.abs(
            outputs["prediction_priority"] - outputs["target_priority"])
        std = np.sqrt(outputs["priority_variance"])
        calibrated_std = np.sqrt(
            outputs["priority_variance"] * scale + floor)
        rows.append({
            "split": split,
            "n": len(raw_error),
            "mae": float(raw_error.mean()),
            "rmse": float(np.sqrt(np.mean(
                (outputs["prediction"] - outputs["target"]) ** 2))),
            "priority_mae": float(priority_error.mean()),
            "mean_priority_std": float(std.mean()),
            "mean_calibrated_priority_std": float(calibrated_std.mean()),
        })
        frames.append(pd.DataFrame({
            "split": split,
            "target": outputs["target"],
            "prediction": outputs["prediction"],
            "priority_error": priority_error,
            "priority_std": std,
            "calibrated_priority_std": calibrated_std,
        }))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(TAB / "ml_metrics.csv", index=False)
    print(metrics.to_string(index=False))
    data = pd.concat(frames, ignore_index=True)

    sample = data[data.split == "test"].sample(
        min(4000, int((data.split == "test").sum())), random_state=0)
    fig, axis = plt.subplots(figsize=(4.3, 4.0))
    axis.scatter(sample.target, sample.prediction, s=4, alpha=0.25,
                 edgecolors="none")
    limit = max(sample.target.max(), sample.prediction.max()) * 1.05
    axis.plot([0, limit], [0, limit], "k--", linewidth=1)
    axis.set(xlabel="true d*/diagonal", ylabel="ensemble prediction",
             title="In-distribution prediction")
    fig.tight_layout()
    fig.savefig(FIG / "ml_pred_vs_true.png", dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.0, 3.7))
    sns.violinplot(data, x="split", y="calibrated_priority_std", cut=0,
                   inner="quartile", density_norm="width", ax=axis)
    axis.set(xlabel="", ylabel="calibrated priority std",
             title="Predictive uncertainty under distribution shift")
    fig.tight_layout()
    fig.savefig(FIG / "ml_uncertainty_shift.png", dpi=200)
    plt.close(fig)

    grouped = data.groupby("split", observed=True).agg(
        priority_mae=("priority_error", "mean"),
        raw_std=("priority_std", "mean"),
        calibrated_std=("calibrated_priority_std", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    for axis, column, title in (
            (axes[0], "raw_std", "raw uncertainty"),
            (axes[1], "calibrated_std", "calibrated uncertainty")):
        sns.scatterplot(grouped, x=column, y="priority_mae", hue="split",
                        s=90, ax=axis, legend=(axis is axes[1]))
        axis.set(title=title, ylabel="priority MAE")
    fig.suptitle(
        f"Variance={calibration['variance_scale']:.3g}v+"
        f"{calibration['variance_floor']:.3g} "
        f"fit on {CALIB_SPLIT}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "ml_calibration.png", dpi=200)
    plt.close(fig)

    ordered = data.sort_values("calibrated_priority_std").copy()
    ordered["uncertainty_decile"] = pd.qcut(
        ordered["calibrated_priority_std"], 10, duplicates="drop")
    curve = ordered.groupby("uncertainty_decile", observed=True).agg(
        uncertainty=("calibrated_priority_std", "mean"),
        error=("priority_error", "mean")).reset_index()
    fig, axis = plt.subplots(figsize=(5.3, 3.7))
    sns.lineplot(curve, x="uncertainty", y="error", marker="o", ax=axis)
    axis.set(xlabel="calibrated uncertainty decile mean",
             ylabel="mean priority error",
             title="Uncertainty-error relationship")
    fig.tight_layout()
    fig.savefig(FIG / "ml_error_vs_std.png", dpi=200)
    plt.close(fig)
    print(f"figures -> {FIG}")


def main() -> None:
    global MOD
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--structural-weight", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"],
        default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument(
        "--artifacts-dir", default="results/models",
        help="versioned checkpoint/calibration directory")
    args = parser.parse_args()
    if args.K < 2:
        parser.error("--K must be at least 2 for uncertainty estimation")
    if args.structural_weight <= 0:
        parser.error("--structural-weight must be positive")
    artifacts_dir = Path(args.artifacts_dir)
    MOD = (
        artifacts_dir if artifacts_dir.is_absolute()
        else ROOT / artifacts_dir)
    MOD.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    if args.amp and device.type not in {"cpu", "cuda"}:
        parser.error("--amp is supported only for CPU/CUDA")
    train_domains = {split: load_split(split) for split in TRAIN_SPLITS}
    val_domains = {split: load_split(split) for split in VAL_SPLITS}
    print(
        "training samples: " + ", ".join(
            f"{name}={len(data[2]):,}" for name, data in train_domains.items()))
    print(
        "validation samples: " + ", ".join(
            f"{name}={len(data[2]):,}" for name, data in val_domains.items()))

    log_rows: list[dict] = []
    nets = [
        train_member(
            member, train_domains, val_domains, device, args.epochs,
            args.batch_size, args.lr, log_rows, args.patience,
            args.structural_weight, args.hidden, args.amp, args.compile)
        for member in range(args.K)
    ]
    fields = [
        "member", "epoch", "train_loss", "val_random_loss",
        "val_structural_loss", "selection_score", "learning_rate",
    ]
    with (MOD / "train_log.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(log_rows)

    calibration = compute_calibration(nets, device)
    with (MOD / "calibration.json").open("w", encoding="utf-8") as stream:
        json.dump(calibration, stream, indent=2)
    manifest = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "target_transform": TARGET_TRANSFORM,
        "model": {
            "patch": DEFAULT_PATCH,
            "extra": DEFAULT_EXTRA,
            "hidden": args.hidden,
            "members": args.K,
        },
        "training": {
            "splits": list(TRAIN_SPLITS),
            "validation_splits": list(VAL_SPLITS),
            "calibration_split": CALIB_SPLIT,
            "structural_weight": args.structural_weight,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "patience": args.patience,
            "device_requested": args.device,
            "device_resolved": device.type,
            "amp": args.amp,
            "compile": args.compile,
        },
    }
    with (MOD / "training_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    print(
        f"calibration -> {MOD / 'calibration.json'} "
        f"(variance={calibration['variance_scale']:.3g}v+"
        f"{calibration['variance_floor']:.3g})")
    make_figures(nets, device, calibration)


if __name__ == "__main__":
    main()
