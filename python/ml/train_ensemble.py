"""Train the deep ensemble (K models) and produce paper-ready figures.

Auto device: CUDA if available, else Apple MPS, else CPU (see model.pick_device).
Each ensemble member gets a different seed AND a bootstrap resample of the
training set (standard deep-ensemble recipe). Early stopping on val loss
(patience-based) prevents overfitting per member.

VARIANCE CALIBRATION (new): raw ensemble variance from bootstrap+seed-only
diversity underestimates predictive error under geometry shift (maze/narrow
maps unseen at training) -- members agree with each other while all being
wrong together. We measure this gap on `val_shift`, a MILD, seed-disjoint
maze/narrow proxy set that is never used for training weights and is
DISTINCT from the shift_family TEST set. The resulting scalar
`variance_calibration` rescales ensemble variance before it is turned into
confidence C(n) = exp(-variance * calibration / tau_c), by folding the
calibration factor into an effective temperature tau_c_effective saved to
results/models/calibration.json. No change to the C++ engine is needed:
tau_c_effective = tau_c / calibration is passed at run time instead of tau_c.

Outputs:
  results/models/member_k.pt            PyTorch checkpoints (best val epoch)
  results/models/train_log.csv          per-epoch losses
  results/models/calibration.json       variance_calibration factor + diagnostics
  results/figures/ml_training_curves.png
  results/figures/ml_pred_vs_true.png
  results/figures/ml_uncertainty_shift.png
  results/figures/ml_calibration.png    <- NEW: error vs raw/calibrated std under shift
  results/figures/ml_error_vs_std.png
  results/tables/ml_metrics.csv         MAE/RMSE/mean_std per split
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
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from ml.model import CostToGoNet, pick_device  # noqa: E402

FIG = ROOT / "results/figures"
TAB = ROOT / "results/tables"
MOD = ROOT / "results/models"
for d in (FIG, TAB, MOD):
    d.mkdir(parents=True, exist_ok=True)

# val_shift is the calibration proxy; shift_family/shift_density/shift_size are
# held out strictly for final reporting and must never influence calibration.
SPLITS_EVAL = ["test", "shift_density", "shift_size", "shift_family"]
CALIB_SPLIT = "val_shift"


def load_split(split: str):
    z = np.load(ROOT / "data/labels" / f"{split}.npz")
    return (torch.from_numpy(z["patch"]), torch.from_numpy(z["extra"]),
            torch.from_numpy(z["y"]))


def train_member(k: int, train, val, device, epochs: int, bs: int, lr: float,
                 log_rows: list, patience: int = 5) -> CostToGoNet:
    torch.manual_seed(1000 + k)
    np.random.seed(1000 + k)

    P, X, Y = train
    idx = torch.from_numpy(np.random.randint(0, len(Y), size=len(Y)))  # bootstrap
    ds = TensorDataset(P[idx], X[idx], Y[idx])
    dl = DataLoader(ds, batch_size=bs, shuffle=True,
                    pin_memory=(device.type == "cuda"))
    vP, vX, vY = (t.to(device) for t in val)

    net = CostToGoNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = torch.nn.SmoothL1Loss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for ep in range(epochs):
        net.train()
        tot = 0.0
        for p, x, y in dl:
            p, x, y = (p.to(device, non_blocking=True), x.to(device, non_blocking=True),
                      y.to(device, non_blocking=True))
            opt.zero_grad()
            loss = lossf(net(p, x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        train_loss = tot / len(ds)

        net.eval()
        with torch.no_grad():
            vloss = lossf(net(vP, vX), vY).item()

        log_rows.append({"member": k, "epoch": ep + 1,
                         "train_loss": train_loss, "val_loss": vloss})
        print(f"  member {k} epoch {ep + 1:2d}/{epochs} "
              f"train {train_loss:.5f} val {vloss:.5f}")

        if vloss < best_val:
            best_val = vloss
            best_state = copy.deepcopy(net.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            print(f"  early stopping member {k}: best val={best_val:.5f}")
            break

    net.load_state_dict(best_state)
    torch.save(best_state, MOD / f"member_{k}.pt")
    return net


@torch.no_grad()
def ensemble_predict(nets, P, X, device, bs=4096):
    preds = []
    for net in nets:
        net.eval()
        outs = []
        for i in range(0, len(P), bs):
            outs.append(net(P[i:i + bs].to(device), X[i:i + bs].to(device)).cpu())
        preds.append(torch.cat(outs))
    stack = torch.stack(preds)  # (K, N)
    return stack.mean(0).numpy(), stack.var(0, unbiased=False).numpy()


def compute_calibration(nets, device) -> dict:
    """Variance-calibration factor from the val_shift proxy (never test data).

    We fit a single positive scalar `calibration` such that
        E[(mu - y)^2]  ~=  calibration * E[variance]
    i.e. calibration = mean squared error / mean predicted variance on the
    calibration proxy split. calibration > 1 means the raw ensemble is
    overconfident (variance underestimates error) under geometry shift.
    """
    P, X, Y = load_split(CALIB_SPLIT)
    mu, var = ensemble_predict(nets, P, X, device)
    y = Y.numpy()
    mse = float(np.mean((mu - y) ** 2))
    mean_var = float(np.mean(var)) + 1e-12
    calibration = mse / mean_var
    return dict(split=CALIB_SPLIT, n=len(y), mse=mse, mean_variance=mean_var,
               variance_calibration=calibration,
               mae=float(np.mean(np.abs(mu - y))),
               mean_std=float(np.sqrt(var).mean()))


def make_figures(nets, device, calib: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # --- training curves ---
    log = pd.read_csv(MOD / "train_log.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for i, col in enumerate(["train_loss", "val_loss"]):
        sns.lineplot(log, x="epoch", y=col, hue="member",
                     palette="viridis", ax=ax[i], legend=(i == 1))
        ax[i].set_title(col.replace("_", " "))
        ax[i].set_ylabel("Smooth L1 loss" if i == 0 else "")
    fig.tight_layout()
    fig.savefig(FIG / "ml_training_curves.png", dpi=200)

    # --- evaluate all report splits (test + 3 held-out shift TEST sets) ---
    rows, frames = [], []
    for split in SPLITS_EVAL:
        P, X, Y = load_split(split)
        mu, var = ensemble_predict(nets, P, X, device)
        err = np.abs(mu - Y.numpy())
        std = np.sqrt(var)
        std_cal = np.sqrt(var * calib["variance_calibration"])
        rows.append(dict(split=split, n=len(Y),
                         mae=float(err.mean()),
                         rmse=float(np.sqrt(((mu - Y.numpy()) ** 2).mean())),
                         mean_std=float(std.mean()),
                         mean_std_calibrated=float(std_cal.mean())))
        frames.append(pd.DataFrame(dict(split=split, std=std, std_cal=std_cal,
                                        abs_err=err, y=Y.numpy(), pred=mu)))
    df = pd.concat(frames, ignore_index=True)
    pd.DataFrame(rows).to_csv(TAB / "ml_metrics.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nvariance_calibration (fit on {CALIB_SPLIT}) = "
          f"{calib['variance_calibration']:.2f}x")

    # --- pred vs true (in-distribution test) ---
    d = df[df.split == "test"].sample(min(4000, (df.split == "test").sum()),
                                      random_state=0)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.scatter(d.y, d.pred, s=4, alpha=0.25, edgecolors="none")
    lim = [0, max(d.y.max(), d.pred.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlabel="true normalized cost-to-go", ylabel="ensemble prediction",
           title="In-distribution test")
    fig.tight_layout()
    fig.savefig(FIG / "ml_pred_vs_true.png", dpi=200)

    # --- ensemble std under distribution shift (raw) ---
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    sns.violinplot(df, x="split", y="std", order=SPLITS_EVAL, cut=0,
                   inner="quartile", ax=ax, density_norm="width")
    ax.set(xlabel="", ylabel="raw ensemble std  $\\sigma_L(n)$",
           title="Predictive uncertainty (raw, uncalibrated)")
    fig.tight_layout()
    fig.savefig(FIG / "ml_uncertainty_shift.png", dpi=200)

    # --- NEW: calibration diagnostic -- MAE vs raw std and vs calibrated std ---
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    g = df.groupby("split", observed=True).agg(
        mae=("abs_err", "mean"), std=("std", "mean"), std_cal=("std_cal", "mean")
    ).reindex(SPLITS_EVAL).reset_index()
    for a, col, title in ((ax[0], "std", "raw std vs MAE"),
                          (ax[1], "std_cal", "calibrated std vs MAE")):
        sns.scatterplot(g, x=col, y="mae", hue="split", s=90, ax=a, legend=(a is ax[1]))
        a.set(xlabel=col, ylabel="MAE" if a is ax[0] else "", title=title)
    fig.suptitle(f"Variance calibration factor = {calib['variance_calibration']:.2f}x "
                 f"(fit on {CALIB_SPLIT})", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "ml_calibration.png", dpi=200)

    # --- error vs uncertainty deciles (calibrated std) ---
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    df["std_bin"] = pd.qcut(df["std_cal"], 10, duplicates="drop")
    gg = df.groupby("std_bin", observed=True)["abs_err"].mean().reset_index()
    gg["bin_center"] = [iv.mid for iv in gg.std_bin]
    sns.lineplot(gg, x="bin_center", y="abs_err", marker="o", ax=ax)
    ax.set(xlabel="calibrated ensemble std (decile bins)", ylabel="mean |error|",
           title="Calibrated uncertainty predicts error")
    fig.tight_layout()
    fig.savefig(FIG / "ml_error_vs_std.png", dpi=200)
    print(f"figures -> {FIG}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()

    device = pick_device()
    train = load_split("train")
    val = load_split("val")
    print(f"train samples: {len(train[2])}, val: {len(val[2])}")

    log_rows: list = []
    nets = [train_member(k, train, val, device, args.epochs, args.batch_size,
                         args.lr, log_rows, args.patience)
            for k in range(args.K)]
    with open(MOD / "train_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["member", "epoch", "train_loss", "val_loss"])
        w.writeheader()
        w.writerows(log_rows)

    calib = compute_calibration(nets, device)
    json.dump(calib, open(MOD / "calibration.json", "w"), indent=2)
    print(f"calibration -> {MOD / 'calibration.json'}  "
          f"(variance_calibration = {calib['variance_calibration']:.2f}x, "
          f"fit on {calib['n']} instances of {CALIB_SPLIT})")

    make_figures(nets, device, calib)


if __name__ == "__main__":
    main()
