"""
visualize.py — Training curve plots
=====================================

Usage
-----
python visualize.py \\
    --experiment_name  mri_pet_concat_ce

Reads  outputs/logs/<experiment_name>.csv
Saves  outputs/logs/<experiment_name>_loss_curve.png
       outputs/logs/<experiment_name>_metrics_curve.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot training curves from a saved CSV log",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--experiment_name", type=str, required=True,
                   help="Must match the name used at training time")
    return p.parse_args()


def check_volume(tensor):
    """
    Render axial / sagittal / coronal mid-slices of a 3-D or 4-D tensor.

    tensor : torch.Tensor  shape (1, H, W, D) or (H, W, D)
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    data    = tensor.detach().cpu().numpy()
    mid_idx = [dim // 2 for dim in data.shape]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(np.rot90(data[:, :, mid_idx[2]]), cmap="hot")
    axes[0].set_title(f"Axial — slice {mid_idx[2]}")

    axes[1].imshow(np.rot90(data[mid_idx[0], :, :]), cmap="hot")
    axes[1].set_title(f"Sagittal — slice {mid_idx[0]}")

    axes[2].imshow(np.rot90(data[:, mid_idx[1], :]), cmap="hot")
    axes[2].set_title(f"Coronal — slice {mid_idx[1]}")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_experiment(experiment_name: str):
    csv_path = Path("outputs") / "logs" / f"{experiment_name}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Log not found: {csv_path}")

    df       = pd.read_csv(csv_path)
    save_dir = csv_path.parent

    print(f"Loaded experiment log: {csv_path}")

    # ── Loss curve ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["epoch"], df["train_loss"], label="Train Loss")
    ax.plot(df["epoch"], df["val_loss"],   label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{experiment_name} — Loss Curve")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    loss_path = save_dir / f"{experiment_name}_loss_curve.png"
    plt.savefig(loss_path)
    plt.close()
    print(f"Saved: {loss_path}")

    # ── Metrics curve ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    # Support both old and new column names gracefully
    acc_col = "acc"      if "acc"    in df.columns else "accuracy"
    f1_col  = "f1"       if "f1"     in df.columns else "macro_f1"
    rec_col = "recall"   if "recall" in df.columns else "macro_recall"

    ax.plot(df["epoch"], df[acc_col], label="Accuracy")
    ax.plot(df["epoch"], df[f1_col],  label="Macro F1")
    ax.plot(df["epoch"], df[rec_col], label="Recall")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title(f"{experiment_name} — Validation Metrics")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    metrics_path = save_dir / f"{experiment_name}_metrics_curve.png"
    plt.savefig(metrics_path)
    plt.close()
    print(f"Saved: {metrics_path}")


def main():
    args = parse_args()
    plot_experiment(args.experiment_name)


if __name__ == "__main__":
    main()