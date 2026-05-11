"""
analyze.py — Visual diagnostics for Alzheimer's ViT model

Usage:
    python analyze.py --config configs/my_experiment.yaml

Outputs (saved to outputs/analysis/<experiment_name>/):
    - confusion_matrix.png         : Normalized + raw confusion matrix
    - per_class_metrics.png        : Bar chart of precision/recall/F1 per class
    - confidence_distribution.png : Softmax confidence histograms per class
    - confidence_vs_accuracy.png  : Reliability/calibration diagram
    - error_analysis.png           : Where mistakes go (misclassification flows)
    - top_errors.png               : Hardest samples (lowest confidence on correct class)
    - summary_report.txt           : Plain-text summary of all findings
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

from src.data import MRIPETDataset
from src.baseline_model import BaselineModel
from src.utils import read_config

# ──────────────────────────────────────────────
# Palette & style
# ──────────────────────────────────────────────
BACKGROUND = "#0d1117"
SURFACE     = "#161b22"
BORDER      = "#30363d"
ACCENT      = "#58a6ff"
GREEN       = "#3fb950"
RED         = "#f85149"
YELLOW      = "#d29922"
PURPLE      = "#bc8cff"
TEXT        = "#e6edf3"
TEXT_DIM    = "#8b949e"

CLASS_COLORS = [ACCENT, GREEN, YELLOW, RED]
CLASS_NAMES  = ["CN", "sMCI", "pMCI", "AD"]


def set_style():
    plt.rcParams.update({
        "figure.facecolor":  BACKGROUND,
        "axes.facecolor":    SURFACE,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   TEXT,
        "xtick.color":       TEXT_DIM,
        "ytick.color":       TEXT_DIM,
        "text.color":        TEXT,
        "grid.color":        BORDER,
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "font.family":       "monospace",
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "legend.facecolor":  SURFACE,
        "legend.edgecolor":  BORDER,
        "legend.labelcolor": TEXT,
    })


# ──────────────────────────────────────────────
# Inference — collect predictions + confidences
# ──────────────────────────────────────────────
def run_inference(model, loader, device):
    model.eval()
    all_preds   = []
    all_labels  = []
    all_probs   = []   # softmax probabilities (N, num_classes)
    all_subject = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Running inference"):
            mri    = batch["mri"].to(device)
            pet    = batch["pet"].to(device)
            labels = batch["label"]
            sids   = batch["subject_id"]

            logits = model(mri, pet)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            preds  = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
            all_subject.extend(sids if isinstance(sids, list) else sids.numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        all_subject,
    )


# ──────────────────────────────────────────────
# Plot 1 — Confusion matrix (raw + normalised)
# ──────────────────────────────────────────────
def plot_confusion_matrix(labels, preds, save_path):
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    cmap = LinearSegmentedColormap.from_list(
        "custom", [BACKGROUND, ACCENT], N=256
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BACKGROUND)
    fig.suptitle("Confusion Matrix", fontsize=16, color=TEXT, y=1.02)

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Raw counts", "Row-normalised (recall per class)"],
        [".0f", ".2f"],
    ):
        ax.set_facecolor(SURFACE)
        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0)

        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                val  = data[i, j]
                text = format(val, fmt)
                color = TEXT if (data[i, j] < data.max() * 0.6) else BACKGROUND
                ax.text(j, i, text, ha="center", va="center",
                        color=color, fontsize=12, fontweight="bold")

        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, fontsize=11)
        ax.set_yticklabels(CLASS_NAMES, fontsize=11)
        ax.set_xlabel("Predicted", labelpad=10)
        ax.set_ylabel("True",      labelpad=10)
        ax.set_title(title, color=TEXT_DIM, fontsize=11)

        # Diagonal highlight
        for k in range(len(CLASS_NAMES)):
            rect = plt.Rectangle(
                (k - 0.5, k - 0.5), 1, 1,
                fill=False, edgecolor=GREEN, linewidth=2
            )
            ax.add_patch(rect)

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color=TEXT_DIM)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_DIM)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BACKGROUND)
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# Plot 2 — Per-class precision / recall / F1
# ──────────────────────────────────────────────
def plot_per_class_metrics(labels, preds, save_path):
    precision = precision_score(labels, preds, average=None, zero_division=0)
    recall    = recall_score(   labels, preds, average=None, zero_division=0)
    f1        = f1_score(       labels, preds, average=None, zero_division=0)
    support   = np.bincount(labels, minlength=len(CLASS_NAMES))

    x      = np.arange(len(CLASS_NAMES))
    width  = 0.26

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(SURFACE)

    bars_p = ax.bar(x - width, precision, width, label="Precision",
                    color=ACCENT,  alpha=0.85, edgecolor=BACKGROUND)
    bars_r = ax.bar(x,          recall,   width, label="Recall",
                    color=GREEN,  alpha=0.85, edgecolor=BACKGROUND)
    bars_f = ax.bar(x + width,  f1,       width, label="F1",
                    color=PURPLE, alpha=0.85, edgecolor=BACKGROUND)

    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom",
                    fontsize=9, color=TEXT)

    # Support annotation below x-axis
    for i, s in enumerate(support):
        ax.text(i, -0.07, f"n={s}", ha="center", va="top",
                fontsize=9, color=TEXT_DIM,
                transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1", color=TEXT)
    ax.legend()
    ax.grid(axis="y")
    ax.axhline(0.5, color=YELLOW, linestyle=":", linewidth=1, alpha=0.6)

    # Mark worst F1 class
    worst = int(np.argmin(f1))
    ax.annotate(
        f"⚠ worst: {CLASS_NAMES[worst]}",
        xy=(worst + width, f1[worst]),
        xytext=(worst + width + 0.4, f1[worst] + 0.12),
        arrowprops=dict(arrowstyle="->", color=RED),
        color=RED, fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BACKGROUND)
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# Plot 3 — Confidence distributions per class
# ──────────────────────────────────────────────
def plot_confidence_distributions(labels, probs, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor(BACKGROUND)
    fig.suptitle("Confidence Distribution (softmax score on predicted class)",
                 fontsize=14, color=TEXT)

    for idx, ax in enumerate(axes.flat):
        ax.set_facecolor(SURFACE)
        mask    = labels == idx
        if mask.sum() == 0:
            ax.set_visible(False)
            continue

        conf_correct   = probs[mask & (np.argmax(probs, axis=1) == labels),  idx]
        conf_incorrect = probs[mask & (np.argmax(probs, axis=1) != labels),  idx]

        bins = np.linspace(0, 1, 25)
        ax.hist(conf_correct,   bins=bins, color=GREEN, alpha=0.7,
                label=f"Correct ({len(conf_correct)})")
        ax.hist(conf_incorrect, bins=bins, color=RED,   alpha=0.7,
                label=f"Wrong   ({len(conf_incorrect)})")

        ax.set_title(f"{CLASS_NAMES[idx]}  (total={mask.sum()})",
                     color=CLASS_COLORS[idx])
        ax.set_xlabel("Confidence on true class")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.grid(axis="y")

        mean_c = conf_correct.mean()   if len(conf_correct)   else 0
        mean_w = conf_incorrect.mean() if len(conf_incorrect) else 0
        ax.axvline(mean_c, color=GREEN, linestyle="--", linewidth=1.5)
        ax.axvline(mean_w, color=RED,   linestyle="--", linewidth=1.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BACKGROUND)
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# Plot 4 — Calibration / reliability diagram
# ──────────────────────────────────────────────
def plot_calibration(labels, probs, save_path, n_bins=10):
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(SURFACE)

    for cls_idx, (cls_name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        binary_labels = (labels == cls_idx).astype(int)
        cls_probs     = probs[:, cls_idx]

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_accs, bin_confs, bin_counts = [], [], []

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (cls_probs >= lo) & (cls_probs < hi)
            if mask.sum() > 0:
                bin_accs.append(binary_labels[mask].mean())
                bin_confs.append(cls_probs[mask].mean())
                bin_counts.append(mask.sum())

        if bin_confs:
            ax.plot(bin_confs, bin_accs, "o-", color=color,
                    label=cls_name, linewidth=2, markersize=6)

    ax.plot([0, 1], [0, 1], "--", color=TEXT_DIM, linewidth=1.5,
            label="Perfect calibration")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color=TEXT_DIM)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (accuracy)")
    ax.set_title("Calibration / Reliability Diagram", color=TEXT)
    ax.legend()
    ax.grid()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BACKGROUND)
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# Plot 5 — Misclassification flow (error heatmap)
# ──────────────────────────────────────────────
def plot_error_analysis(labels, preds, save_path):
    """Shows where errors go — only the off-diagonal cells."""
    cm      = confusion_matrix(labels, preds)
    errors  = cm.copy().astype(float)
    np.fill_diagonal(errors, 0)

    cmap = LinearSegmentedColormap.from_list(
        "err", [SURFACE, RED], N=256
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(SURFACE)

    im = ax.imshow(errors, cmap=cmap, aspect="auto")

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i == j:
                ax.text(j, i, "✓", ha="center", va="center",
                        color=GREEN, fontsize=16)
            else:
                val = errors[i, j]
                color = TEXT if val < errors.max() * 0.5 else BACKGROUND
                ax.text(j, i, f"{int(val)}", ha="center", va="center",
                        color=color, fontsize=13, fontweight="bold")

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels([f"Pred: {c}" for c in CLASS_NAMES], fontsize=11)
    ax.set_yticklabels([f"True: {c}" for c in CLASS_NAMES], fontsize=11)
    ax.set_title("Error Flow  (off-diagonal mistakes)", color=TEXT)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color=TEXT_DIM)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_DIM)

    # Annotate total errors per row
    row_errors = errors.sum(axis=1)
    for i, n in enumerate(row_errors):
        ax.text(len(CLASS_NAMES) - 0.35, i,
                f" → {int(n)} err", va="center",
                color=RED, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BACKGROUND)
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# Plot 6 — Hardest samples (top-N lowest confidence)
# ──────────────────────────────────────────────
def plot_hardest_samples(labels, preds, probs, subjects, save_path, top_n=20):
    correct_class_conf = probs[np.arange(len(labels)), labels]
    wrong_mask         = preds != labels
    wrong_indices      = np.where(wrong_mask)[0]

    if len(wrong_indices) == 0:
        print("  No errors found — skipping hardest samples plot.")
        return

    # Sort wrong predictions by confidence in the true class (ascending)
    sorted_idx  = wrong_indices[np.argsort(correct_class_conf[wrong_indices])]
    top_indices = sorted_idx[:top_n]

    fig, ax = plt.subplots(figsize=(12, max(5, len(top_indices) * 0.38)))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(SURFACE)

    y_pos = np.arange(len(top_indices))

    for i, idx in enumerate(top_indices):
        true_cls  = labels[idx]
        pred_cls  = preds[idx]
        conf_true = correct_class_conf[idx]
        conf_pred = probs[idx, pred_cls]

        # Bar for predicted confidence
        ax.barh(i, conf_pred, color=RED, alpha=0.7, height=0.6)
        # Bar for true class confidence
        ax.barh(i, conf_true, color=GREEN, alpha=0.7, height=0.6)

        label_text = (
            f"True: {CLASS_NAMES[true_cls]}  "
            f"Pred: {CLASS_NAMES[pred_cls]}  "
            f"[conf_true={conf_true:.2f} | conf_pred={conf_pred:.2f}]"
        )
        ax.text(1.01, i, label_text, va="center",
                color=TEXT_DIM, fontsize=8,
                transform=ax.get_yaxis_transform())

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"#{i+1}" for i in range(len(top_indices))],
                       fontsize=9)
    ax.set_xlabel("Confidence")
    ax.set_xlim(0, 1)
    ax.set_title(f"Top-{len(top_indices)} Hardest Errors  "
                 f"(sorted by confidence in true class ↑)",
                 color=TEXT)

    legend = [
        mpatches.Patch(color=GREEN, alpha=0.7, label="Conf. for true class"),
        mpatches.Patch(color=RED,   alpha=0.7, label="Conf. for predicted class"),
    ]
    ax.legend(handles=legend, loc="lower right")
    ax.grid(axis="x")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BACKGROUND)
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# Text summary
# ──────────────────────────────────────────────
def write_summary(labels, preds, probs, experiment_name, save_path):
    acc        = accuracy_score(labels, preds)
    macro_f1   = f1_score(labels, preds, average="macro",   zero_division=0)
    macro_rec  = recall_score(labels, preds, average="macro", zero_division=0)
    macro_prec = precision_score(labels, preds, average="macro", zero_division=0)

    per_f1   = f1_score(labels, preds, average=None, zero_division=0)
    per_prec = precision_score(labels, preds, average=None, zero_division=0)
    per_rec  = recall_score(labels, preds, average=None, zero_division=0)
    support  = np.bincount(labels, minlength=len(CLASS_NAMES))

    cm = confusion_matrix(labels, preds)
    mean_conf = probs[np.arange(len(labels)), preds].mean()

    lines = [
        "=" * 60,
        f"  ANALYSIS REPORT — {experiment_name}",
        "=" * 60,
        "",
        "OVERALL METRICS",
        f"  Accuracy      : {acc:.4f}",
        f"  Macro F1      : {macro_f1:.4f}",
        f"  Macro Recall  : {macro_rec:.4f}",
        f"  Macro Precision: {macro_prec:.4f}",
        f"  Mean confidence: {mean_conf:.4f}",
        "",
        "PER-CLASS BREAKDOWN",
        f"  {'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}",
        "  " + "-" * 50,
    ]
    for i, cls in enumerate(CLASS_NAMES):
        lines.append(
            f"  {cls:<8} {per_prec[i]:>10.4f} {per_rec[i]:>10.4f}"
            f" {per_f1[i]:>10.4f} {support[i]:>10}"
        )

    worst_f1  = CLASS_NAMES[int(np.argmin(per_f1))]
    worst_rec = CLASS_NAMES[int(np.argmin(per_rec))]

    lines += [
        "",
        "KEY WEAKNESSES",
        f"  Worst F1 class     : {worst_f1}",
        f"  Worst recall class : {worst_rec}",
        "",
        "CONFUSION MATRIX",
    ]
    header = "           " + "  ".join(f"{c:>6}" for c in CLASS_NAMES)
    lines.append(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>6}" for v in row)
        lines.append(f"  {CLASS_NAMES[i]:<8}   {row_str}")

    lines += [
        "",
        "TOP CONFUSION PAIRS (true → predicted)",
    ]
    cm_off = cm.copy(); np.fill_diagonal(cm_off, 0)
    flat   = [(cm_off[i, j], CLASS_NAMES[i], CLASS_NAMES[j])
              for i in range(len(CLASS_NAMES))
              for j in range(len(CLASS_NAMES)) if i != j]
    for cnt, true_cls, pred_cls in sorted(flat, reverse=True)[:6]:
        lines.append(f"  {true_cls} → {pred_cls} : {cnt} errors")

    lines += ["", "=" * 60]

    with open(save_path, "w") as f:
        f.write("\n".join(lines))

    # Also print
    print("\n" + "\n".join(lines))
    print(f"\n  Saved: {save_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Visual error analysis for Alzheimer ViT model"
    )
    parser.add_argument("--config",  type=str, required=True,
                        help="Path to experiment yaml")
    parser.add_argument("--system_config", type=str,
                        default="configs/kaggle.yaml",
                        help="Path to system/data yaml (default: configs/kaggle.yaml)")
    parser.add_argument("--split",   type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Which split to analyse (default: test)")
    parser.add_argument("--top_n",   type=int, default=20,
                        help="Number of hardest errors to show (default: 20)")
    args = parser.parse_args()

    # ── Load configs ────────────────────────────────
    config        = read_config(args.system_config)
    config_exp    = read_config(args.config)
    device        = config_exp["training"]["device"]
    exp_name      = config_exp["experiment"]["name"]

    # ── Output directory ────────────────────────────
    save_dir = Path("outputs") / "analysis" / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Analysing: {exp_name}")
    print(f"  Split    : {args.split}")
    print(f"  Device   : {device}")
    print(f"  Output   : {save_dir}")
    print(f"{'='*60}\n")

    # ── Dataset & split ─────────────────────────────
    dataset   = MRIPETDataset(root=config["data"]["root"])
    generator = torch.Generator().manual_seed(12345)

    train_size = int(config["split"]["train_ratio"] * len(dataset))
    val_size   = int(config["split"]["val_ratio"]   * len(dataset))
    test_size  = len(dataset) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    split_map = {"train": train_ds, "val": val_ds, "test": test_ds}
    chosen_ds = split_map[args.split]

    loader = DataLoader(
        chosen_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    print(f"  Dataset size ({args.split}): {len(chosen_ds)} samples\n")

    # ── Load model ──────────────────────────────────
    model_path = f"[{exp_name}].pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found: {model_path}\n"
            f"Train first with train.py --config {args.config}"
        )

    model = BaselineModel(
        class_num=config_exp["model"]["num_classes"],
        fusion_method=config_exp["model"]["fusion_type"],
    ).to(device)

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    print(f"  Loaded weights from: {model_path}\n")

    # ── Inference ───────────────────────────────────
    labels, preds, probs, subjects = run_inference(model, loader, device)

    # ── Plots ───────────────────────────────────────
    set_style()

    print("\nGenerating plots...")
    plot_confusion_matrix(
        labels, preds,
        save_dir / "confusion_matrix.png"
    )
    plot_per_class_metrics(
        labels, preds,
        save_dir / "per_class_metrics.png"
    )
    plot_confidence_distributions(
        labels, probs,
        save_dir / "confidence_distribution.png"
    )
    plot_calibration(
        labels, probs,
        save_dir / "calibration.png"
    )
    plot_error_analysis(
        labels, preds,
        save_dir / "error_analysis.png"
    )
    plot_hardest_samples(
        labels, preds, probs, subjects,
        save_dir / "hardest_samples.png",
        top_n=args.top_n,
    )
    write_summary(
        labels, preds, probs, exp_name,
        save_dir / "summary_report.txt"
    )

    print(f"\n✓ All outputs saved to: {save_dir}/")


if __name__ == "__main__":
    main()