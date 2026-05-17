"""
analyse.py — Visual diagnostics for Alzheimer's ViT model
==========================================================

All configuration is passed through CLI arguments — no YAML required.

Usage
-----
python analyse.py \\
    --experiment_name  mri_pet_concat_ce \\
    --data_root        /data/paired_npz \\
    --fusion_type      concat \\
    --loss             crossentropy \\
    --split            test

What this script does
---------------------
1.  Loads the trained model from  [experiment_name].pth
2.  Runs inference on the chosen split (default: test)
3.  Prints every diagnostic plot inline (Kaggle / Jupyter)
4.  Saves all plots + a subject-ID error log to
      outputs/analysis/<experiment_name>/

Outputs — CrossEntropy / Focal mode
-------------------------------------
  confusion_matrix.png
  per_class_metrics.png
  confidence_distribution.png
  calibration.png
  error_analysis.png
  hardest_samples.png
  confused_*_MRI/PET.png      scan comparison panels
  error_subjects.json
  summary_report.txt

Outputs — MSE mode
-------------------
  Same files, but:
    confidence_distribution.png  ->  mse_residual_distribution.png
    calibration.png              ->  mse_regression_scatter.png

Extending to a new loss / model
---------------------------------
Pass a custom ``decode_fn`` to ``run_inference``.  The rest of the
analysis pipeline only ever sees integer label arrays and (optionally)
a scores array, so it is completely decoupled from the model internals.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    precision_score, confusion_matrix,
)
from tqdm import tqdm

from src.data          import MRIPETDataset
from src.baseline_model import BaselineModel
from src.engine        import build_model, build_criterion
from src.utils         import set_global_seed


# ══════════════════════════════════════════════
# Palette & global style
# ══════════════════════════════════════════════
BACKGROUND  = "#0d1117"
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


def set_style():
    plt.rcParams.update({
        "figure.facecolor": BACKGROUND,
        "axes.facecolor":   SURFACE,
        "axes.edgecolor":   BORDER,
        "axes.labelcolor":  TEXT,
        "xtick.color":      TEXT_DIM,
        "ytick.color":      TEXT_DIM,
        "text.color":       TEXT,
        "grid.color":       BORDER,
        "grid.linestyle":   "--",
        "grid.alpha":       0.5,
        "font.family":      "monospace",
        "axes.titlesize":   13,
        "axes.labelsize":   11,
        "legend.facecolor": SURFACE,
        "legend.edgecolor": BORDER,
        "legend.labelcolor":TEXT,
    })


def _savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    plt.show()
    plt.close(fig)
    print(f"  Saved -> {path}\n")


# ══════════════════════════════════════════════
# Inference  —  decode_fn is injected externally
# ══════════════════════════════════════════════

def run_inference(model, loader, device, decode_fn, score_fn, num_classes=4):
    """
    Run the model on `loader` and collect everything needed for analysis.

    Parameters
    ----------
    model      : nn.Module (already on `device`, eval mode set here)
    loader     : DataLoader
    device     : str
    decode_fn  : callable(outputs) -> 1-D LongTensor of predicted class indices
                 Injected by the caller, so this function is loss-agnostic.
    score_fn   : callable(outputs) -> ndarray
                 Returns the per-sample score used for confidence/calibration
                 plots.  For CE → (N, C) softmax array.
                 For MSE / other → (N,) float array.
                 Return None to skip score-dependent plots.
    num_classes : int

    Returns
    -------
    labels   : np.ndarray (N,)
    preds    : np.ndarray (N,)
    scores   : np.ndarray | None
    subjects : list
    buckets  : dict  { (true_cls, pred_cls): [{"mri","pet","subject_id","conf"}, ...] }
    """
    MAX_STORE = 6
    model.eval()

    all_preds, all_labels, all_scores, all_subjects = [], [], [], []
    buckets = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            mri    = batch["mri"].to(device)
            pet    = batch["pet"].to(device)
            labels = batch["label"]
            sids   = batch["subject_id"]

            outputs = model(mri, pet)

            preds_t  = decode_fn(outputs).cpu()
            scores_t = score_fn(outputs)          # ndarray or None

            for i in range(len(labels)):
                t   = int(labels[i])
                p   = int(preds_t[i])
                sid = sids[i].item() if hasattr(sids[i], "item") else sids[i]

                all_preds.append(p)
                all_labels.append(t)
                all_subjects.append(sid)

                if scores_t is not None:
                    all_scores.append(scores_t[i])
                    # conf: scalar closeness to correct class
                    if scores_t.ndim == 2:
                        conf_val = float(scores_t[i, p])
                    else:
                        residual = abs(float(scores_t[i]) - t)
                        conf_val = max(0.0, 1.0 - residual / max(num_classes - 1, 1))
                else:
                    conf_val = float(p == t)

                key = (t, p)
                if key not in buckets:
                    buckets[key] = []
                if len(buckets[key]) < MAX_STORE:
                    buckets[key].append({
                        "mri":        mri[i].cpu(),
                        "pet":        pet[i].cpu(),
                        "subject_id": sid,
                        "conf":       conf_val,
                    })

    scores_arr = np.array(all_scores) if all_scores else None
    return (
        np.array(all_labels),
        np.array(all_preds),
        scores_arr,
        all_subjects,
        buckets,
    )


def _make_decode_and_score(args):
    """
    Build the two callables that ``run_inference`` needs, driven by args.loss.

    decode_fn(outputs) -> LongTensor
    score_fn(outputs)  -> np.ndarray | None

    This is the only place in analyse.py that knows about the loss type.
    """
    if args.loss in ("crossentropy", "focal"):
        def decode_fn(outputs):
            return torch.argmax(outputs, dim=1)

        def score_fn(outputs):
            return F.softmax(outputs, dim=1).cpu().numpy()

    elif args.loss == "mse":
        def decode_fn(outputs):
            return (
                outputs.squeeze(1)
                       .round()
                       .long()
                       .clamp(0, args.num_classes - 1)
                       .cpu()
            )

        def score_fn(outputs):
            return outputs.squeeze(1).cpu().numpy()

    else:
        raise ValueError(f"Unknown loss '{args.loss}'")

    return decode_fn, score_fn


# ══════════════════════════════════════════════
# Plot helpers  (unchanged from original)
# ══════════════════════════════════════════════

def plot_confusion_matrix(labels, preds, class_names, save_path):
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cmap    = LinearSegmentedColormap.from_list("cm", [BACKGROUND, ACCENT], N=256)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor(BACKGROUND)
    fig.suptitle("Confusion Matrix", fontsize=16, color=TEXT, y=1.01)

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Raw counts", "Row-normalised  (diagonal = per-class recall)"],
        [".0f", ".2f"],
    ):
        ax.set_facecolor(SURFACE)
        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                v     = data[i, j]
                color = TEXT if v < data.max() * 0.6 else BACKGROUND
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        color=color, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=11)
        ax.set_yticklabels(class_names, fontsize=11)
        ax.set_xlabel("Predicted", labelpad=10)
        ax.set_ylabel("True",      labelpad=10)
        ax.set_title(title, color=TEXT_DIM, fontsize=11)
        for k in range(len(class_names)):
            ax.add_patch(plt.Rectangle(
                (k - 0.5, k - 0.5), 1, 1,
                fill=False, edgecolor=GREEN, linewidth=2
            ))
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color=TEXT_DIM)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_DIM)

    plt.tight_layout()
    _savefig(fig, save_path)


def plot_per_class_metrics(labels, preds, class_names, save_path):
    precision = precision_score(labels, preds, average=None, zero_division=0)
    recall    = recall_score(   labels, preds, average=None, zero_division=0)
    f1        = f1_score(       labels, preds, average=None, zero_division=0)
    support   = np.bincount(labels, minlength=len(class_names))

    x, w = np.arange(len(class_names)), 0.26
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(SURFACE)

    for offset, vals, color, label in [
        (-w, precision, ACCENT,  "Precision"),
        ( 0, recall,    GREEN,   "Recall"),
        ( w, f1,        PURPLE,  "F1"),
    ]:
        bars = ax.bar(x + offset, vals, w, color=color,
                      alpha=0.85, edgecolor=BACKGROUND, label=label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=TEXT)

    for i, s in enumerate(support):
        ax.text(i, -0.07, f"n={s}", ha="center", va="top", fontsize=9,
                color=TEXT_DIM, transform=ax.get_xaxis_transform())

    worst = int(np.argmin(f1))
    ax.annotate(
        f"  worst F1: {class_names[worst]}",
        xy=(worst + w, f1[worst]),
        xytext=(worst + w + 0.35, f1[worst] + 0.12),
        arrowprops=dict(arrowstyle="->", color=RED),
        color=RED, fontsize=10,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1", color=TEXT)
    ax.legend()
    ax.grid(axis="y")
    ax.axhline(0.5, color=YELLOW, linestyle=":", linewidth=1, alpha=0.6)
    plt.tight_layout()
    _savefig(fig, save_path)


def plot_confidence_distributions(labels, probs, class_names, save_path):
    """CE / Focal mode only — skipped gracefully if probs is None or 1-D."""
    if probs is None or probs.ndim != 2:
        print("  [skip] confidence_distribution: requires 2-D softmax scores")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor(BACKGROUND)
    fig.suptitle("Confidence Distribution (softmax on true class)",
                 fontsize=14, color=TEXT)
    bins = np.linspace(0, 1, 25)

    for idx, ax in enumerate(axes.flat):
        ax.set_facecolor(SURFACE)
        if idx >= len(class_names):
            ax.set_visible(False)
            continue
        mask = labels == idx
        if mask.sum() == 0:
            ax.set_visible(False)
            continue
        correct_mask   = mask & (np.argmax(probs, axis=1) == labels)
        incorrect_mask = mask & (np.argmax(probs, axis=1) != labels)
        conf_c = probs[correct_mask,   idx]
        conf_w = probs[incorrect_mask, idx]
        ax.hist(conf_c, bins=bins, color=GREEN, alpha=0.7,
                label=f"Correct ({len(conf_c)})")
        ax.hist(conf_w, bins=bins, color=RED,   alpha=0.7,
                label=f"Wrong   ({len(conf_w)})")
        if len(conf_c) > 0:
            ax.axvline(conf_c.mean(), color=GREEN, linestyle="--", linewidth=1.5)
        if len(conf_w) > 0:
            ax.axvline(conf_w.mean(), color=RED,   linestyle="--", linewidth=1.5)
        ax.set_title(f"{class_names[idx]}  (total={mask.sum()})",
                     color=CLASS_COLORS[min(idx, len(CLASS_COLORS)-1)])
        ax.set_xlabel("Confidence on true class")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.grid(axis="y")

    plt.tight_layout()
    _savefig(fig, save_path)


def plot_calibration(labels, probs, class_names, save_path, n_bins=10):
    """CE / Focal mode only."""
    if probs is None or probs.ndim != 2:
        print("  [skip] calibration: requires 2-D softmax scores")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, 1], color=TEXT_DIM, linestyle="--",
            linewidth=1, label="Perfect calibration")

    bin_edges = np.linspace(0, 1, n_bins + 1)

    for cls_idx, (cls_name, color) in enumerate(
            zip(class_names, CLASS_COLORS)):
        binary  = (labels == cls_idx).astype(int)
        conf    = probs[:, cls_idx]
        xs, ys  = [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (conf >= lo) & (conf < hi)
            if mask.sum() == 0:
                continue
            xs.append(conf[mask].mean())
            ys.append(binary[mask].mean())
        ax.plot(xs, ys, "o-", color=color, label=cls_name, linewidth=2)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction")
    ax.set_title("Reliability / Calibration Diagram", color=TEXT)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    _savefig(fig, save_path)


def plot_mse_residuals(labels, raw_scores, class_names, save_path):
    """MSE mode equivalent of confidence_distribution."""
    if raw_scores is None or raw_scores.ndim != 1:
        print("  [skip] mse_residual_distribution: requires 1-D raw scores")
        return

    residuals = raw_scores - labels.astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BACKGROUND)

    ax = axes[0]
    ax.set_facecolor(SURFACE)
    ax.hist(residuals, bins=40, color=ACCENT, edgecolor=BACKGROUND, alpha=0.85)
    ax.axvline(0, color=RED, linestyle="--")
    ax.set_title("Residuals  (predicted − true)", color=TEXT)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.grid(axis="y")

    ax = axes[1]
    ax.set_facecolor(SURFACE)
    ax.scatter(labels + np.random.uniform(-0.15, 0.15, len(labels)),
               raw_scores, alpha=0.5, s=18, c=ACCENT)
    lo, hi = labels.min() - 0.5, labels.max() + 0.5
    ax.plot([lo, hi], [lo, hi], color=RED, linestyle="--")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_xlabel("True class (jittered)")
    ax.set_ylabel("Raw predicted scalar")
    ax.set_title("Regression scatter", color=TEXT)
    ax.grid(True)

    plt.tight_layout()
    _savefig(fig, save_path)


def plot_error_analysis(labels, preds, class_names, save_path):
    cm     = confusion_matrix(labels, preds)
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(SURFACE)
    cmap = LinearSegmentedColormap.from_list("err", [SURFACE, RED], N=256)
    ax.imshow(cm_off, cmap=cmap, aspect="auto")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm_off[i, j] > 0:
                ax.text(j, i, str(cm_off[i, j]), ha="center", va="center",
                        fontsize=12, color=TEXT, fontweight="bold")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted (wrong)")
    ax.set_ylabel("True")
    ax.set_title("Error Flow  (off-diagonal mistakes)", color=TEXT)
    plt.tight_layout()
    _savefig(fig, save_path)


def plot_hardest_samples(labels, preds, scores, subjects,
                         class_names, save_path, top_n=20):
    """Show subjects where the model was most confidently wrong."""
    wrong = np.where(labels != preds)[0]
    if len(wrong) == 0:
        print("  [skip] hardest_samples: no errors found")
        return

    if scores is not None and scores.ndim == 2:
        wrong_conf = scores[wrong, preds[wrong]]
    elif scores is not None:
        wrong_conf = np.abs(scores[wrong] - labels[wrong].astype(float))
        wrong_conf = 1.0 - wrong_conf / max(len(class_names) - 1, 1)
    else:
        wrong_conf = np.ones(len(wrong))

    order   = np.argsort(-wrong_conf)[:top_n]
    indices = wrong[order]

    fig, ax = plt.subplots(figsize=(14, max(4, top_n * 0.35)))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(SURFACE)

    y    = np.arange(len(indices))
    conf = wrong_conf[order]
    bars = ax.barh(y, conf, color=RED, alpha=0.85, edgecolor=BACKGROUND)

    for i, (idx, c) in enumerate(zip(indices, conf)):
        label = f"ID={subjects[idx]}  true={class_names[labels[idx]]}  pred={class_names[preds[idx]]}"
        ax.text(c + 0.005, i, label, va="center", fontsize=8, color=TEXT)

    ax.set_yticks([])
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Confidence on wrong prediction")
    ax.set_title(f"Top-{top_n} most confidently wrong predictions", color=TEXT)
    plt.tight_layout()
    _savefig(fig, save_path)


def _mid_slices(tensor):
    """Return axial mid-slice as (H, W) numpy array from (1,H,W,D) tensor."""
    vol = tensor.squeeze(0).numpy()
    return np.rot90(vol[:, :, vol.shape[2] // 2])


def plot_confused_pairs(buckets, class_names, save_dir, max_samples=3):
    """Render scan comparison panels for the most common confusion pairs."""
    # Only pairs that actually appear in the data
    pairs = [
        (k, v) for k, v in buckets.items()
        if k[0] != k[1] and len(v) > 0
    ]
    # Sort by error count descending, take top 4
    pairs.sort(key=lambda x: -len(x[1]))

    for (true_cls, pred_cls), samples in pairs[:4]:
        n = min(len(samples), max_samples)
        true_name = class_names[true_cls]
        pred_name = class_names[pred_cls]

        fig, axes = plt.subplots(2, n, figsize=(5 * n, 9))
        fig.patch.set_facecolor(BACKGROUND)
        fig.suptitle(
            f"True: {true_name}  →  Predicted: {pred_name}  ({n} samples)",
            fontsize=13, color=TEXT,
        )

        if n == 1:
            axes = axes.reshape(2, 1)

        for col, sample in enumerate(samples[:n]):
            for row, (key, cmap) in enumerate([("mri", "gray"), ("pet", "hot")]):
                ax = axes[row, col]
                ax.imshow(_mid_slices(sample[key]), cmap=cmap)
                ax.set_title(
                    f"{key.upper()}  conf={sample['conf']:.2f}",
                    color=TEXT, fontsize=9,
                )
                ax.axis("off")

        plt.tight_layout()
        tag = f"confused_{true_name.lower()}_as_{pred_name.lower()}"
        _savefig(fig, save_dir / f"{tag}.png")


def plot_misclassification_comparison(buckets, class_names, save_dir):
    """
    For each requested confusion pair, produce one figure where two subjects
    are shown side-by-side — one from each true class — but both were
    predicted as the same (wrong) class.

    Layout per figure  (8 columns):
    ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
    │ MRI      │ MRI      │ PET      │ PET      │ MRI      │ MRI      │ PET      │ PET      │
    │ class A  │ hist     │ class A  │ hist     │ class B  │ hist     │ class B  │ hist     │
    └──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘

    Both subjects were predicted as `pred_cls`.

    Requested pairs
    ---------------
    (CN,   sMCI) both predicted as CN        → confused_cn_smci_both_as_cn.png
    (CN,   sMCI) both predicted as sMCI      → confused_cn_smci_both_as_smci.png
    (pMCI, AD)   both predicted as AD        → confused_pmci_ad_both_as_ad.png
    (pMCI, AD)   both predicted as pMCI      → confused_pmci_ad_both_as_pmci.png
    (sMCI, pMCI) both predicted as sMCI      → confused_smci_pmci_both_as_smci.png

    Gracefully skipped if either bucket is empty (not enough misclassified
    samples for a given pair exist in the test set).
    """

    def _vol_to_slice(tensor):
        """Axial mid-slice, shape (H, W), from (1, H, W, D) or (H, W, D)."""
        vol = tensor.squeeze(0).numpy()
        return np.rot90(vol[:, :, vol.shape[2] // 2])

    def _draw_image_and_hist(axes, col, volume_tensor, modality_label,
                             class_label, img_cmap, accent_color):
        """
        Fill two adjacent axes (axes[col], axes[col+1]) with:
          left  — axial mid-slice of the volume
          right — intensity histogram of the full 3-D volume
        """
        vol    = volume_tensor.squeeze(0).numpy()          # (H, W, D)
        slc    = np.rot90(vol[:, :, vol.shape[2] // 2])   # axial mid-slice

        # ── image ──────────────────────────────────────────────────────────
        ax_img = axes[col]
        ax_img.imshow(slc, cmap=img_cmap, aspect="equal")
        ax_img.set_title(f"{modality_label}\n{class_label}",
                         color=TEXT, fontsize=8, pad=4)
        ax_img.axis("off")
        ax_img.set_facecolor(SURFACE)

        # ── histogram ──────────────────────────────────────────────────────
        ax_hist = axes[col + 1]
        ax_hist.set_facecolor(SURFACE)
        flat = vol.flatten()
        # clip extreme tails for readability (1st–99th percentile)
        lo, hi = np.percentile(flat, 1), np.percentile(flat, 99)
        clipped = flat[(flat >= lo) & (flat <= hi)]
        ax_hist.hist(clipped, bins=60, color=accent_color,
                     alpha=0.85, edgecolor=BACKGROUND, linewidth=0.3)
        ax_hist.axvline(clipped.mean(), color=TEXT, linestyle="--",
                        linewidth=1, label=f"μ={clipped.mean():.3f}")
        ax_hist.set_xlabel("Intensity", color=TEXT_DIM, fontsize=7)
        ax_hist.set_ylabel("Count",     color=TEXT_DIM, fontsize=7)
        ax_hist.tick_params(colors=TEXT_DIM, labelsize=6)
        ax_hist.set_title(f"Histogram\n{class_label}",
                          color=TEXT, fontsize=8, pad=4)
        ax_hist.legend(fontsize=6, loc="upper right")
        ax_hist.grid(axis="y", alpha=0.3)
        for spine in ax_hist.spines.values():
            spine.set_edgecolor(BORDER)

    # ── define the five requested pairs ────────────────────────────────────
    # Each entry: (true_cls_A, true_cls_B, pred_cls, filename_tag)
    # true_cls_A and true_cls_B are indices into class_names.
    # pred_cls is also an index — both subjects must have been predicted as this.
    def _idx(name):
        try:
            return class_names.index(name)
        except ValueError:
            return None   # class not present in this experiment's class_names

    CN, sMCI, pMCI, AD = _idx("CN"), _idx("sMCI"), _idx("pMCI"), _idx("AD")

    requested = []
    if None not in (CN, sMCI):
        requested.append((CN, sMCI, CN,   "confused_cn_smci_both_as_cn"))
        requested.append((CN, sMCI, sMCI, "confused_cn_smci_both_as_smci"))
    if None not in (pMCI, AD):
        requested.append((pMCI, AD, AD,   "confused_pmci_ad_both_as_ad"))
        requested.append((pMCI, AD, pMCI, "confused_pmci_ad_both_as_pmci"))
    if None not in (sMCI, pMCI):
        requested.append((sMCI, pMCI, sMCI, "confused_smci_pmci_both_as_smci"))

    mri_cmaps  = ["gray",  "bone",  "gray",  "bone",  "gray"]
    pet_cmaps  = ["hot",   "hot",   "hot",   "hot",   "hot"]
    # per-class accent colours for histograms
    hist_colors = {
        CN:   ACCENT,
        sMCI: GREEN,
        pMCI: YELLOW,
        AD:   RED,
    }

    for idx_req, (true_a, true_b, pred_cls, tag) in enumerate(requested):

        # bucket key: (true_class, predicted_class)
        sample_a = buckets.get((true_a, pred_cls), [])
        sample_b = buckets.get((true_b, pred_cls), [])

        if not sample_a or not sample_b:
            name_a = class_names[true_a]
            name_b = class_names[true_b]
            pred_n = class_names[pred_cls]
            print(f"  [skip] {tag}: no samples found where "
                  f"{name_a}→{pred_n} and {name_b}→{pred_n} both exist")
            continue

        s_a    = sample_a[0]   # take the first stored sample for class A
        s_b    = sample_b[0]   # take the first stored sample for class B

        name_a = class_names[true_a]
        name_b = class_names[true_b]
        pred_n = class_names[pred_cls]

        mri_cmap = mri_cmaps[idx_req % len(mri_cmaps)]
        pet_cmap = pet_cmaps[idx_req % len(pet_cmaps)]

        # ── 8-column figure ────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 8, figsize=(24, 4))
        fig.patch.set_facecolor(BACKGROUND)
        for ax in axes:
            ax.set_facecolor(SURFACE)

        fig.suptitle(
            f"Both predicted as  {pred_n}  "
            f"(left: true {name_a} | right: true {name_b})\n"
            f"Subject {name_a}: ID={s_a['subject_id']}  conf={s_a['conf']:.2f}   "
            f"Subject {name_b}: ID={s_b['subject_id']}  conf={s_b['conf']:.2f}",
            fontsize=10, color=TEXT, y=1.02,
        )

        # columns 0-1 : MRI of class A + its histogram
        _draw_image_and_hist(axes, 0, s_a["mri"], "MRI", name_a,
                             mri_cmap, hist_colors.get(true_a, ACCENT))

        # columns 2-3 : PET of class A + its histogram
        _draw_image_and_hist(axes, 2, s_a["pet"], "PET", name_a,
                             pet_cmap, hist_colors.get(true_a, ACCENT))

        # columns 4-5 : MRI of class B + its histogram
        _draw_image_and_hist(axes, 4, s_b["mri"], "MRI", name_b,
                             mri_cmap, hist_colors.get(true_b, GREEN))

        # columns 6-7 : PET of class B + its histogram
        _draw_image_and_hist(axes, 6, s_b["pet"], "PET", name_b,
                             pet_cmap, hist_colors.get(true_b, GREEN))

        # vertical divider between the two subjects
        fig.add_artist(plt.Line2D(
            [0.5, 0.5], [0.0, 1.0],
            transform=fig.transFigure,
            color=BORDER, linewidth=1.5, linestyle="--"
        ))

        plt.tight_layout()
        _savefig(fig, save_dir / f"{tag}.png")


def save_error_subjects(labels, preds, subjects, class_names, save_path):
    errors = {}
    for t, p, s in zip(labels.tolist(), preds.tolist(), subjects):
        if t != p:
            key = f"{class_names[t]}_as_{class_names[p]}"
            errors.setdefault(key, []).append(s)
    with open(save_path, "w") as f:
        json.dump(errors, f, indent=2)
    print(f"  Saved -> {save_path}\n")


def write_summary(labels, preds, scores, class_names,
                  experiment_name, save_path):
    acc       = accuracy_score(labels, preds)
    macro_f1  = f1_score(labels, preds, average="macro",    zero_division=0)
    macro_rec = recall_score(labels, preds, average="macro", zero_division=0)
    macro_pre = precision_score(labels, preds, average="macro", zero_division=0)

    per_f1  = f1_score(labels, preds, average=None, zero_division=0)
    per_pre = precision_score(labels, preds, average=None, zero_division=0)
    per_rec = recall_score(labels, preds, average=None, zero_division=0)
    support = np.bincount(labels, minlength=len(class_names))
    cm      = confusion_matrix(labels, preds)
    cm_off  = cm.copy()
    np.fill_diagonal(cm_off, 0)

    mean_conf_line = ""
    if scores is not None and scores.ndim == 2:
        mean_conf = scores[np.arange(len(labels)), preds].mean()
        mean_conf_line = f"  Mean confidence  : {mean_conf:.4f}\n"

    lines = [
        "=" * 62,
        f"  ANALYSIS REPORT — {experiment_name}",
        "=" * 62,
        "",
        "OVERALL METRICS",
        f"  Accuracy          : {acc:.4f}",
        f"  Macro F1          : {macro_f1:.4f}",
        f"  Macro Recall      : {macro_rec:.4f}",
        f"  Macro Precision   : {macro_pre:.4f}",
    ]
    if mean_conf_line:
        lines.append(mean_conf_line.rstrip())

    lines += [
        "",
        "PER-CLASS BREAKDOWN",
        f"  {'Class':<8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'n':>8}",
        "  " + "-" * 40,
    ]
    for i, c in enumerate(class_names):
        lines.append(
            f"  {c:<8} {per_pre[i]:>8.4f} {per_rec[i]:>8.4f}"
            f" {per_f1[i]:>8.4f} {support[i]:>8}"
        )

    lines += [
        "",
        "KEY WEAKNESSES",
        f"  Worst F1     : {class_names[int(np.argmin(per_f1))]}",
        f"  Worst recall : {class_names[int(np.argmin(per_rec))]}",
        "",
        "CONFUSION MATRIX  (rows=true, cols=pred)",
        "  " + "         " + "  ".join(f"{c:>6}" for c in class_names),
    ]
    for i, row in enumerate(cm):
        lines.append(
            "  " + f"{class_names[i]:<8}  " +
            "  ".join(f"{v:>6}" for v in row)
        )

    flat = [
        (cm_off[i, j], class_names[i], class_names[j])
        for i in range(len(class_names))
        for j in range(len(class_names)) if i != j
    ]
    lines += ["", "TOP CONFUSION PAIRS  (true -> predicted)"]
    for cnt, tc, pc in sorted(flat, reverse=True)[:8]:
        lines.append(f"  {tc} -> {pc} : {int(cnt)} errors")
    lines += ["", "=" * 62]

    text = "\n".join(lines)
    with open(save_path, "w") as f:
        f.write(text)
    print(text)
    print(f"\n  Saved -> {save_path}\n")


# ══════════════════════════════════════════════
# Argument parser
# ══════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Visual diagnostics for Alzheimer ViT model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Identity ──────────────────────────────────────────────────────────
    p.add_argument("--experiment_name", type=str, required=True)

    # ── Data ──────────────────────────────────────────────────────────────
    p.add_argument("--data_root",   type=str, required=True)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio",   type=float, default=0.1)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--num_workers", type=int,   default=4)

    # ── Model ─────────────────────────────────────────────────────────────
    p.add_argument("--model_type",   type=str, default="fusion",
                   choices=["fusion", "mri_only", "pet_only"],
                   help="Must match what was used at training time")
    p.add_argument("--fusion_type",  type=str, default="concat",
                   choices=["concat", "sum", "film", "gated", "CrossAttention"])
    p.add_argument("--num_classes",  type=int,  default=4)
    p.add_argument("--feature_dim",  type=int,  default=768)
    p.add_argument("--pretrained",   action="store_true")
    p.add_argument("--class_names",  type=str,  nargs="+",
                   default=["CN", "sMCI", "pMCI", "AD"])

    # ── Loss (must match what was used at training time) ──────────────────
    p.add_argument("--loss", type=str, default="crossentropy",
                   choices=["crossentropy", "mse", "focal"])

    # ── KFold (must match what was used at training time) ─────────────────
    p.add_argument("--kfold", type=int, default=0,
                   help="Number of folds used at training time. 0 = single split.")
    p.add_argument("--fold",  type=int, default=None,
                   help="Which fold checkpoint to load (1-based). "
                        "Required when --kfold > 0. "
                        "Use the 'Best fold' number from the training summary.")

    # ── Analysis ──────────────────────────────────────────────────────────
    p.add_argument("--split",        type=str, default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--top_n",        type=int, default=20)
    p.add_argument("--scan_samples", type=int, default=3)
    p.add_argument("--device",       type=str, default="cuda:0")
    p.add_argument("--seed",         type=int, default=12345)

    return p.parse_args()


# ══════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════

def main():
    args = parse_args()
    set_global_seed(args.seed)

    save_dir = Path("outputs") / "analysis" / args.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  Experiment : {args.experiment_name}")
    print(f"  Split      : {args.split}")
    print(f"  Loss mode  : {args.loss}")
    print(f"  Device     : {args.device}")
    print(f"  Output dir : {save_dir}")
    print(f"{'='*62}\n")

    # ── Dataset & split reconstruction ────────────────────────────────────
    # Must use the exact same split logic as train.py so test indices match.
    dataset = MRIPETDataset(root=args.data_root)

    if args.kfold > 0:
        # KFold mode: replicate the numpy permutation from run_kfold in train.py
        import numpy as _np
        from torch.utils.data import Subset
        from src.utils import seed_worker
        all_indices  = _np.arange(len(dataset))
        test_ratio   = 1.0 - args.train_ratio - args.val_ratio
        test_size    = int(test_ratio * len(dataset))
        rng          = _np.random.default_rng(args.seed)
        shuffled     = rng.permutation(all_indices)
        test_indices = shuffled[:test_size]
        chosen_ds    = Subset(dataset, test_indices)
        print(f"  KFold mode — test split: {len(test_indices)} samples "
              f"(same held-out set as training)\n")
    else:
        # Single split: replicate random_split from run_single in train.py
        generator = torch.Generator().manual_seed(args.seed)
        train_sz  = int(args.train_ratio * len(dataset))
        val_sz    = int(args.val_ratio   * len(dataset))
        test_sz   = len(dataset) - train_sz - val_sz
        train_ds, val_ds, test_ds = random_split(
            dataset, [train_sz, val_sz, test_sz], generator=generator
        )
        chosen_ds = {"train": train_ds, "val": val_ds, "test": test_ds}[args.split]
        print(f"  Single-split mode — {args.split}: {len(chosen_ds)} samples\n")

    loader = DataLoader(
        chosen_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    # ── Checkpoint path ───────────────────────────────────────────────────
    if args.kfold > 0:
        if args.fold is None:
            raise ValueError(
                "When --kfold > 0 you must also pass --fold N "
                "(the 'Best fold' number from the training summary)."
            )
        model_path = f"[{args.experiment_name}_fold{args.fold}].pth"
    else:
        model_path = f"[{args.experiment_name}].pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Weights not found: {model_path}\n"
            f"Train first:  python train.py --experiment_name {args.experiment_name} ..."
        )

    model = build_model(args)
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    print(f"  Loaded weights: {model_path}\n")

    # ── Inference with injected decode/score functions ────────────────────
    decode_fn, score_fn = _make_decode_and_score(args)

    labels, preds, scores, subjects, buckets = run_inference(
        model     = model,
        loader    = loader,
        device    = args.device,
        decode_fn = decode_fn,
        score_fn  = score_fn,
        num_classes = args.num_classes,
    )

    set_style()

    # ── Metric plots ──────────────────────────────────────────────────────
    print("\n" + "-" * 62)
    print("  METRICS & DIAGNOSTIC PLOTS")
    print("-" * 62 + "\n")

    plot_confusion_matrix(labels, preds, args.class_names,
                          save_dir / "confusion_matrix.png")
    plot_per_class_metrics(labels, preds, args.class_names,
                           save_dir / "per_class_metrics.png")
    plot_error_analysis(labels, preds, args.class_names,
                        save_dir / "error_analysis.png")
    plot_hardest_samples(labels, preds, scores, subjects, args.class_names,
                         save_dir / "hardest_samples.png", top_n=args.top_n)

    if args.loss in ("crossentropy", "focal"):
        plot_confidence_distributions(labels, scores, args.class_names,
                                      save_dir / "confidence_distribution.png")
        plot_calibration(labels, scores, args.class_names,
                         save_dir / "calibration.png")
    elif args.loss == "mse":
        plot_mse_residuals(labels, scores, args.class_names,
                           save_dir / "mse_residual_distribution.png")

    # ── Scan panels ───────────────────────────────────────────────────────
    print("\n" + "-" * 62)
    print("  SCAN COMPARISON PANELS")
    print("-" * 62 + "\n")
    plot_confused_pairs(buckets, args.class_names, save_dir,
                        max_samples=args.scan_samples)

    # ── Misclassification comparison (image + histogram side-by-side) ──────
    print("\n" + "-" * 62)
    print("  MISCLASSIFICATION COMPARISON  (image + intensity histogram)")
    print("-" * 62 + "\n")
    plot_misclassification_comparison(buckets, args.class_names, save_dir)

    # ── Subject log ───────────────────────────────────────────────────────
    save_error_subjects(labels, preds, subjects, args.class_names,
                        save_dir / "error_subjects.json")

    # ── Summary ───────────────────────────────────────────────────────────
    write_summary(labels, preds, scores, args.class_names,
                  args.experiment_name, save_dir / "summary_report.txt")

    print(f"\n{'='*62}")
    print(f"  All outputs -> {save_dir}/")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()