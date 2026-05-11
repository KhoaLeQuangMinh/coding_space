"""
analyze.py — Visual diagnostics for Alzheimer's ViT model
==========================================================

Usage (Kaggle / Jupyter notebook):
    %run analyze.py --config configs/my_experiment.yaml

Usage (terminal):
    python analyze.py --config configs/my_experiment.yaml

What this script does
---------------------
1.  Loads the trained model from  [experiment_name].pth
2.  Runs inference on the chosen split (default: test)
3.  Prints and saves every diagnostic plot inline (Kaggle/Jupyter cells)
4.  Saves all plots + a subject-ID error log to
      outputs/analysis/<experiment_name>/

Outputs
-------
  confusion_matrix.png              Raw + normalised confusion matrix
  per_class_metrics.png             Precision / Recall / F1 per class
  confidence_distribution.png       Softmax confidence histograms per class
  calibration.png                   Reliability / calibration diagram
  error_analysis.png                Off-diagonal mistake flow heatmap
  hardest_samples.png               Top-N most confidently wrong predictions
  confused_smci_as_cn_MRI/PET.png   sMCI predicted as CN  vs  sMCI correct
  confused_cn_as_smci_MRI/PET.png   CN predicted as sMCI  vs  CN correct
  confused_pmci_as_ad_MRI/PET.png   pMCI predicted as AD  vs  pMCI correct
  confused_ad_as_pmci_MRI/PET.png   AD predicted as pMCI  vs  AD correct
  error_subjects.json               Subject IDs for every error bucket
  summary_report.txt                Plain-text metric summary
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
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
)
from tqdm import tqdm

from src.data import MRIPETDataset
from src.baseline_model import BaselineModel
from src.utils import read_config


# ══════════════════════════════════════════════
# 0.  Palette & global style
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
CLASS_NAMES  = ["CN", "sMCI", "pMCI", "AD"]

# Label indices
CN_IDX   = 0
sMCI_IDX = 1
pMCI_IDX = 2
AD_IDX   = 3


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
    """Save figure and keep it visible in Kaggle / Jupyter inline."""
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    plt.show()        # renders inline in the notebook cell
    plt.close(fig)
    print(f"  Saved -> {path}\n")


# ══════════════════════════════════════════════
# 1.  Inference
# ══════════════════════════════════════════════
def run_inference(model, loader, device):
    """
    Run the model on a DataLoader and collect predictions, probabilities,
    ground-truth labels, subject IDs, and raw MRI/PET tensors for a
    representative subset of samples.

    Returns
    -------
    labels   : np.ndarray (N,)        integer class indices
    preds    : np.ndarray (N,)        predicted class indices
    probs    : np.ndarray (N, C)      softmax probabilities
    subjects : list                   subject IDs from the dataset
    buckets  : dict                   (true_cls, pred_cls) -> list of
                                      {"mri", "pet", "subject_id", "conf"}
                                      capped at MAX_STORE per bucket so RAM
                                      usage stays bounded even on large datasets
    """
    MAX_STORE = 6     # max volumes kept per (true, pred) bucket for scan panels

    model.eval()
    all_preds, all_labels, all_probs, all_subjects = [], [], [], []
    buckets = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            mri    = batch["mri"].to(device)
            pet    = batch["pet"].to(device)
            labels = batch["label"]
            sids   = batch["subject_id"]

            logits = model(mri, pet)
            probs  = F.softmax(logits, dim=1).cpu()
            preds  = probs.argmax(dim=1)

            for i in range(len(labels)):
                t   = int(labels[i])
                p   = int(preds[i])
                sid = sids[i].item() if hasattr(sids[i], "item") else sids[i]
                key = (t, p)

                all_preds.append(p)
                all_labels.append(t)
                all_probs.append(probs[i].numpy())
                all_subjects.append(sid)

                if key not in buckets:
                    buckets[key] = []
                if len(buckets[key]) < MAX_STORE:
                    buckets[key].append({
                        "mri":        mri[i].cpu(),
                        "pet":        pet[i].cpu(),
                        "subject_id": sid,
                        "conf":       float(probs[i, p]),
                    })

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        all_subjects,
        buckets,
    )


# ══════════════════════════════════════════════
# 2.  Confusion matrix
# ══════════════════════════════════════════════
def plot_confusion_matrix(labels, preds, save_path):
    """
    Plot the confusion matrix in two forms side by side.

    LEFT  — Raw counts
        Each cell shows the absolute number of samples where the true class
        is the row and the predicted class is the column.  The diagonal
        holds correct predictions.

    RIGHT — Row-normalised  (= per-class recall view)
        Each row sums to 1.0.  Cell [i, j] is the fraction of class-i
        samples predicted as class j.  The diagonal value equals the
        recall for that class.

    How to read it
    ~~~~~~~~~~~~~~
    * A bright off-diagonal cell means the model systematically confuses
      those two classes.  E.g. a bright cell at (sMCI, pMCI) means many
      sMCI patients are predicted as pMCI — clinically over-alarming.
    * If the diagonal is uniformly bright, the model is well-balanced.
    * If one row has a dim diagonal, that class has low recall — the model
      misses most real cases of it.  Check what class it bleeds into.
    * The green border on each diagonal cell highlights correct predictions.
    """
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

        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                v     = data[i, j]
                color = TEXT if v < data.max() * 0.6 else BACKGROUND
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        color=color, fontsize=12, fontweight="bold")

        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, fontsize=11)
        ax.set_yticklabels(CLASS_NAMES, fontsize=11)
        ax.set_xlabel("Predicted", labelpad=10)
        ax.set_ylabel("True",      labelpad=10)
        ax.set_title(title, color=TEXT_DIM, fontsize=11)

        for k in range(len(CLASS_NAMES)):
            ax.add_patch(plt.Rectangle(
                (k - 0.5, k - 0.5), 1, 1,
                fill=False, edgecolor=GREEN, linewidth=2
            ))

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color=TEXT_DIM)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_DIM)

    plt.tight_layout()
    _savefig(fig, save_path)


# ══════════════════════════════════════════════
# 3.  Per-class precision / recall / F1
# ══════════════════════════════════════════════
def plot_per_class_metrics(labels, preds, save_path):
    """
    Grouped bar chart showing Precision, Recall, and F1 for every class.

    Definitions (plain language)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Precision — "Of all subjects the model called class X, what fraction
                 actually have class X?"
                 High precision = few false alarms / false positives.

    Recall    — "Of all subjects that truly have class X, what fraction did
                 the model catch?"
                 High recall = few missed cases / false negatives.
                 For clinical AI, recall is usually the priority — missing
                 a real case is more dangerous than a false alarm.

    F1        — Harmonic mean of precision and recall.  Useful single number
                 for ranking class difficulty; penalises both extremes.

    How to read it
    ~~~~~~~~~~~~~~
    * Recall bar much shorter than Precision bar -> the model is conservative:
      it only predicts that class when very sure, but misses many real cases.
    * Precision bar much shorter than Recall bar -> the model over-predicts
      that class: it catches most real cases but over-diagnoses.
    * The dashed yellow line at 0.5 is a rough "better-than-chance" floor
      for a 4-class problem (random = 0.25).
    * The red arrow marks the weakest class by F1 — fix this first.
    * "n=..." below each class shows the support count; small n means
      noisier metrics — interpret with caution.
    """
    precision = precision_score(labels, preds, average=None, zero_division=0)
    recall    = recall_score(   labels, preds, average=None, zero_division=0)
    f1        = f1_score(       labels, preds, average=None, zero_division=0)
    support   = np.bincount(labels, minlength=len(CLASS_NAMES))

    x, w = np.arange(len(CLASS_NAMES)), 0.26

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
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9, color=TEXT)

    for i, s in enumerate(support):
        ax.text(i, -0.07, f"n={s}", ha="center", va="top", fontsize=9,
                color=TEXT_DIM, transform=ax.get_xaxis_transform())

    worst = int(np.argmin(f1))
    ax.annotate(
        f"  worst F1: {CLASS_NAMES[worst]}",
        xy=(worst + w, f1[worst]),
        xytext=(worst + w + 0.35, f1[worst] + 0.12),
        arrowprops=dict(arrowstyle="->", color=RED),
        color=RED, fontsize=10,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1", color=TEXT)
    ax.legend()
    ax.grid(axis="y")
    ax.axhline(0.5, color=YELLOW, linestyle=":", linewidth=1, alpha=0.6)

    plt.tight_layout()
    _savefig(fig, save_path)


# ══════════════════════════════════════════════
# 4.  Confidence distributions
# ══════════════════════════════════════════════
def plot_confidence_distributions(labels, probs, save_path):
    """
    For each class, histogram the model's softmax confidence on the
    true class, separated into correct vs. wrong predictions.

    What is "confidence" here?
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Confidence = softmax probability assigned to the true class.
    A value of 1.0 means the model is certain; 0.25 is random chance
    for a 4-class problem.

    How to read it
    ~~~~~~~~~~~~~~
    Green bars (correct predictions)
        Ideally cluster near 1.0.  If they spread widely across
        [0.25 – 1.0] the model is uncertain even when it is right —
        a sign of under-training or hard inherent class overlap.

    Red bars (wrong predictions)
        If red bars cluster at HIGH confidence (e.g. > 0.7) the model
        is confidently wrong — overconfident / miscalibrated.  This often
        happens when two classes share imaging patterns (sMCI vs pMCI).
        If red bars cluster near 0.25, the model is appropriately
        uncertain when wrong — less worrying.

    Dashed vertical lines = mean confidence for each group.

    Warning signs to look for
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    * Red peak near 0.8 – 1.0 -> dangerously overconfident errors;
      consider adding label smoothing or temperature scaling.
    * Green and red bars heavily overlapping -> the model cannot
      reliably distinguish this class; features are not discriminative.
    * Tiny green bars relative to large red bars -> low recall;
      the model is mostly missing this class.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor(BACKGROUND)
    fig.suptitle(
        "Confidence Distribution  (softmax score on the true class)",
        fontsize=14, color=TEXT
    )

    bins = np.linspace(0, 1, 25)

    for idx, ax in enumerate(axes.flat):
        ax.set_facecolor(SURFACE)
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

        if len(conf_c) > 0: ax.axvline(conf_c.mean(), color=GREEN,
                                        linestyle="--", linewidth=1.5)
        if len(conf_w) > 0: ax.axvline(conf_w.mean(), color=RED,
                                        linestyle="--", linewidth=1.5)

        ax.set_title(f"{CLASS_NAMES[idx]}  (total={mask.sum()})",
                     color=CLASS_COLORS[idx])
        ax.set_xlabel("Confidence on true class")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.grid(axis="y")

    plt.tight_layout()
    _savefig(fig, save_path)


# ══════════════════════════════════════════════
# 5.  Calibration / reliability diagram
# ══════════════════════════════════════════════
def plot_calibration(labels, probs, save_path, n_bins=10):
    """
    Reliability / calibration diagram — one curve per class (one-vs-rest).

    What it shows
    ~~~~~~~~~~~~~
    X-axis = the model's predicted probability for a class.
    Y-axis = the observed fraction of samples that truly belong to that
             class within that confidence bin.

    Perfect calibration -> all points fall on the diagonal.

    How to read it
    ~~~~~~~~~~~~~~
    Curve ABOVE the diagonal  -> underconfident: when the model says
        "60% chance of sMCI" the true rate is actually 80%.  The model
        is more accurate than it believes.

    Curve BELOW the diagonal  -> overconfident: when the model says
        "80% chance of AD" the true rate is only 50%.  Softmax scores
        should not be used as raw risk probabilities.

    Why it matters for clinical use
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    If a clinician uses the softmax score as a "probability of disease",
    a poorly calibrated model gives misleading risk estimates.
    Post-hoc calibration via temperature scaling is cheap and effective:
    divide logits by a learned scalar T before softmax.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(SURFACE)

    for cls_idx, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        binary = (labels == cls_idx).astype(int)
        cprobs = probs[:, cls_idx]
        edges  = np.linspace(0, 1, n_bins + 1)
        bconfs, baccs = [], []

        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (cprobs >= lo) & (cprobs < hi)
            if m.sum() > 0:
                bconfs.append(cprobs[m].mean())
                baccs.append(binary[m].mean())

        if bconfs:
            ax.plot(bconfs, baccs, "o-", color=color,
                    label=name, linewidth=2, markersize=6)

    ax.plot([0, 1], [0, 1], "--", color=TEXT_DIM, linewidth=1.5,
            label="Perfect calibration")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color=TEXT_DIM)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (true rate)")
    ax.set_title("Calibration / Reliability Diagram", color=TEXT)
    ax.legend()
    ax.grid()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    _savefig(fig, save_path)


# ══════════════════════════════════════════════
# 6.  Error flow heatmap
# ══════════════════════════════════════════════
def plot_error_analysis(labels, preds, save_path):
    """
    Off-diagonal confusion heatmap — shows only where mistakes go,
    not the correct predictions.

    The diagonal is replaced with a green checkmark so the eye is
    drawn entirely to the error structure.

    How to read it
    ~~~~~~~~~~~~~~
    Cell [i, j]  (i != j)  = number of true-class-i subjects wrongly
    predicted as class j.

    The annotation "-> N err" on the right is the total error count
    per row (how many subjects of that class were misclassified overall).

    Patterns to look for
    ~~~~~~~~~~~~~~~~~~~~
    * Bright cells between ADJACENT stages (CN<->sMCI, sMCI<->pMCI,
      pMCI<->AD) — expected because disease is a continuum, but still
      worth reducing.  These are the "boundary" cases.
    * Bright cells SKIPPING a stage (e.g. CN predicted as AD) — rare
      and alarming; may indicate data leakage, scanner confounds, or
      severe class imbalance.
    * A row where nearly all mass is off-diagonal means that class is
      almost entirely missed — consider class-weighted loss, focal loss,
      or oversampling for it.
    """
    cm     = confusion_matrix(labels, preds)
    errors = cm.copy().astype(float)
    np.fill_diagonal(errors, 0)

    cmap = LinearSegmentedColormap.from_list("err", [SURFACE, RED], N=256)

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(SURFACE)

    im = ax.imshow(errors, cmap=cmap, aspect="auto")

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i == j:
                ax.text(j, i, "v", ha="center", va="center",
                        color=GREEN, fontsize=16)
            else:
                v     = errors[i, j]
                color = TEXT if v < errors.max() * 0.5 else BACKGROUND
                ax.text(j, i, f"{int(v)}", ha="center", va="center",
                        color=color, fontsize=13, fontweight="bold")

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels([f"Pred: {c}" for c in CLASS_NAMES], fontsize=11)
    ax.set_yticklabels([f"True: {c}" for c in CLASS_NAMES], fontsize=11)
    ax.set_title("Error Flow  (off-diagonal mistakes only)", color=TEXT)

    for i, n in enumerate(errors.sum(axis=1)):
        ax.text(len(CLASS_NAMES) - 0.3, i, f"  -> {int(n)} err",
                va="center", color=RED, fontsize=9)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color=TEXT_DIM)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_DIM)

    plt.tight_layout()
    _savefig(fig, save_path)


# ══════════════════════════════════════════════
# 7.  Hardest samples
# ══════════════════════════════════════════════
def plot_hardest_samples(labels, preds, probs, subjects, save_path, top_n=20):
    """
    Horizontal bar chart of the model's most egregious wrong predictions,
    ranked by confidence on the TRUE class (ascending) — so rank #1 is
    the sample where the model was MOST certain it was not what it actually is.

    Each row shows two overlapping bars
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Green = softmax confidence the model assigns to the TRUE class.
    Red   = softmax confidence the model assigns to its (WRONG) predicted class.

    The right-side annotation shows:
        True:<class>  Pred:<class>  [conf_true=X | conf_pred=Y]

    How to use this
    ~~~~~~~~~~~~~~~
    * Subjects appearing repeatedly in the top-error list across
      different experiments or folds are likely label-noisy or
      genuinely prodromal edge cases — worth manual inspection.
    * Large red bar + tiny green bar -> the model is very confident
      in the wrong answer.  Check for scanner-site bias, age distribution
      skew, or class overlap in feature space.
    * If top errors cluster within one class, that class needs targeted
      help: more augmentation, focal loss, or additional data.
    """
    correct_conf = probs[np.arange(len(labels)), labels]
    wrong_mask   = preds != labels
    wrong_idx    = np.where(wrong_mask)[0]

    if len(wrong_idx) == 0:
        print("  No errors found — skipping hardest samples plot.")
        return

    sorted_idx  = wrong_idx[np.argsort(correct_conf[wrong_idx])]
    top_indices = sorted_idx[:top_n]

    fig, ax = plt.subplots(figsize=(12, max(5, len(top_indices) * 0.38)))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(SURFACE)

    for i, idx in enumerate(top_indices):
        t         = int(labels[idx])
        p         = int(preds[idx])
        conf_true = correct_conf[idx]
        conf_pred = probs[idx, p]

        ax.barh(i, conf_pred, color=RED,   alpha=0.7, height=0.6)
        ax.barh(i, conf_true, color=GREEN, alpha=0.7, height=0.6)

        note = (f"True:{CLASS_NAMES[t]}  Pred:{CLASS_NAMES[p]}  "
                f"[conf_true={conf_true:.2f} | conf_pred={conf_pred:.2f}]")
        ax.text(1.01, i, note, va="center", color=TEXT_DIM, fontsize=8,
                transform=ax.get_yaxis_transform())

    ax.set_yticks(np.arange(len(top_indices)))
    ax.set_yticklabels([f"#{i+1}" for i in range(len(top_indices))], fontsize=9)
    ax.set_xlabel("Confidence")
    ax.set_xlim(0, 1)
    ax.set_title(
        f"Top-{len(top_indices)} Hardest Errors  "
        f"(sorted by conf on true class, ascending)",
        color=TEXT
    )
    ax.legend(handles=[
        mpatches.Patch(color=GREEN, alpha=0.7, label="Conf for true class"),
        mpatches.Patch(color=RED,   alpha=0.7, label="Conf for predicted class"),
    ], loc="lower right")
    ax.grid(axis="x")

    plt.tight_layout()
    _savefig(fig, save_path)


# ══════════════════════════════════════════════
# 8.  Subject ID error log
# ══════════════════════════════════════════════
def save_error_subjects(labels, preds, subjects, save_path):
    """
    Save a JSON file listing subject IDs for every error bucket.

    Top-level keys (focused clinical pairs)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    "sMCI_predicted_as_CN"   — early MCI missed as healthy
    "CN_predicted_as_sMCI"   — healthy subjects over-diagnosed as MCI
    "pMCI_predicted_as_AD"   — progressive MCI over-called as dementia
    "AD_predicted_as_pMCI"   — dementia under-called as MCI

    "all_errors" sub-dict
    ~~~~~~~~~~~~~~~~~~~~~~
    Every possible (true, predicted) pair that has at least one error,
    keyed as "true_<class>_pred_<class>": [subject_ids].

    Why save subject IDs?
    ~~~~~~~~~~~~~~~~~~~~~
    * Spot repeat offenders across experiments — subjects always
      misclassified may be mislabelled, scanner outliers, or
      genuinely ambiguous prodromal cases.
    * Build a "hard set" for targeted data cleaning or curriculum
      learning (train on easy samples first, then introduce hard ones).
    * Track whether improvements actually fix the SAME subjects or
      just shuffle different ones — a model that fixes 10 errors but
      breaks 10 others is not improving.
    """
    subjects = np.array(subjects)

    focused = {
        "sMCI_predicted_as_CN":  [],
        "CN_predicted_as_sMCI":  [],
        "pMCI_predicted_as_AD":  [],
        "AD_predicted_as_pMCI":  [],
    }

    for i, (t, p) in enumerate(zip(labels, preds)):
        sid = str(subjects[i])
        if t == sMCI_IDX and p == CN_IDX:   focused["sMCI_predicted_as_CN"].append(sid)
        if t == CN_IDX   and p == sMCI_IDX: focused["CN_predicted_as_sMCI"].append(sid)
        if t == pMCI_IDX and p == AD_IDX:   focused["pMCI_predicted_as_AD"].append(sid)
        if t == AD_IDX   and p == pMCI_IDX: focused["AD_predicted_as_pMCI"].append(sid)

    all_errors = {}
    for ti, tn in enumerate(CLASS_NAMES):
        for pi, pn in enumerate(CLASS_NAMES):
            if ti == pi:
                continue
            mask = (labels == ti) & (preds == pi)
            if mask.sum() > 0:
                all_errors[f"true_{tn}_pred_{pn}"] = [
                    str(s) for s in subjects[mask]
                ]

    output = {**focused, "all_errors": all_errors}
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n  Error Subject ID Summary")
    print("  " + "-" * 42)
    for k, v in focused.items():
        print(f"  {k:<32}: {len(v)} subjects")
    print(f"\n  Total errors: {int((labels != preds).sum())} / {len(labels)}")
    print(f"  Saved -> {save_path}\n")


# ══════════════════════════════════════════════
# 9.  Volume visualisation helpers
# ══════════════════════════════════════════════
def _vol_to_numpy(tensor):
    """Convert a (1,H,W,D) or (H,W,D) CPU tensor to a numpy float array."""
    return tensor.squeeze().numpy()


def _plot_three_views(ax_row, volume, title, color, conf=None):
    """
    Render the axial, sagittal, and coronal mid-slices of a 3D brain
    volume into a pre-existing row of 3 matplotlib axes.

    The three standard neuroimaging views
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Axial    — horizontal cross-section; brain seen from above.
               Good for cortical/subcortical symmetry.
    Sagittal — left/right cross-section; profile view.
               Good for corpus callosum, cerebellum, brainstem.
    Coronal  — front/back cross-section; face-on view.
               Good for hippocampus, temporal lobe — the primary
               region affected in early Alzheimer's disease.

    Using all three matters because atrophy or hypometabolism (on PET)
    can be subtle and plane-dependent.  A lesion invisible in the axial
    plane may be clearly visible in the coronal plane.
    """
    v   = _vol_to_numpy(volume)
    mid = [s // 2 for s in v.shape]

    for ax, (img, vtitle) in zip(ax_row, [
        (np.rot90(v[:, :, mid[2]]),  f"Axial z={mid[2]}"),
        (np.rot90(v[mid[0], :, :]),  f"Sagittal x={mid[0]}"),
        (np.rot90(v[:, mid[1], :]),  f"Coronal y={mid[1]}"),
    ]):
        ax.imshow(img, cmap="hot", interpolation="bilinear")
        ax.set_title(vtitle, color=color, fontsize=8)
        ax.axis("off")

    label = f"{title}  (conf={conf:.2f})" if conf is not None else title
    ax_row[1].set_title(label, color=color, fontsize=9)


def _build_comparison_panel(
    bucket_wrong,  label_wrong,   color_wrong,
    bucket_right,  label_right,   color_right,
    modality,
    suptitle,
    save_path,
    max_samples=3,
):
    """
    Side-by-side scan comparison panel for two confusion buckets.

    Layout
    ~~~~~~
    LEFT  half  (3 cols) = wrong bucket  e.g. "sMCI -> CN  (error)"
    RIGHT half  (3 cols) = correct bucket  e.g. "sMCI -> sMCI  (correct)"

    Each row = one subject showing Axial | Sagittal | Coronal views.
    The subject ID is printed on the left margin of each row.

    Why this is useful
    ~~~~~~~~~~~~~~~~~~
    Comparing the scans the model gets wrong versus right for the same
    true class answers a key question:

    "Does the model fail on genuinely ambiguous scans or on
     scans that look obviously different from the rest of the class?"

    If wrong scans look visually milder (less atrophy, less
    hypometabolism on PET), the confusion is clinically explainable —
    early-stage cases sit near the boundary.  This suggests the model
    needs more boundary-stage training data or a softer loss.

    If wrong scans look identical to correct ones, there is no visual
    signal driving the error — likely a data imbalance, label noise,
    or the model over-fitting to a confound (scanner site, age, etc.).

    MRI vs PET
    ~~~~~~~~~~
    This function is called twice per pair — once for MRI, once for PET.
    PET (FDG metabolic activity) typically shows decline earlier than
    structural MRI.  If PET panels look more discriminative than MRI
    panels, the model may be under-utilising the PET stream.
    """
    n = min(max_samples, len(bucket_wrong), len(bucket_right))
    if n == 0:
        print(f"  Not enough samples for panel: {suptitle} [{modality}]")
        return

    ncols, nrows = 6, n
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 2.6, nrows * 2.6 + 1.4))
    fig.patch.set_facecolor(BACKGROUND)
    fig.suptitle(f"{suptitle}  [{modality}]", fontsize=13,
                 color=TEXT, y=1.01)

    if nrows == 1:
        axes = axes[np.newaxis, :]

    # Group headers
    axes[0, 1].set_title(label_wrong, color=color_wrong,
                          fontsize=11, fontweight="bold", pad=16)
    axes[0, 4].set_title(label_right, color=color_right,
                          fontsize=11, fontweight="bold", pad=16)

    key = "mri" if modality == "MRI" else "pet"

    for row in range(n):
        sw = bucket_wrong[row]
        sr = bucket_right[row]

        _plot_three_views(axes[row, 0:3], sw[key], label_wrong,
                          color_wrong, conf=sw["conf"])
        _plot_three_views(axes[row, 3:6], sr[key], label_right,
                          color_right, conf=sr["conf"])

        axes[row, 0].set_ylabel(
            f"ID:{sw['subject_id']}", color=TEXT_DIM,
            fontsize=7, rotation=0, labelpad=52, va="center"
        )

    plt.tight_layout()
    _savefig(fig, save_path)


# ══════════════════════════════════════════════
# 10.  Confused scan panels
# ══════════════════════════════════════════════
def plot_confused_pairs(buckets, save_dir, max_samples=3):
    """
    Generate MRI and PET scan comparison panels for the two most
    clinically important confusion groups.

    Group A — sMCI vs CN  (early MCI / healthy boundary)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Panel 1: sMCI predicted as CN  vs  sMCI predicted as sMCI
        The model is treating some MCI patients as healthy.
        Do those scans look structurally / metabolically normal?
        If yes -> the cases are genuinely early-stage (minimal atrophy).
        If no  -> possible label noise or scanner confound.

    Panel 2: CN predicted as sMCI  vs  CN predicted as CN
        The model is over-diagnosing some healthy subjects as MCI.
        Are those CN scans atypical (older, more atrophy)?  Or random?

    Group B — pMCI vs AD  (late MCI / dementia boundary)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Panel 3: pMCI predicted as AD  vs  pMCI predicted as pMCI
        The model over-calls progressive MCI as full dementia.
        Does this correlate with more severe hypometabolism on PET?

    Panel 4: AD predicted as pMCI  vs  AD predicted as AD
        The model under-calls dementia.
        Are these atypical or mild AD presentations?

    Each panel is produced for both MRI and PET so you can compare
    which modality better separates the confused groups.

    Files saved per panel
    ~~~~~~~~~~~~~~~~~~~~~
    confused_smci_as_cn_MRI.png   confused_smci_as_cn_PET.png
    confused_cn_as_smci_MRI.png   confused_cn_as_smci_PET.png
    confused_pmci_as_ad_MRI.png   confused_pmci_as_ad_PET.png
    confused_ad_as_pmci_MRI.png   confused_ad_as_pmci_PET.png
    """
    pairs = [
        # wrong_key          correct_key         wrong_label            correct_label           w_col  c_col  stem
        ((sMCI_IDX, CN_IDX),   (sMCI_IDX, sMCI_IDX), "sMCI->CN (wrong)",    "sMCI->sMCI (correct)", RED,   GREEN, "confused_smci_as_cn"),
        ((CN_IDX,   sMCI_IDX), (CN_IDX,   CN_IDX),   "CN->sMCI (wrong)",    "CN->CN (correct)",     RED,   GREEN, "confused_cn_as_smci"),
        ((pMCI_IDX, AD_IDX),   (pMCI_IDX, pMCI_IDX), "pMCI->AD (wrong)",    "pMCI->pMCI (correct)", RED,   GREEN, "confused_pmci_as_ad"),
        ((AD_IDX,   pMCI_IDX), (AD_IDX,   AD_IDX),   "AD->pMCI (wrong)",    "AD->AD (correct)",     RED,   GREEN, "confused_ad_as_pmci"),
    ]

    for wk, ck, wl, cl, wc, cc, stem in pairs:
        bw = buckets.get(wk, [])
        bc = buckets.get(ck, [])

        wname = f"{CLASS_NAMES[wk[0]]}->{CLASS_NAMES[wk[1]]}"
        if len(bw) == 0:
            print(f"  No samples for {wname} — skipping.")
            continue

        suptitle = f"{wl}   |   {cl}"
        for modality in ["MRI", "PET"]:
            _build_comparison_panel(
                bw, wl, wc,
                bc, cl, cc,
                modality,
                suptitle,
                save_dir / f"{stem}_{modality}.png",
                max_samples=max_samples,
            )


# ══════════════════════════════════════════════
# 11.  Text summary
# ══════════════════════════════════════════════
def write_summary(labels, preds, probs, experiment_name, save_path):
    acc       = accuracy_score(labels, preds)
    macro_f1  = f1_score(labels, preds, average="macro",    zero_division=0)
    macro_rec = recall_score(labels, preds, average="macro", zero_division=0)
    macro_pre = precision_score(labels, preds, average="macro", zero_division=0)

    per_f1  = f1_score(labels, preds, average=None, zero_division=0)
    per_pre = precision_score(labels, preds, average=None, zero_division=0)
    per_rec = recall_score(labels, preds, average=None, zero_division=0)
    support = np.bincount(labels, minlength=len(CLASS_NAMES))

    cm     = confusion_matrix(labels, preds)
    cm_off = cm.copy(); np.fill_diagonal(cm_off, 0)
    mean_conf = probs[np.arange(len(labels)), preds].mean()

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
        f"  Mean confidence   : {mean_conf:.4f}",
        "",
        "PER-CLASS BREAKDOWN",
        f"  {'Class':<8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'n':>8}",
        "  " + "-" * 40,
    ]
    for i, c in enumerate(CLASS_NAMES):
        lines.append(
            f"  {c:<8} {per_pre[i]:>8.4f} {per_rec[i]:>8.4f}"
            f" {per_f1[i]:>8.4f} {support[i]:>8}"
        )

    lines += [
        "",
        "KEY WEAKNESSES",
        f"  Worst F1     : {CLASS_NAMES[int(np.argmin(per_f1))]}",
        f"  Worst recall : {CLASS_NAMES[int(np.argmin(per_rec))]}",
        "",
        "CONFUSION MATRIX  (rows=true, cols=pred)",
        "  " + "         " + "  ".join(f"{c:>6}" for c in CLASS_NAMES),
    ]
    for i, row in enumerate(cm):
        lines.append("  " + f"{CLASS_NAMES[i]:<8}  " +
                     "  ".join(f"{v:>6}" for v in row))

    flat = [
        (cm_off[i, j], CLASS_NAMES[i], CLASS_NAMES[j])
        for i in range(len(CLASS_NAMES))
        for j in range(len(CLASS_NAMES)) if i != j
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
# 12.  Entry point
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Visual error diagnostics for Alzheimer ViT model"
    )
    parser.add_argument("--config",        type=str, required=True,
                        help="Path to experiment yaml")
    parser.add_argument("--system_config", type=str,
                        default="configs/kaggle.yaml",
                        help="System/data yaml (default: configs/kaggle.yaml)")
    parser.add_argument("--split",         type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--top_n",         type=int, default=20,
                        help="Hard-error samples to show (default: 20)")
    parser.add_argument("--scan_samples",  type=int, default=3,
                        help="Scan pairs per confusion panel (default: 3)")
    args = parser.parse_args()

    config     = read_config(args.system_config)
    config_exp = read_config(args.config)
    device     = config_exp["training"]["device"]
    exp_name   = config_exp["experiment"]["name"]

    save_dir = Path("outputs") / "analysis" / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  Experiment : {exp_name}")
    print(f"  Split      : {args.split}")
    print(f"  Device     : {device}")
    print(f"  Output dir : {save_dir}")
    print(f"{'='*62}\n")

    # Dataset
    dataset   = MRIPETDataset(root=config["data"]["root"])
    generator = torch.Generator().manual_seed(12345)

    train_sz = int(config["split"]["train_ratio"] * len(dataset))
    val_sz   = int(config["split"]["val_ratio"]   * len(dataset))
    test_sz  = len(dataset) - train_sz - val_sz

    train_ds, val_ds, test_ds = random_split(
        dataset, [train_sz, val_sz, test_sz], generator=generator
    )
    chosen_ds = {"train": train_ds, "val": val_ds, "test": test_ds}[args.split]

    loader = DataLoader(
        chosen_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )
    print(f"  Dataset ({args.split}): {len(chosen_ds)} samples\n")

    # Model
    model_path = f"[{exp_name}].pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Weights not found: {model_path}\n"
            f"Train first: python train.py --config {args.config}"
        )

    model = BaselineModel(
        class_num=1,
        fusion_method=config_exp["model"]["fusion_type"],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"  Loaded weights: {model_path}\n")

    # Inference
    labels, preds, probs, subjects, buckets = run_inference(
        model, loader, device
    )

    set_style()

    # ── Metric plots ─────────────────────────────────
    print("\n" + "-" * 62)
    print("  METRICS & DIAGNOSTIC PLOTS")
    print("-" * 62 + "\n")

    plot_confusion_matrix(
        labels, preds, save_dir / "confusion_matrix.png")
    plot_per_class_metrics(
        labels, preds, save_dir / "per_class_metrics.png")
    plot_confidence_distributions(
        labels, probs, save_dir / "confidence_distribution.png")
    plot_calibration(
        labels, probs, save_dir / "calibration.png")
    plot_error_analysis(
        labels, preds, save_dir / "error_analysis.png")
    plot_hardest_samples(
        labels, preds, probs, subjects,
        save_dir / "hardest_samples.png", top_n=args.top_n)

    # ── Scan panels ──────────────────────────────────
    print("\n" + "-" * 62)
    print("  SCAN COMPARISON PANELS")
    print("-" * 62 + "\n")

    plot_confused_pairs(buckets, save_dir, max_samples=args.scan_samples)

    # ── Subject ID log ───────────────────────────────
    print("\n" + "-" * 62)
    print("  SUBJECT ID ERROR LOG")
    print("-" * 62 + "\n")

    save_error_subjects(
        labels, preds, subjects,
        save_dir / "error_subjects.json")

    # ── Summary ──────────────────────────────────────
    print("\n" + "-" * 62)
    print("  SUMMARY REPORT")
    print("-" * 62 + "\n")

    write_summary(
        labels, preds, probs, exp_name,
        save_dir / "summary_report.txt")

    print(f"\n{'='*62}")
    print(f"  All outputs -> {save_dir}/")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()