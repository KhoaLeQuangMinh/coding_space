"""
plot_roi_histograms
===================
Drop-in addition to analyse.py.

Replaces the single whole-volume histogram in plot_misclassification_comparison
with per-ROI histograms grouped by anatomical lobe, using the AAL3 atlas.

New public function
-------------------
plot_roi_histograms(buckets, class_names, save_dir,
                    atlas_path, modality="pet",
                    max_samples=5, roi_group=None)

    atlas_path  : path to AAL__1__nii.gz (or any AAL3-compatible atlas)
    modality    : "pet" or "mri"
    roi_group   : None → plot all lobe groups (one figure per group)
                  or a string key from ROI_GROUPS, e.g. "temporal"

How atlas resampling works
--------------------------
The atlas is 91×109×91 at 2 mm MNI.
Your volumes (from the buckets) may be a different shape.
We resample the atlas to match the volume shape with nearest-neighbour
interpolation using scipy.ndimage.zoom — no external NIfTI library needed.

Output files (per confusion pair × per lobe group)
---------------------------------------------------
  roi_<modality>_<pair_tag>_<lobe>.png

Each figure shows:
  rows  = up to max_samples subject pairs (class A left | class B right)
  cols  = ROIs in the lobe group (one histogram per ROI)

Each cell: overlapping histograms for class A (left colour) and class B (right colour).
"""

import gzip
import struct
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import zoom

# ── colour palette (same as analyse.py) ─────────────────────────────────────
BACKGROUND = "#0d1117"
SURFACE    = "#161b22"
BORDER     = "#30363d"
ACCENT     = "#58a6ff"
GREEN      = "#3fb950"
RED        = "#f85149"
YELLOW     = "#d29922"
PURPLE     = "#bc8cff"
TEXT       = "#e6edf3"
TEXT_DIM   = "#8b949e"


# ── AAL3 label → short name map ──────────────────────────────────────────────
# Labels ending in 1 = left hemisphere, 2 = right hemisphere.
# Only Alzheimer's-relevant ROIs are included; extend freely.
AAL3_LABELS = {
    # ── Frontal ──────────────────────────────────────────────────────────
    2001: "PreCG.L",    2002: "PreCG.R",
    2101: "SFG.L",      2102: "SFG.R",
    2111: "SFGmed.L",   2112: "SFGmed.R",
    2201: "MFG.L",      2202: "MFG.R",
    2211: "MFGorb.L",   2212: "MFGorb.R",
    2301: "IFGtri.L",   2302: "IFGtri.R",
    2311: "IFGoper.L",  2312: "IFGoper.R",
    2321: "IFGorb.L",   2322: "IFGorb.R",
    2331: "Rolandic.L", 2332: "Rolandic.R",
    2401: "SMA.L",      2402: "SMA.R",
    2501: "OFCmed.L",   2502: "OFCmed.R",
    2601: "OFClat.L",   2602: "OFClat.R",
    2611: "OFCpost.L",  2612: "OFCpost.R",
    2701: "Rectal.L",   2702: "Rectal.R",
    3001: "Insula.L",   3002: "Insula.R",
    # ── Cingulate ────────────────────────────────────────────────────────
    4001: "ACG.L",      4002: "ACG.R",
    4011: "MCG.L",      4012: "MCG.R",
    4021: "PCG.L",      4022: "PCG.R",
    4101: "HIP.L",      4102: "HIP.R",
    4111: "PHG.L",      4112: "PHG.R",
    4201: "Amyg.L",     4202: "Amyg.R",
    # ── Parietal ─────────────────────────────────────────────────────────
    5001: "PostCG.L",   5002: "PostCG.R",
    5011: "SPG.L",      5012: "SPG.R",
    5021: "IPG.L",      5022: "IPG.R",
    5101: "SMG.L",      5102: "SMG.R",
    5201: "AG.L",       5202: "AG.R",
    5301: "Precun.L",   5302: "Precun.R",
    5401: "PCL.L",      5402: "PCL.R",
    # ── Temporal ─────────────────────────────────────────────────────────
    6001: "Heschls.L",  6002: "Heschls.R",
    6101: "STG.L",      6102: "STG.R",
    6201: "TPsup.L",    6202: "TPsup.R",
    6211: "AAL.L",      6212: "AAL.R",
    6221: "MTG.L",      6222: "MTG.R",
    6301: "TPmid.L",    6302: "TPmid.R",
    6401: "ITG.L",      6402: "ITG.R",
    # ── Occipital ────────────────────────────────────────────────────────
    7001: "CUN.L",      7002: "CUN.R",
    7011: "SOG.L",      7012: "SOG.R",
    7021: "MOG.L",      7022: "MOG.R",
    7101: "IOG.L",      7102: "IOG.R",
    # ── Subcortical ──────────────────────────────────────────────────────
    8101: "Caudate.L",  8102: "Caudate.R",
    8111: "Putamen.L",  8112: "Putamen.R",
    8121: "Pallidum.L", 8122: "Pallidum.R",
    8201: "Thal.L",     8202: "Thal.R",
    8211: "Heschl2.L",  8212: "Heschl2.R",
    8301: "NAcc.L",     8302: "NAcc.R",
    # ── Cerebellum ───────────────────────────────────────────────────────
    9001: "Cereb1.L",   9002: "Cereb1.R",
    9011: "Cereb2.L",   9012: "Cereb2.R",
    9021: "Cereb3.L",   9022: "Cereb3.R",
    9031: "Cereb4.L",   9032: "Cereb4.R",
    9041: "Cereb4_5.L", 9042: "Cereb4_5.R",
    9051: "Cereb5.L",   9052: "Cereb5.R",
    9061: "Cereb6.L",   9062: "Cereb6.R",
    9071: "Cereb7b.L",  9072: "Cereb7b.R",
    9081: "Cereb8.L",   9082: "Cereb8.R",
    9100: "Vermis1_2",  9110: "Vermis3",
    9120: "Vermis4_5",  9130: "Vermis6",
    9140: "Vermis7",    9150: "Vermis8",
    9160: "Vermis9",    9170: "Vermis10",
}

# ── Lobe groups — keys users can pass as roi_group ───────────────────────────
# Alzheimer's-most-relevant regions are marked with * in comments.
ROI_GROUPS = {
    "frontal": [
        2001, 2002,  # PreCG         * motor cortex
        2101, 2102,  # SFG           * executive function
        2111, 2112,  # SFGmed        * default mode
        2201, 2202,  # MFG           * working memory
        2211, 2212,  # MFGorb
        2301, 2302,  # IFGtri        * language
        2311, 2312,  # IFGoper
        2321, 2322,  # IFGorb
        2501, 2502,  # OFCmed        * reward/emotion
        2601, 2602,  # OFClat
        2611, 2612,  # OFCpost
        2401, 2402,  # SMA
        2701, 2702,  # Rectal
        3001, 3002,  # Insula        * salience
    ],
    "cingulate_medial_temporal": [
        4001, 4002,  # ACG           * attention
        4011, 4012,  # MCG
        4021, 4022,  # PCG           * default mode *
        4101, 4102,  # HIP           * *** hippocampus — key AD marker ***
        4111, 4112,  # PHG           * *** parahippocampal gyrus ***
        4201, 4202,  # Amyg          * *** amygdala ***
    ],
    "parietal": [
        5001, 5002,  # PostCG
        5011, 5012,  # SPG
        5021, 5022,  # IPG           * parietal association *
        5101, 5102,  # SMG           * supramarginal
        5201, 5202,  # AG            * angular gyrus / default mode *
        5301, 5302,  # Precun        * *** precuneus — early AD hypometabolism ***
        5401, 5402,  # PCL
    ],
    "temporal": [
        6001, 6002,  # Heschls       * primary auditory
        6101, 6102,  # STG           * superior temporal *
        6201, 6202,  # TPsup
        6221, 6222,  # MTG           * *** middle temporal — AD atrophy ***
        6301, 6302,  # TPmid
        6401, 6402,  # ITG           * *** inferior temporal ***
    ],
    "occipital": [
        7001, 7002,  # CUN           * cuneus
        7011, 7012,  # SOG
        7021, 7022,  # MOG
        7101, 7102,  # IOG
    ],
    "subcortical": [
        8101, 8102,  # Caudate       *
        8111, 8112,  # Putamen       *
        8121, 8122,  # Pallidum
        8201, 8202,  # Thal          * *** thalamus ***
        8301, 8302,  # NAcc          * nucleus accumbens
    ],
    "cerebellum": [
        9001, 9002, 9011, 9012, 9021, 9022, 9031, 9032,
        9041, 9042, 9051, 9052, 9061, 9062, 9071, 9072,
        9081, 9082,
        9100, 9110, 9120, 9130, 9140, 9150, 9160, 9170,
    ],
}

# Friendly titles for figures
GROUP_TITLES = {
    "frontal":                   "Frontal Lobe",
    "cingulate_medial_temporal": "Cingulate + Medial Temporal (Hippocampus / Amygdala / PHG)",
    "parietal":                  "Parietal Lobe  (incl. Precuneus)",
    "temporal":                  "Temporal Lobe  (incl. MTG / ITG)",
    "occipital":                 "Occipital Lobe",
    "subcortical":               "Subcortical  (Thalamus / Basal Ganglia)",
    "cerebellum":                "Cerebellum & Vermis",
}


# ── Atlas loader ─────────────────────────────────────────────────────────────

def _load_atlas(atlas_path):
    """
    Load AAL3 atlas from a .nii.gz or plain .nii file without nibabel.
    Returns (atlas_data: np.ndarray int32, shape=(91,109,91))
    Auto-detects gzip vs raw by peeking at the first two bytes.
    """
    atlas_path = Path(atlas_path)
    with open(atlas_path, "rb") as f:
        magic = f.read(2)

    if magic == b"\x1f\x8b":          # gzip magic bytes
        with gzip.open(atlas_path, "rb") as f:
            raw = f.read()
    else:                              # plain uncompressed .nii
        with open(atlas_path, "rb") as f:
            raw = f.read()

    # NIfTI-1, big-endian (verified from header of this specific atlas)
    nx, ny, nz = 91, 109, 91
    offset = 352
    atlas = (
        np.frombuffer(raw[offset : offset + nx * ny * nz * 2],
                      dtype=np.dtype(">i2"))
        .reshape((nx, ny, nz))
        .astype(np.int32)
    )
    return atlas


def _resample_atlas(atlas, target_shape):
    """
    Nearest-neighbour resample atlas from (91,109,91) to target_shape.
    Uses scipy.ndimage.zoom — no NIfTI header needed.
    """
    if atlas.shape == target_shape:
        return atlas
    zoom_factors = tuple(t / s for t, s in zip(target_shape, atlas.shape))
    return zoom(atlas, zoom_factors, order=0, prefilter=False).astype(np.int32)


# ── Single-ROI histogram helper ──────────────────────────────────────────────

def _roi_hist(ax, vol, mask, color, label, bins=40):
    """
    Draw a histogram of vol[mask] into ax.
    Skips gracefully if the ROI has no voxels after masking.
    Returns mean value or None.
    """
    vox = vol[mask]
    if vox.size == 0:
        ax.text(0.5, 0.5, "no voxels", ha="center", va="center",
                transform=ax.transAxes, color=TEXT_DIM, fontsize=6)
        return None

    # Remove near-zero background
    # threshold = vox.max() * 0.05
    # vox = vox[vox > threshold]
    if vox.size < 5:
        ax.text(0.5, 0.5, "sparse", ha="center", va="center",
                transform=ax.transAxes, color=TEXT_DIM, fontsize=6)
        return None

    lo, hi = np.percentile(vox, 1), np.percentile(vox, 99)
    vox = vox[(vox >= lo) & (vox <= hi)]
    mu = vox.mean()

    ax.hist(vox, bins=bins, color=color, alpha=0.65,
            edgecolor=BACKGROUND, linewidth=0.2)
    ax.axvline(mu, color=color, linestyle="--", linewidth=1.2,
               alpha=0.9, label=f"μ={mu:.3f}")
    return mu


# ── Main public function ─────────────────────────────────────────────────────

def plot_roi_histograms(
    buckets,
    class_names,
    save_dir,
    atlas_path,
    modality="pet",        # "pet" or "mri"
    max_samples=5,
    roi_group=None,        # None → all groups; or one key from ROI_GROUPS
):
    """
    For each confusion pair, plot per-ROI intensity histograms (one per brain region)
    instead of a single whole-volume histogram.

    Parameters
    ----------
    buckets      : dict from run_inference  { (true_cls, pred_cls): [sample, ...] }
    class_names  : list of str, e.g. ["CN","sMCI","pMCI","AD"]
    save_dir     : Path — output directory
    atlas_path   : str/Path to AAL__1__nii.gz
    modality     : "pet" or "mri"
    max_samples  : max subject pairs per figure row
    roi_group    : None or one of ROI_GROUPS keys
    """
    save_dir = Path(save_dir)
    atlas_path = Path(atlas_path)

    atlas_raw = _load_atlas(atlas_path)

    # Colour assignments per class index
    hist_colors = {
        0: ACCENT,   # CN
        1: GREEN,    # sMCI
        2: YELLOW,   # pMCI
        3: RED,      # AD
    }

    def _idx(name):
        try:
            return class_names.index(name)
        except ValueError:
            return None

    CN, sMCI, pMCI, AD = _idx("CN"), _idx("sMCI"), _idx("pMCI"), _idx("AD")

    # Confusion pairs to iterate
    requested = []
    if None not in (CN, sMCI):
        requested.append((CN,   sMCI, CN,   "roi_cn_smci_as_cn"))
        requested.append((CN,   sMCI, sMCI, "roi_cn_smci_as_smci"))
    if None not in (pMCI, AD):
        requested.append((pMCI, AD,   AD,   "roi_pmci_ad_as_ad"))
        requested.append((pMCI, AD,   pMCI, "roi_pmci_ad_as_pmci"))
    if None not in (sMCI, pMCI):
        requested.append((sMCI, pMCI, sMCI, "roi_smci_pmci_as_smci"))
        requested.append((sMCI, pMCI, pMCI, "roi_smci_pmci_as_pmci"))

    # Decide which groups to plot
    groups_to_plot = (
        {roi_group: ROI_GROUPS[roi_group]}
        if roi_group is not None
        else ROI_GROUPS
    )

    for true_a, true_b, pred_cls, base_tag in requested:

        samples_a = buckets.get((true_a, pred_cls), [])[:max_samples]
        samples_b = buckets.get((true_b, pred_cls), [])[:max_samples]

        if not samples_a or not samples_b:
            name_a = class_names[true_a]
            name_b = class_names[true_b]
            pred_n = class_names[pred_cls]
            print(f"  [skip roi] {base_tag}: missing samples for "
                  f"{name_a}→{pred_n} or {name_b}→{pred_n}")
            continue

        n_rows = min(len(samples_a), len(samples_b))  # only paired rows
        name_a = class_names[true_a]
        name_b = class_names[true_b]
        pred_n = class_names[pred_cls]
        col_a  = hist_colors.get(true_a, ACCENT)
        col_b  = hist_colors.get(true_b, GREEN)

        # Infer volume shape from first sample
        vol0 = samples_a[0][modality].squeeze(0).numpy()
        atlas = _resample_atlas(atlas_raw, vol0.shape)

        for group_key, label_list in groups_to_plot.items():

            # Filter to labels that actually exist in this atlas version
            valid_labels = [l for l in label_list if l in AAL3_LABELS
                            and np.any(atlas == l)]
            if not valid_labels:
                continue

            n_rois = len(valid_labels)
            # Layout: n_rows rows × n_rois cols, two overlapping hists per cell
            fig, axes = plt.subplots(
                n_rows, n_rois,
                figsize=(max(n_rois * 1.8, 10), n_rows * 2.8),
                squeeze=False,
            )
            fig.patch.set_facecolor(BACKGROUND)
            fig.suptitle(
                f"{GROUP_TITLES[group_key]}  —  {modality.upper()} intensities\n"
                f"Both predicted as {pred_n}  "
                f"(━ {name_a}  ┅ {name_b})",
                fontsize=10, color=TEXT, y=1.01,
            )

            for row in range(n_rows):
                vol_a = samples_a[row][modality].squeeze(0).numpy()
                vol_b = samples_b[row][modality].squeeze(0).numpy()

                # Resample atlas to this subject's volume shape if needed
                if atlas.shape != vol_a.shape:
                    atlas_s = _resample_atlas(atlas_raw, vol_a.shape)
                else:
                    atlas_s = atlas

                for col, lbl in enumerate(valid_labels):
                    ax = axes[row, col]
                    ax.set_facecolor(SURFACE)
                    for spine in ax.spines.values():
                        spine.set_edgecolor(BORDER)
                    ax.tick_params(colors=TEXT_DIM, labelsize=5)

                    roi_mask = (atlas_s == lbl)

                    mu_a = _roi_hist(ax, vol_a, roi_mask, col_a, name_a)
                    mu_b = _roi_hist(ax, vol_b, roi_mask, col_b, name_b)

                    roi_name = AAL3_LABELS[lbl]
                    title_lines = [roi_name]
                    if mu_a is not None:
                        title_lines.append(f"{name_a[:3]} μ={mu_a:.2f}")
                    if mu_b is not None:
                        title_lines.append(f"{name_b[:3]} μ={mu_b:.2f}")

                    ax.set_title("\n".join(title_lines),
                                 color=TEXT, fontsize=5.5, pad=2)
                    ax.set_xlabel("Intensity", color=TEXT_DIM, fontsize=5)
                    ax.set_ylabel("Count" if col == 0 else "",
                                  color=TEXT_DIM, fontsize=5)
                    ax.grid(axis="y", alpha=0.25, color=BORDER)

                # Row label on the left
                axes[row, 0].set_ylabel(
                    f"Pair {row+1}\n{name_a}:{samples_a[row]['subject_id']}\n"
                    f"{name_b}:{samples_b[row]['subject_id']}",
                    color=TEXT_DIM, fontsize=5, labelpad=4,
                )

            # Legend
            legend_patches = [
                mpatches.Patch(color=col_a, alpha=0.75, label=f"True {name_a} → pred {pred_n}"),
                mpatches.Patch(color=col_b, alpha=0.75, label=f"True {name_b} → pred {pred_n}"),
            ]
            fig.legend(handles=legend_patches, loc="lower center",
                       ncol=2, fontsize=8,
                       facecolor=SURFACE, edgecolor=BORDER,
                       labelcolor=TEXT, bbox_to_anchor=(0.5, -0.01))

            plt.tight_layout(rect=[0, 0.03, 1, 1])
            out_path = save_dir / f"{base_tag}_{modality}_{group_key}.png"
            fig.savefig(out_path, dpi=130, bbox_inches="tight",
                        facecolor=BACKGROUND)
            plt.close(fig)
            print(f"  Saved → {out_path}")


# ── Group-level ROI comparison ───────────────────────────────────────────────

def plot_roi_group_comparison(
    buckets,
    class_names,
    save_dir,
    atlas_path,
    modality="pet",
    roi_group=None,
):
    """
    For each confusion pair, collect per-ROI mean intensity across ALL subjects
    of each class (not just within pairs), then produce a grouped strip+boxplot
    per ROI showing class A vs class B as two separate distributions.

    This is statistically valid because:
      - All class-A subjects in the bucket share the same misclassification
        condition, so they form a comparable group.
      - All class-B subjects likewise form a comparable group.
      - Any subject from group A can be compared to any subject from group B.

    For each ROI a Mann-Whitney U test is run (non-parametric, appropriate for
    small n). ROIs are sorted by effect size (absolute Cohen's d) so the most
    discriminative regions appear first.

    Output files
    ------------
      roigrp_<modality>_<pair_tag>_<lobe>.png

    Each figure: one subplot per ROI (sorted by separation), showing:
      - Box + whisker for each class group
      - Individual subject dots (strip plot) overlaid
      - p-value annotation (Mann-Whitney U)
      - Effect size (Cohen's d)
    """
    from scipy.stats import mannwhitneyu

    save_dir  = Path(save_dir)
    atlas_raw = _load_atlas(atlas_path)

    hist_colors = {0: ACCENT, 1: GREEN, 2: YELLOW, 3: RED}

    def _idx(name):
        try:    return class_names.index(name)
        except: return None

    CN, sMCI, pMCI, AD = _idx("CN"), _idx("sMCI"), _idx("pMCI"), _idx("AD")

    requested = []
    if None not in (CN, sMCI):
        requested.append((CN,   sMCI, CN,   "roigrp_cn_smci_as_cn"))
        requested.append((CN,   sMCI, sMCI, "roigrp_cn_smci_as_smci"))
    if None not in (pMCI, AD):
        requested.append((pMCI, AD,   AD,   "roigrp_pmci_ad_as_ad"))
        requested.append((pMCI, AD,   pMCI, "roigrp_pmci_ad_as_pmci"))
    if None not in (sMCI, pMCI):
        requested.append((sMCI, pMCI, sMCI, "roigrp_smci_pmci_as_smci"))
        requested.append((sMCI, pMCI, pMCI, "roigrp_smci_pmci_as_pmci"))

    groups_to_plot = (
        {roi_group: ROI_GROUPS[roi_group]}
        if roi_group is not None
        else ROI_GROUPS
    )

    def _roi_mean(vol, mask):
        """Mean intensity of vol inside mask after percentile clipping."""
        vox = vol[mask]
        if vox.size < 5:
            return np.nan
        lo, hi = np.percentile(vox, 1), np.percentile(vox, 99)
        vox = vox[(vox >= lo) & (vox <= hi)]
        return float(vox.mean()) if vox.size >= 3 else np.nan

    def _cohens_d(a, b):
        a, b = np.array(a), np.array(b)
        if len(a) < 2 or len(b) < 2:
            return 0.0
        pooled_std = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
        return float((a.mean() - b.mean()) / (pooled_std + 1e-9))

    def _pval_str(p):
        if p < 0.001: return "p<0.001"
        if p < 0.01:  return "p<0.01"
        if p < 0.05:  return "p<0.05"
        return f"p={p:.2f}"

    for true_a, true_b, pred_cls, base_tag in requested:

        all_a = buckets.get((true_a, pred_cls), [])
        all_b = buckets.get((true_b, pred_cls), [])

        if not all_a or not all_b:
            print(f"  [skip roigrp] {base_tag}: empty bucket")
            continue

        name_a = class_names[true_a]
        name_b = class_names[true_b]
        pred_n = class_names[pred_cls]
        col_a  = hist_colors.get(true_a, ACCENT)
        col_b  = hist_colors.get(true_b, GREEN)

        # Prepare resampled atlas once using first sample's shape
        vol0  = all_a[0][modality].squeeze(0).numpy()
        atlas = _resample_atlas(atlas_raw, vol0.shape)

        for group_key, label_list in groups_to_plot.items():

            valid_labels = [l for l in label_list
                            if l in AAL3_LABELS and np.any(atlas == l)]
            if not valid_labels:
                continue

            # ── Collect per-subject ROI means across ALL subjects ────────────
            means_a = {l: [] for l in valid_labels}
            means_b = {l: [] for l in valid_labels}

            for sample in all_a:
                vol = sample[modality].squeeze(0).numpy()
                atl = _resample_atlas(atlas_raw, vol.shape) if vol.shape != atlas.shape else atlas
                for l in valid_labels:
                    means_a[l].append(_roi_mean(vol, atl == l))

            for sample in all_b:
                vol = sample[modality].squeeze(0).numpy()
                atl = _resample_atlas(atlas_raw, vol.shape) if vol.shape != atlas.shape else atlas
                for l in valid_labels:
                    means_b[l].append(_roi_mean(vol, atl == l))

            # ── Stats per ROI ────────────────────────────────────────────────
            roi_stats = []
            for l in valid_labels:
                a_vals = [v for v in means_a[l] if not np.isnan(v)]
                b_vals = [v for v in means_b[l] if not np.isnan(v)]
                if len(a_vals) < 2 or len(b_vals) < 2:
                    continue
                d = _cohens_d(a_vals, b_vals)
                try:
                    _, p = mannwhitneyu(a_vals, b_vals, alternative="two-sided")
                except Exception:
                    p = 1.0
                roi_stats.append((l, a_vals, b_vals, d, p))

            if not roi_stats:
                continue

            # Sort by |Cohen's d| descending — most discriminative first
            roi_stats.sort(key=lambda x: abs(x[3]), reverse=True)

            n_rois     = len(roi_stats)
            n_cols     = min(n_rois, 12)
            n_rows_fig = int(np.ceil(n_rois / n_cols))

            fig, axes = plt.subplots(
                n_rows_fig, n_cols,
                figsize=(n_cols * 2.2, n_rows_fig * 3.2),
                squeeze=False,
            )
            fig.patch.set_facecolor(BACKGROUND)
            fig.suptitle(
                f"{GROUP_TITLES[group_key]}  —  {modality.upper()}  group comparison\n"
                f"Both predicted as {pred_n}  |  "
                f"{name_a} n={len(all_a)}   {name_b} n={len(all_b)}  |  "
                f"sorted by |Cohen's d| ↓",
                fontsize=9, color=TEXT, y=1.02,
            )

            for idx, (lbl, a_vals, b_vals, d, p) in enumerate(roi_stats):
                row_i = idx // n_cols
                col_i = idx % n_cols
                ax    = axes[row_i, col_i]
                ax.set_facecolor(SURFACE)
                for spine in ax.spines.values():
                    spine.set_edgecolor(BORDER)
                ax.tick_params(colors=TEXT_DIM, labelsize=6)

                # ── Boxplot ──────────────────────────────────────────────────
                bp = ax.boxplot(
                    [a_vals, b_vals],
                    positions=[0, 1],
                    widths=0.35,
                    patch_artist=True,
                    medianprops=dict(color=TEXT, linewidth=1.5),
                    whiskerprops=dict(color=TEXT_DIM, linewidth=1),
                    capprops=dict(color=TEXT_DIM, linewidth=1),
                    flierprops=dict(marker="o", markersize=3,
                                   markerfacecolor=TEXT_DIM, alpha=0.5),
                    boxprops=dict(linewidth=0.8),
                )
                bp["boxes"][0].set_facecolor(col_a + "55")
                bp["boxes"][0].set_edgecolor(col_a)
                bp["boxes"][1].set_facecolor(col_b + "55")
                bp["boxes"][1].set_edgecolor(col_b)

                # ── Strip plot ───────────────────────────────────────────────
                np.random.seed(42)
                jitter_a = np.random.uniform(-0.08, 0.08, len(a_vals))
                jitter_b = np.random.uniform(-0.08, 0.08, len(b_vals))
                ax.scatter(0 + jitter_a, a_vals, color=col_a,
                           s=22, zorder=5, alpha=0.9, edgecolors="none")
                ax.scatter(1 + jitter_b, b_vals, color=col_b,
                           s=22, zorder=5, alpha=0.9, edgecolors="none")

                # ── Significance bar ─────────────────────────────────────────
                all_vals = a_vals + b_vals
                y_max    = max(all_vals)
                y_min    = min(all_vals)
                y_range  = max(y_max - y_min, 1e-9)
                bar_y    = y_max + y_range * 0.10
                sig_color = RED if p < 0.05 else TEXT_DIM
                ax.plot([0, 1], [bar_y, bar_y], color=sig_color, linewidth=0.8)
                ax.text(0.5, bar_y + y_range * 0.05,
                        _pval_str(p), ha="center", va="bottom",
                        fontsize=5.5, color=sig_color)

                # ── Title with effect size coloured by magnitude ─────────────
                roi_name = AAL3_LABELS[lbl]
                if abs(d) > 0.8:   d_color = RED
                elif abs(d) > 0.5: d_color = YELLOW
                else:              d_color = TEXT_DIM

                ax.set_title(f"{roi_name}\nd={d:.2f}",
                             color=TEXT, fontsize=6, pad=2)
                ax.set_xticks([0, 1])
                ax.set_xticklabels([name_a[:4], name_b[:4]], fontsize=6,
                                   color=TEXT_DIM)
                ax.set_ylabel("mean intensity" if col_i == 0 else "",
                              color=TEXT_DIM, fontsize=5.5)
                ax.set_xlim(-0.5, 1.5)
                ax.grid(axis="y", alpha=0.2, color=BORDER)

            # Hide unused axes
            for extra in range(len(roi_stats), n_rows_fig * n_cols):
                axes[extra // n_cols, extra % n_cols].set_visible(False)

            # Legend
            legend_patches = [
                mpatches.Patch(color=col_a, alpha=0.8,
                               label=f"True {name_a} → pred {pred_n}  (n={len(all_a)})"),
                mpatches.Patch(color=col_b, alpha=0.8,
                               label=f"True {name_b} → pred {pred_n}  (n={len(all_b)})"),
                mpatches.Patch(color=RED,     alpha=0.8, label="p < 0.05"),
                mpatches.Patch(color=TEXT_DIM, alpha=0.6, label="p ≥ 0.05"),
            ]
            fig.legend(handles=legend_patches, loc="lower center",
                       ncol=4, fontsize=7,
                       facecolor=SURFACE, edgecolor=BORDER,
                       labelcolor=TEXT, bbox_to_anchor=(0.5, -0.02))

            plt.tight_layout(rect=[0, 0.04, 1, 1])
            out_path = save_dir / f"{base_tag}_{modality}_{group_key}.png"
            fig.savefig(out_path, dpi=130, bbox_inches="tight",
                        facecolor=BACKGROUND)
            plt.close(fig)
            print(f"  Saved → {out_path}")