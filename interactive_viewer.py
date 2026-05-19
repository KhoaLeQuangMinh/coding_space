"""
interactive_viewer.py
=====================
Interactive brain scan explorer for mis-classified subjects.

Prerequisites
-------------
1.  Run ``run_inference_and_save.py`` first to produce
    outputs/inference/<experiment_name>/results.json
    and the accompanying volume .npy files.

2.  (Optional) Supply an AAL3 atlas NIfTI file via --atlas_path.
    Without it, hover annotations show "No atlas" instead of ROI names.

Usage
-----
python interactive_viewer.py \\
    --results_json  outputs/inference/mri_pet_concat_ce/results.json \\
    --atlas_path    /kaggle/input/aal3-atlas/AAL3v1_1mm.nii.gz

Controls
--------
  LEFT PANEL  — subject list (wrong / correct toggle)
    • Click a subject row to load its MRI + PET volumes

  CENTRE PANEL — three linked slice views (axial / sagittal / coronal)
    • Scroll wheel         : step through slices
    • Hover over any slice : look up AAL region under cursor,
                             show intensity histogram of that region
                             in the right panel

  RIGHT PANEL — live ROI histogram
    • Updates on mouse move (throttled to ~20 Hz)
    • Shows MRI (blue) and PET (orange) intensity distributions
      for the ROI under the cursor

  Keyboard
    m   — toggle between MRI / PET / Both overlay modes
    n   — step to next wrong subject
    p   — step to previous wrong subject
    q   — quit

Dependencies
------------
    pip install matplotlib numpy scipy nibabel
    (nibabel only needed if using a .nii/.nii.gz atlas)
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")           # works on Kaggle with display; fall back below
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
import numpy as np
from scipy.ndimage import zoom


# ─────────────────────────────────────────────────────────────────────────────
# AAL3 label map  (copied from plot_roi_histogram.py — kept self-contained)
# ─────────────────────────────────────────────────────────────────────────────

AAL3_LABELS = {
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
    4001: "ACG.L",      4002: "ACG.R",
    4011: "MCG.L",      4012: "MCG.R",
    4021: "PCG.L",      4022: "PCG.R",
    4101: "HIP.L",      4102: "HIP.R",
    4111: "PHG.L",      4112: "PHG.R",
    4201: "Amyg.L",     4202: "Amyg.R",
    5001: "PostCG.L",   5002: "PostCG.R",
    5011: "SPG.L",      5012: "SPG.R",
    5021: "IPG.L",      5022: "IPG.R",
    5101: "SMG.L",      5102: "SMG.R",
    5201: "AG.L",       5202: "AG.R",
    5301: "Precun.L",   5302: "Precun.R",
    5401: "PCL.L",      5402: "PCL.R",
    6001: "Heschls.L",  6002: "Heschls.R",
    6101: "STG.L",      6102: "STG.R",
    6201: "TPsup.L",    6202: "TPsup.R",
    6221: "MTG.L",      6222: "MTG.R",
    6301: "TPmid.L",    6302: "TPmid.R",
    6401: "ITG.L",      6402: "ITG.R",
    7001: "CUN.L",      7002: "CUN.R",
    7011: "SOG.L",      7012: "SOG.R",
    7021: "MOG.L",      7022: "MOG.R",
    7101: "IOG.L",      7102: "IOG.R",
    8101: "Caudate.L",  8102: "Caudate.R",
    8111: "Putamen.L",  8112: "Putamen.R",
    8121: "Pallidum.L", 8122: "Pallidum.R",
    8201: "Thal.L",     8202: "Thal.R",
    8211: "Heschl2.L",  8212: "Heschl2.R",
    8301: "NAcc.L",     8302: "NAcc.R",
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

# ─────────────────────────────────────────────────────────────────────────────
# Colour theme (dark — matches the rest of the codebase)
# ─────────────────────────────────────────────────────────────────────────────

BG     = "#0d1117"
SURF   = "#161b22"
BORDER = "#30363d"
ACCENT = "#58a6ff"
GREEN  = "#3fb950"
RED    = "#f85149"
YELLOW = "#d29922"
TEXT   = "#e6edf3"
DIM    = "#8b949e"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    SURF,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "xtick.color":       DIM,
    "ytick.color":       DIM,
    "text.color":        TEXT,
    "grid.color":        BORDER,
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
    "font.family":       "monospace",
})


# ─────────────────────────────────────────────────────────────────────────────
# Atlas helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_atlas_npy(path: str) -> np.ndarray:
    """
    Load an AAL3 atlas file.  Supports:
      .npy           — already a numpy array
      .nii / .nii.gz — NIfTI (requires nibabel)
    Returns integer ndarray shape (H, W, D) or None on failure.
    """
    path = Path(path)
    if not path.exists():
        print(f"[atlas] File not found: {path}")
        return None

    try:
        if path.suffix == ".npy":
            return np.load(str(path)).astype(np.int32)

        # Try nibabel for NIfTI
        try:
            import nibabel as nib
            img = nib.load(str(path))
            data = np.asarray(img.dataobj, dtype=np.int32)
            print(f"[atlas] Loaded via nibabel: {data.shape}")
            return data
        except ImportError:
            print("[atlas] nibabel not installed — trying raw gz read")

        # Minimal NIfTI reader (header-only, single-file .nii.gz)
        import gzip, struct
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(str(path), "rb") as f:
            hdr = f.read(348)
            dims = struct.unpack_from("<8h", hdr, 40)
            ndim, nx, ny, nz = dims[0], dims[1], dims[2], dims[3]
            f.seek(0)
            raw = f.read()
        # vox_offset stored at byte 108 as float32
        vox_off = int(struct.unpack_from("<f", hdr, 108)[0])
        data = np.frombuffer(raw[vox_off:], dtype=np.int16).reshape(nx, ny, nz)
        return data.astype(np.int32)

    except Exception as e:
        print(f"[atlas] Load failed: {e}")
        return None


def _resample_to(atlas: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Nearest-neighbour zoom to match target volume shape."""
    if atlas.shape == target_shape:
        return atlas
    factors = [t / s for t, s in zip(target_shape, atlas.shape)]
    return zoom(atlas, factors, order=0, prefilter=False).astype(np.int32)


def roi_label_at(atlas_resampled: np.ndarray, x: int, y: int, z: int) -> str:
    """Return the AAL name of the voxel at (x, y, z), or '' if background."""
    if atlas_resampled is None:
        return "No atlas loaded"
    H, W, D = atlas_resampled.shape
    x = int(np.clip(x, 0, H - 1))
    y = int(np.clip(y, 0, W - 1))
    z = int(np.clip(z, 0, D - 1))
    lbl = int(atlas_resampled[x, y, z])
    if lbl == 0:
        return "Background"
    return AAL3_LABELS.get(lbl, f"Label {lbl}")


def roi_voxels(atlas_resampled: np.ndarray, x: int, y: int, z: int):
    """
    Return a boolean mask for the ROI that contains voxel (x, y, z).
    Returns None if atlas is None or voxel is background.
    """
    if atlas_resampled is None:
        return None
    lbl = int(atlas_resampled[
        int(np.clip(x, 0, atlas_resampled.shape[0]-1)),
        int(np.clip(y, 0, atlas_resampled.shape[1]-1)),
        int(np.clip(z, 0, atlas_resampled.shape[2]-1)),
    ])
    if lbl == 0:
        return None
    return atlas_resampled == lbl


# ─────────────────────────────────────────────────────────────────────────────
# Main viewer class
# ─────────────────────────────────────────────────────────────────────────────

class BrainViewer:
    """
    Interactive three-plane brain viewer with AAL ROI hover annotations.

    Layout (1 figure, multiple GridSpec regions)
    ─────────────────────────────────────────────
    ┌─────────────┬──────────────────────────────┬────────────────────┐
    │  Subject    │  Axial  │ Sagittal │ Coronal  │   ROI Histogram    │
    │  List       │ (scroll)│ (scroll) │ (scroll) │  (live, on hover)  │
    │  (click)    │                               │                    │
    │             │   slice position indicators   │  ROI name + stats  │
    └─────────────┴──────────────────────────────┴────────────────────┘
    """

    THROTTLE_SEC = 0.05   # minimum seconds between histogram refreshes

    def __init__(self, subjects: list, class_names: list, atlas=None, show_wrong_only=True):
        self.all_subjects   = subjects
        self.class_names    = class_names
        self.atlas_raw      = atlas          # full-res atlas or None
        self.atlas_cached   = None           # resampled to current vol shape
        self.atlas_vol_shape = None

        # Filter list
        self.show_wrong_only = show_wrong_only
        self._refresh_subject_list()

        # State
        self.current_idx   = 0              # index into self.display_subjects
        self.mri_vol       = None           # (H,W,D) float32
        self.pet_vol       = None           # (H,W,D) float32
        self.sx = self.sy = self.sz = 0     # current slice indices
        self.overlay_mode  = "both"         # "mri" | "pet" | "both"
        self._last_hover   = 0.0
        self._hover_roi    = None           # last ROI name

        self._build_figure()
        self._connect_events()

        # Load first subject immediately
        if self.display_subjects:
            self._load_subject(0)

    # ─── subject list management ─────────────────────────────────────────────

    def _refresh_subject_list(self):
        if self.show_wrong_only:
            self.display_subjects = [s for s in self.all_subjects if not s["correct"]]
        else:
            self.display_subjects = list(self.all_subjects)

    # ─── figure construction ─────────────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=(22, 10), facecolor=BG)
        self.fig.canvas.manager.set_window_title("Brain Scan Interactive Viewer")

        # Outer grid: [list | slices | histogram]
        outer = gridspec.GridSpec(
            1, 3, figure=self.fig,
            width_ratios=[1, 2.6, 1.4],
            wspace=0.08, left=0.01, right=0.99, top=0.93, bottom=0.08,
        )

        # ── LEFT: subject list ────────────────────────────────────────────
        self.ax_list = self.fig.add_subplot(outer[0])
        self.ax_list.set_facecolor(SURF)
        self.ax_list.set_title("Subjects", color=TEXT, fontsize=10)
        self.ax_list.axis("off")

        # toggle button (wrong / all)
        btn_ax = self.fig.add_axes([0.01, 0.01, 0.10, 0.04])
        self.btn_toggle = Button(btn_ax, "Show: wrong", color=SURF, hovercolor=BORDER)
        self.btn_toggle.label.set_color(TEXT)
        self.btn_toggle.label.set_fontsize(8)
        self.btn_toggle.on_clicked(self._toggle_filter)

        # prev / next buttons
        prv_ax = self.fig.add_axes([0.01, 0.06, 0.04, 0.04])
        nxt_ax = self.fig.add_axes([0.07, 0.06, 0.04, 0.04])
        self.btn_prev = Button(prv_ax, "◀ prev", color=SURF, hovercolor=BORDER)
        self.btn_next = Button(nxt_ax, "next ▶", color=SURF, hovercolor=BORDER)
        for b in (self.btn_prev, self.btn_next):
            b.label.set_color(TEXT)
            b.label.set_fontsize(8)
        self.btn_prev.on_clicked(lambda _: self._step_subject(-1))
        self.btn_next.on_clicked(lambda _: self._step_subject(+1))

        # ── CENTRE: three slice panels ────────────────────────────────────
        inner_slices = gridspec.GridSpecFromSubplotSpec(
            2, 3, subplot_spec=outer[1],
            height_ratios=[20, 1],
            hspace=0.05, wspace=0.05,
        )
        self.ax_axial    = self.fig.add_subplot(inner_slices[0, 0])
        self.ax_sagittal = self.fig.add_subplot(inner_slices[0, 1])
        self.ax_coronal  = self.fig.add_subplot(inner_slices[0, 2])
        self.slice_axes  = [self.ax_axial, self.ax_sagittal, self.ax_coronal]
        self.slice_labels = ["Axial (scroll Z)", "Sagittal (scroll X)", "Coronal (scroll Y)"]

        for ax, lbl in zip(self.slice_axes, self.slice_labels):
            ax.set_facecolor("#000000")
            ax.set_title(lbl, color=DIM, fontsize=9)
            ax.axis("off")

        # Status bar below slices
        self.ax_status = self.fig.add_subplot(inner_slices[1, :])
        self.ax_status.axis("off")
        self.status_txt = self.ax_status.text(
            0.5, 0.5, "", ha="center", va="center",
            fontsize=9, color=DIM, transform=self.ax_status.transAxes,
        )

        # ── RIGHT: histogram + ROI info ───────────────────────────────────
        inner_hist = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer[2],
            height_ratios=[1, 6, 1],
            hspace=0.15,
        )
        self.ax_roi_label = self.fig.add_subplot(inner_hist[0])
        self.ax_roi_label.axis("off")
        self.roi_label_txt = self.ax_roi_label.text(
            0.5, 0.5, "Hover over a brain slice",
            ha="center", va="center", fontsize=10, color=ACCENT,
            transform=self.ax_roi_label.transAxes, wrap=True,
        )

        self.ax_hist = self.fig.add_subplot(inner_hist[1])
        self.ax_hist.set_facecolor(SURF)
        self.ax_hist.set_title("ROI intensity distribution", color=TEXT, fontsize=9)

        self.ax_hist_stats = self.fig.add_subplot(inner_hist[2])
        self.ax_hist_stats.axis("off")
        self.hist_stats_txt = self.ax_hist_stats.text(
            0.5, 0.5, "", ha="center", va="center",
            fontsize=8, color=DIM, transform=self.ax_hist_stats.transAxes,
        )

        # Main title
        self.fig_title = self.fig.suptitle(
            "Loading…", fontsize=12, color=TEXT, y=0.98,
        )

        # Image handles (will be set on first draw)
        self.im_axial    = None
        self.im_sagittal = None
        self.im_coronal  = None

        # Crosshair lines
        self._ch_lines = {ax: [] for ax in self.slice_axes}

    # ─── event connections ───────────────────────────────────────────────────

    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event",  self._on_click)
        self.fig.canvas.mpl_connect("scroll_event",        self._on_scroll)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.fig.canvas.mpl_connect("key_press_event",     self._on_key)

    # ─── subject loading ─────────────────────────────────────────────────────

    def _load_subject(self, idx: int):
        if not self.display_subjects:
            return
        idx = int(np.clip(idx, 0, len(self.display_subjects) - 1))
        self.current_idx = idx
        rec = self.display_subjects[idx]

        self.mri_vol = np.load(rec["mri_path"])   # (H,W,D)
        self.pet_vol = np.load(rec["pet_path"])

        H, W, D = self.mri_vol.shape
        self.sx, self.sy, self.sz = H // 2, W // 2, D // 2

        # Re-sample atlas if needed
        if self.atlas_raw is not None:
            if self.mri_vol.shape != self.atlas_vol_shape:
                self.atlas_cached   = _resample_to(self.atlas_raw, self.mri_vol.shape)
                self.atlas_vol_shape = self.mri_vol.shape
        else:
            self.atlas_cached = None

        # Title
        true_n = rec["true_name"]
        pred_n = rec["pred_name"]
        conf   = rec["confidence"]
        status = "✗ WRONG" if not rec["correct"] else "✓ correct"
        color  = RED if not rec["correct"] else GREEN
        self.fig_title.set_text(
            f"Subject {rec['subject_id']}   |   True: {true_n}   Pred: {pred_n}   "
            f"Conf: {conf:.3f}   [{status}]"
        )
        self.fig_title.set_color(color)

        self._redraw_slices()
        self._redraw_list()
        self._update_status()
        self.fig.canvas.draw_idle()

    # ─── slice drawing ───────────────────────────────────────────────────────

    def _slice_image(self, plane: str) -> np.ndarray:
        """
        Return the 2-D image for a given plane, blended according to overlay_mode.
        Plane: "axial" | "sagittal" | "coronal"
        Returns float32 (H, W) in [0, 1].
        """
        sx, sy, sz = self.sx, self.sy, self.sz

        if plane == "axial":
            mri_sl = np.rot90(self.mri_vol[:, :, sz])
            pet_sl = np.rot90(self.pet_vol[:, :, sz])
        elif plane == "sagittal":
            mri_sl = np.rot90(self.mri_vol[sx, :, :])
            pet_sl = np.rot90(self.pet_vol[sx, :, :])
        else:  # coronal
            mri_sl = np.rot90(self.mri_vol[:, sy, :])
            pet_sl = np.rot90(self.pet_vol[:, sy, :])

        def _norm(x):
            lo, hi = x.min(), x.max()
            return (x - lo) / (hi - lo + 1e-9)

        if self.overlay_mode == "mri":
            return _norm(mri_sl)
        elif self.overlay_mode == "pet":
            return _norm(pet_sl)
        else:  # both: MRI as green, PET as hot, blended
            # Use a simple RGB blend: MRI → grey, PET → warm overlay
            grey = _norm(mri_sl)
            warm = _norm(pet_sl)
            # Stack to RGB: R = grey+warm, G = grey, B = grey
            r = np.clip(grey * 0.6 + warm * 0.6, 0, 1)
            g = np.clip(grey * 0.6,               0, 1)
            b = np.clip(grey * 0.6,               0, 1)
            return np.stack([r, g, b], axis=-1)

    def _cmap_for_mode(self):
        if self.overlay_mode == "mri":   return "gray"
        if self.overlay_mode == "pet":   return "hot"
        return None   # RGB array — no cmap needed

    def _redraw_slices(self):
        if self.mri_vol is None:
            return

        planes  = ["axial", "sagittal", "coronal"]
        handles = [self.im_axial, self.im_sagittal, self.im_coronal]
        axes    = self.slice_axes

        new_handles = []
        for ax, plane, old_im in zip(axes, planes, handles):
            img  = self._slice_image(plane)
            cmap = self._cmap_for_mode()

            if old_im is None:
                ax.axis("off")
                if cmap:
                    h = ax.imshow(img, cmap=cmap, origin="lower", aspect="equal")
                else:
                    h = ax.imshow(img, origin="lower", aspect="equal")
            else:
                old_im.set_data(img)
                if cmap:
                    old_im.set_cmap(cmap)
                h = old_im

            new_handles.append(h)

        self.im_axial, self.im_sagittal, self.im_coronal = new_handles

        # Crosshairs
        self._draw_crosshairs()

    def _draw_crosshairs(self):
        H, W, D = self.mri_vol.shape
        sx, sy, sz = self.sx, self.sy, self.sz

        for ax, lines in self._ch_lines.items():
            for l in lines:
                try:
                    l.remove()
                except Exception:
                    pass
        self._ch_lines = {ax: [] for ax in self.slice_axes}

        kw = dict(color=ACCENT, alpha=0.5, linewidth=0.8, linestyle="--")

        # Axial slice: x-axis = X (rows→H), y-axis = Y (cols→W)
        lh = self.ax_axial.axhline(sx, **kw)
        lv = self.ax_axial.axvline(sy, **kw)
        self._ch_lines[self.ax_axial] = [lh, lv]

        # Sagittal slice: axes are Y (cols→W), Z (rows→D)
        lh = self.ax_sagittal.axhline(sz, **kw)
        lv = self.ax_sagittal.axvline(sy, **kw)
        self._ch_lines[self.ax_sagittal] = [lh, lv]

        # Coronal slice: axes are X (rows→H), Z (rows→D)
        lh = self.ax_coronal.axhline(sz, **kw)
        lv = self.ax_coronal.axvline(sx, **kw)
        self._ch_lines[self.ax_coronal] = [lh, lv]

    def _update_status(self):
        if self.mri_vol is None:
            return
        H, W, D = self.mri_vol.shape
        n  = len(self.display_subjects)
        i  = self.current_idx
        mode_str = self.overlay_mode.upper()
        self.status_txt.set_text(
            f"[{i+1}/{n}]  X={self.sx}/{H-1}  Y={self.sy}/{W-1}  Z={self.sz}/{D-1}"
            f"  |  mode={mode_str}  |  scroll=change slice  |  m=toggle mode"
        )

    # ─── subject list render ─────────────────────────────────────────────────

    def _redraw_list(self):
        self.ax_list.cla()
        self.ax_list.set_facecolor(SURF)
        self.ax_list.axis("off")
        label = "Wrong subjects" if self.show_wrong_only else "All subjects"
        self.ax_list.set_title(label, color=TEXT, fontsize=9)

        subjects = self.display_subjects
        max_show = 28
        start    = max(0, self.current_idx - max_show // 2)
        end      = min(len(subjects), start + max_show)
        start    = max(0, end - max_show)

        for row_i, s_idx in enumerate(range(start, end)):
            s    = subjects[s_idx]
            y    = 1.0 - (row_i + 1) / (max_show + 1)
            active = (s_idx == self.current_idx)
            bg_col = BORDER if active else SURF
            txt_col = TEXT if active else DIM

            # Background highlight
            self.ax_list.axhspan(y - 0.015, y + 0.015,
                                  color=bg_col, alpha=0.9, zorder=1)

            status_sym = "✗" if not s["correct"] else "✓"
            sym_col    = RED  if not s["correct"] else GREEN
            label_txt  = (
                f"{status_sym} {s['subject_id']}  "
                f"{s['true_name']}→{s['pred_name']}"
            )
            self.ax_list.text(
                0.05, y, label_txt,
                ha="left", va="center",
                fontsize=7.5, color=txt_col,
                transform=self.ax_list.transAxes,
                zorder=2,
            )
            self.ax_list.text(
                0.92, y, status_sym,
                ha="right", va="center",
                fontsize=8, color=sym_col,
                transform=self.ax_list.transAxes,
                zorder=2,
            )

        self.ax_list.set_xlim(0, 1)
        self.ax_list.set_ylim(0, 1)

    # ─── ROI histogram ───────────────────────────────────────────────────────

    def _update_roi_histogram(self, roi_name: str, mask: np.ndarray | None):
        ax = self.ax_hist
        ax.cla()
        ax.set_facecolor(SURF)
        ax.set_title("ROI intensity distribution", color=TEXT, fontsize=9)

        if mask is None or self.mri_vol is None:
            ax.text(0.5, 0.5, "Background / no atlas",
                    ha="center", va="center", color=DIM,
                    transform=ax.transAxes)
            self.roi_label_txt.set_text("Background")
            self.hist_stats_txt.set_text("")
            self.fig.canvas.draw_idle()
            return

        mri_vox = self.mri_vol[mask].ravel()
        pet_vox = self.pet_vol[mask].ravel()

        def _clip(v):
            lo, hi = np.percentile(v, 1), np.percentile(v, 99)
            return v[(v >= lo) & (v <= hi)]

        mri_c = _clip(mri_vox)
        pet_c = _clip(pet_vox)

        bins = 40
        if len(mri_c) > 0:
            ax.hist(mri_c, bins=bins, color=ACCENT,  alpha=0.65, label="MRI",
                    density=True, edgecolor="none")
        if len(pet_c) > 0:
            ax.hist(pet_c, bins=bins, color=YELLOW, alpha=0.65, label="PET",
                    density=True, edgecolor="none")

        ax.legend(fontsize=8, facecolor=SURF, edgecolor=BORDER)
        ax.set_xlabel("Intensity", color=DIM, fontsize=8)
        ax.set_ylabel("Density",   color=DIM, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.3)

        # ROI name display
        self.roi_label_txt.set_text(roi_name)

        # Stats line
        stats = []
        if len(mri_c) > 0:
            stats.append(f"MRI μ={mri_c.mean():.3f}  σ={mri_c.std():.3f}")
        if len(pet_c) > 0:
            stats.append(f"PET μ={pet_c.mean():.3f}  σ={pet_c.std():.3f}")
        stats.append(f"voxels={mask.sum()}")
        self.hist_stats_txt.set_text("   ".join(stats))

        self.fig.canvas.draw_idle()

    # ─── event handlers ──────────────────────────────────────────────────────

    def _on_click(self, event):
        """Click in list panel → load that subject.  Click in slice → set crosshair."""
        if event.inaxes == self.ax_list:
            # Map y-coordinate to a subject index
            max_show = 28
            n = len(self.display_subjects)
            start = max(0, self.current_idx - max_show // 2)
            end   = min(n, start + max_show)
            start = max(0, end - max_show)

            y_frac   = event.ydata  # in data coords, ax has [0,1]
            row_i    = int((1.0 - y_frac) * (max_show + 1)) - 1
            s_idx    = start + row_i
            if 0 <= s_idx < n:
                self._load_subject(s_idx)
            return

        if event.inaxes in self.slice_axes and self.mri_vol is not None:
            self._set_crosshair_from_click(event)

    def _set_crosshair_from_click(self, event):
        H, W, D = self.mri_vol.shape
        col = int(np.clip(round(event.xdata), 0, max(H, W, D) - 1))
        row = int(np.clip(round(event.ydata), 0, max(H, W, D) - 1))

        if event.inaxes == self.ax_axial:
            self.sy = col
            self.sx = row
        elif event.inaxes == self.ax_sagittal:
            self.sy = col
            self.sz = row
        elif event.inaxes == self.ax_coronal:
            self.sx = col
            self.sz = row

        self._clamp_slices()
        self._redraw_slices()
        self._update_status()
        self.fig.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.inaxes not in self.slice_axes or self.mri_vol is None:
            return
        H, W, D = self.mri_vol.shape
        delta = 1 if event.button == "up" else -1

        if event.inaxes == self.ax_axial:
            self.sz = int(np.clip(self.sz + delta, 0, D - 1))
        elif event.inaxes == self.ax_sagittal:
            self.sx = int(np.clip(self.sx + delta, 0, H - 1))
        elif event.inaxes == self.ax_coronal:
            self.sy = int(np.clip(self.sy + delta, 0, W - 1))

        self._redraw_slices()
        self._update_status()
        self.fig.canvas.draw_idle()

    def _on_hover(self, event):
        if self.mri_vol is None:
            return
        now = time.monotonic()
        if now - self._last_hover < self.THROTTLE_SEC:
            return
        if event.inaxes not in self.slice_axes:
            return
        if event.xdata is None or event.ydata is None:
            return

        self._last_hover = now

        # Convert 2-D mouse position to 3-D voxel coordinate
        col = int(np.clip(round(event.xdata), 0, self.mri_vol.shape[1] - 1))
        row = int(np.clip(round(event.ydata), 0, self.mri_vol.shape[0] - 1))

        H, W, D = self.mri_vol.shape

        if event.inaxes == self.ax_axial:
            vx, vy, vz = row, col, self.sz
        elif event.inaxes == self.ax_sagittal:
            vx, vy, vz = self.sx, col, row
        else:  # coronal
            vx, vy, vz = col, self.sy, row

        vx = int(np.clip(vx, 0, H - 1))
        vy = int(np.clip(vy, 0, W - 1))
        vz = int(np.clip(vz, 0, D - 1))

        roi_name = roi_label_at(self.atlas_cached, vx, vy, vz)

        if roi_name != self._hover_roi:
            self._hover_roi = roi_name
            mask = roi_voxels(self.atlas_cached, vx, vy, vz)
            self._update_roi_histogram(roi_name, mask)

    def _on_key(self, event):
        if event.key == "m":
            modes = ["both", "mri", "pet"]
            cur   = modes.index(self.overlay_mode)
            self.overlay_mode = modes[(cur + 1) % len(modes)]
            self._redraw_slices()
            self._update_status()
            self.fig.canvas.draw_idle()
        elif event.key == "n":
            self._step_subject(+1)
        elif event.key == "p":
            self._step_subject(-1)
        elif event.key == "q":
            plt.close(self.fig)

    # ─── helpers ─────────────────────────────────────────────────────────────

    def _clamp_slices(self):
        if self.mri_vol is None:
            return
        H, W, D = self.mri_vol.shape
        self.sx = int(np.clip(self.sx, 0, H - 1))
        self.sy = int(np.clip(self.sy, 0, W - 1))
        self.sz = int(np.clip(self.sz, 0, D - 1))

    def _step_subject(self, delta: int):
        n = len(self.display_subjects)
        if n == 0:
            return
        new_idx = int(np.clip(self.current_idx + delta, 0, n - 1))
        self._load_subject(new_idx)

    def _toggle_filter(self, _event=None):
        self.show_wrong_only = not self.show_wrong_only
        self._refresh_subject_list()
        label = "Show: wrong" if self.show_wrong_only else "Show: all"
        self.btn_toggle.label.set_text(label)
        self.current_idx = 0
        if self.display_subjects:
            self._load_subject(0)
        else:
            self._redraw_list()
            self.fig.canvas.draw_idle()

    # ─── run ─────────────────────────────────────────────────────────────────

    def show(self):
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Interactive brain scan viewer for mis-classified subjects",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results_json", type=str, required=True,
                   help="Path to results.json produced by run_inference_and_save.py")
    p.add_argument("--atlas_path",   type=str, default=None,
                   help="AAL3 atlas file (.nii / .nii.gz / .npy). "
                        "Without this, hover shows no ROI names.")
    p.add_argument("--show_all",     action="store_true",
                   help="Start with all subjects visible (default: wrong only)")
    p.add_argument("--backend",      type=str, default="TkAgg",
                   help="Matplotlib backend. Use 'Qt5Agg' if Tk is unavailable.")
    return p.parse_args()


def main():
    args = parse_args()

    # Switch backend if requested
    try:
        matplotlib.use(args.backend)
    except Exception as e:
        print(f"[warn] Could not set backend {args.backend}: {e}")

    # ── Load results JSON ─────────────────────────────────────────────────
    json_path = Path(args.results_json)
    if not json_path.exists():
        raise FileNotFoundError(
            f"results.json not found: {json_path}\n"
            "Run run_inference_and_save.py first."
        )

    with open(json_path) as f:
        data = json.load(f)

    subjects    = data["subjects"]
    class_names = data.get("class_names", ["CN", "sMCI", "pMCI", "AD"])

    n_wrong   = sum(1 for s in subjects if not s["correct"])
    n_correct = len(subjects) - n_wrong
    print(f"\n  Loaded {len(subjects)} subjects  "
          f"(wrong: {n_wrong}  correct: {n_correct})")
    print(f"  Experiment: {data.get('experiment_name', '?')}  "
          f"Split: {data.get('split', '?')}\n")

    # ── Load atlas (optional) ─────────────────────────────────────────────
    atlas = None
    if args.atlas_path:
        print(f"  Loading atlas: {args.atlas_path}")
        atlas = _load_atlas_npy(args.atlas_path)
        if atlas is not None:
            print(f"  Atlas shape: {atlas.shape}  "
                  f"Unique labels: {len(np.unique(atlas))}")
        else:
            print("  Atlas failed to load — hover will show no ROI names.")
    else:
        print("  No atlas supplied (--atlas_path).  Hover shows no ROI names.\n")

    # ── Launch viewer ─────────────────────────────────────────────────────
    viewer = BrainViewer(
        subjects        = subjects,
        class_names     = class_names,
        atlas           = atlas,
        show_wrong_only = not args.show_all,
    )
    viewer.show()


if __name__ == "__main__":
    main()