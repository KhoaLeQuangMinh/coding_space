"""
gradcam_vis.py — Grad-CAM Brain Visualization for HOPE Model
=============================================================
Usage:
    python gradcam_vis.py --variant hope_original [--fold 1] [--checkpoint best_4c_net]
                          [--data_dir ../data] [--output_dir ./gradcam_output]
                          [--gpu_ids 0] [--max_samples 20]

The script will:
  1. Load the variant config from pipeline_config.json
  2. Resolve the checkpoint path automatically
  3. Run Grad-CAM (last conv layer of layer4) on all test samples
  4. Save axial / sagittal / coronal heatmap overlays
  5. Save a misclassification summary grid
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ── local imports ──────────────────────────────────────────────────────────────
from utils.Dataset import Dataset
from utils.tools import define_Cls

# ── class labels ──────────────────────────────────────────────────────────────
LABEL_MAP_4C = {0: 'CN', 1: 'sMCI', 2: 'pMCI', 3: 'AD'}
LABEL_MAP_3C = {0: 'CN', 1: 'MCI',  2: 'AD'}
CLASS_COLORS  = {0: '#2196F3', 1: '#FF9800', 2: '#F44336', 3: '#9C27B0'}

# ── custom hot-blue colormap for heatmap ─────────────────────────────────────
_cam_cmap = LinearSegmentedColormap.from_list(
    'cam', ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF0000'], N=256)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Config helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)

def get_variant_params(variant: str, config: dict) -> dict:
    variants = config.get('variants', config)
    if variant not in variants:
        raise KeyError(f"Variant '{variant}' not found in pipeline_config.json")
    return variants[variant]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Grad-CAM (pure-PyTorch, no extra library needed)
# ══════════════════════════════════════════════════════════════════════════════

class GradCAM3D:
    """Hooks into the target layer and computes the 3-D class-activation map."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None

        # forward hook — save feature maps
        target_layer.register_forward_hook(self._save_activation)
        # backward hook — save gradients
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, images: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Returns a 3-D CAM volume (D, H, W) normalised to [0, 1],
        upsampled to match the original input spatial resolution.

        Model forward returns: (x_ori [features], x [main logits], spmci_prob)
        We use x (position [1]) — the main num_classes classifier logits.
        """
        self.model.eval()
        images = images.clone().requires_grad_(True)

        _, logits, _ = self.model(images)         # forward — use main classifier (pos 1)
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()                          # backward

        # pool gradients over spatial dims  →  (C,)
        weights = self.gradients[0].mean(dim=(1, 2, 3))   # (512,)
        cam = (weights[:, None, None, None] * self.activations[0]).sum(0)  # (D, H, W)
        cam = F.relu(cam)

        # upsample to original MRI size
        cam = cam.unsqueeze(0).unsqueeze(0)               # (1,1,D,H,W)
        input_size = images.shape[2:]
        cam = F.interpolate(cam, size=input_size, mode='trilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # normalise
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Overlay helper
# ══════════════════════════════════════════════════════════════════════════════

def _overlay(ax, mri_slice: np.ndarray, cam_slice: np.ndarray,
             alpha: float = 0.45, title: str = ''):
    """Plot MRI slice with semi-transparent CAM heatmap overlay."""
    ax.imshow(mri_slice, cmap='gray', interpolation='bilinear')
    ax.imshow(cam_slice,  cmap=_cam_cmap, alpha=alpha,
              interpolation='bilinear', vmin=0, vmax=1)
    ax.set_title(title, fontsize=8, pad=2)
    ax.axis('off')


def plot_three_plane(mri_vol: np.ndarray, cam_vol: np.ndarray,
                     true_label: str, pred_label: str,
                     correct: bool, sample_id: int,
                     save_path: str):
    """
    One figure per sample: axial · sagittal · coronal slices,
    both raw MRI and CAM overlay side-by-side.
    """
    D, H, W = mri_vol.shape
    ax_idx  = D // 2
    sag_idx = W // 2
    cor_idx = H // 2

    fig = plt.figure(figsize=(14, 5))
    status = '✓ Correct' if correct else '✗ Wrong'
    color  = '#2e7d32' if correct else '#c62828'
    fig.suptitle(
        f'Sample #{sample_id}  |  True: {true_label}  Pred: {pred_label}  [{status}]',
        fontsize=11, fontweight='bold', color=color, y=1.01)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.05, wspace=0.05)

    planes = [
        ('Axial',    mri_vol[ax_idx, :, :],  cam_vol[ax_idx, :, :]),
        ('Sagittal', mri_vol[:, :, sag_idx],  cam_vol[:, :, sag_idx]),
        ('Coronal',  mri_vol[:, cor_idx, :],  cam_vol[:, cor_idx, :]),
    ]
    for col, (plane_name, mri_slice, cam_slice) in enumerate(planes):
        ax_top = fig.add_subplot(gs[0, col])
        ax_bot = fig.add_subplot(gs[1, col])
        ax_top.imshow(mri_slice, cmap='gray', interpolation='bilinear')
        ax_top.set_title(plane_name, fontsize=9)
        ax_top.axis('off')
        _overlay(ax_bot, mri_slice, cam_slice, title='+ Grad-CAM')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Summary grid (misclassified only)
# ══════════════════════════════════════════════════════════════════════════════

def plot_misclassification_grid(records: list, save_path: str, num_classes: int):
    """
    Grid of axial-slice CAM overlays for all misclassified samples,
    grouped by true class.
    """
    if not records:
        print("  No misclassifications — skipping grid.")
        return

    label_map = LABEL_MAP_4C if num_classes == 4 else LABEL_MAP_3C
    n = len(records)
    ncols = min(n, 6)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, rec in enumerate(records):
        mri_vol, cam_vol = rec['mri'], rec['cam']
        ax_idx  = mri_vol.shape[0] // 2
        _overlay(axes[i],
                 mri_vol[ax_idx], cam_vol[ax_idx],
                 title=f"T:{rec['true']} P:{rec['pred']}\n#{rec['id']}", alpha=0.5)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle('Misclassified Samples — Axial Grad-CAM', fontsize=13,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Misclassification grid → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Grad-CAM brain visualization')
    # ── Kaggle shortcut: set this to the dataset root and checkpoints_dir is auto-resolved
    #    e.g. --kaggle_input /kaggle/input/notebooks/baygiokemmuoi/hope-original
    #    resolves to:  <kaggle_input>/coding_space/Hope Implementation/checkpoints
    parser.add_argument('--kaggle_input', type=str, default=None,
                        help='Kaggle dataset root (auto-resolves checkpoints_dir and data_dir)')
    parser.add_argument('--variant',    type=str,  required=True,
                        help='Pipeline config variant key (e.g. hope_original)')
    parser.add_argument('--fold',       type=int,  default=1,
                        help='Which fold checkpoint to load (1-5)')
    parser.add_argument('--checkpoint', type=str,  default='best_4c_net',
                        choices=['best_2c_net', 'best_3c_net', 'best_4c_net'],
                        # Note: extension is always .pth (Kaggle saves as .pth)
                        help='Checkpoint filename (without .pt)')
    parser.add_argument('--config',     type=str,  default='pipeline_config.json')
    parser.add_argument('--data_dir',   type=str,  default='../data')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--output_dir', type=str,  default='./gradcam_output')
    parser.add_argument('--gpu_ids',    type=str,  default='0')
    parser.add_argument('--max_samples',type=int,  default=50,
                        help='Max test samples to visualise (set 0 for all)')
    parser.add_argument('--only_wrong', action='store_true',
                        help='Only visualise misclassified samples')
    args = parser.parse_args()

    # ── resolve Kaggle paths if --kaggle_input is given ─────────────────────
    if args.kaggle_input is not None:
        kaggle_root = args.kaggle_input.rstrip('/')
        hope_impl   = os.path.join(kaggle_root, 'coding_space', 'Hope Implementation')
        args.checkpoints_dir = os.path.join(hope_impl, 'checkpoints')
        args.config = os.path.join(hope_impl, 'pipeline_config.json')
        print(f"  [Kaggle] checkpoints_dir → {args.checkpoints_dir}")
        print(f"  [Kaggle] config          → {args.config}")
        print(f"  [Kaggle] data_dir        → {args.data_dir}  (set via --data_dir)")
        if args.data_dir == '../data':
            print("\n  [WARNING] --data_dir is still the default '../data'.")
            print("  Pass --data_dir with the actual path to your NPZ files on Kaggle,")
            print("  e.g.  --data_dir /kaggle/working/coding_space/data\n")

    # ── load variant config ──────────────────────────────────────────────────
    config = load_config(args.config)
    params = get_variant_params(args.variant, config)
    num_classes = params.get('num_classes', 3)
    m           = params.get('m', 0.999)
    dist_ema    = params.get('dist_ema', False)
    no_classifier = params.get('no_classifier', False)

    print(f"\n{'='*60}")
    print(f"  Variant     : {args.variant}")
    print(f"  Num classes : {num_classes}")
    print(f"  Fold        : {args.fold}")
    print(f"  Checkpoint  : {args.checkpoint}.pt")
    print(f"{'='*60}\n")

    # ── resolve checkpoint path ──────────────────────────────────────────────
    expr_name   = f"ablation_loss_{args.variant}"
    ckpt_folder = os.path.join(
        args.checkpoints_dir,
        f"{expr_name}_fold{args.fold}"
    )
    ckpt_path   = os.path.join(ckpt_folder, f"{args.checkpoint}.pth")

    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found:\n  {ckpt_path}")
        print("  Make sure --checkpoints_dir points to the folder containing "
              f"'{expr_name}_fold{args.fold}/'")
        sys.exit(1)

    print(f"  Loading checkpoint: {ckpt_path}")

    # ── build model & load weights ───────────────────────────────────────────
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = define_Cls('resnet3d', class_num=num_classes, init_type='kaiming',
                       init_gain=0.02, m=m, gpu_ids=args.gpu_ids,
                       no_classifier=no_classifier, use_dist_ema=dist_ema)

    state_dict = torch.load(ckpt_path, map_location=device)
    # strip DataParallel prefix if present
    state_dict_clean = {k.replace('module.', ''): v for k, v in state_dict.items()}
    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict_clean, strict=False)
        inner = model.module
    else:
        model.load_state_dict(state_dict_clean, strict=False)
        inner = model
        
    # Explicitly load saved running prototypes (crucial for sMCI vs pMCI split)
    if 'prototypes' in state_dict:
        proto_tensor = state_dict['prototypes'].to(device)
        inner.prototypes = torch.nn.Parameter(proto_tensor, requires_grad=False)
        if hasattr(model, 'module'):
            model.module.prototypes = torch.nn.Parameter(proto_tensor, requires_grad=False)
        print("  Explicitly loaded saved prototypes from state_dict")
        
    model.eval()
    print(f"  Model loaded on {device}")

    # ── register Grad-CAM on layer4[-1] ─────────────────────────────────────
    target_layer = inner.layer4[-1]   # last BasicBlock — 512-channel feature maps
    grad_cam     = GradCAM3D(model, target_layer)
    print(f"  Grad-CAM target layer: layer4[-1]  (512 channels)")

    # ── load validation / test dataset ──────────────────────────────────────
    return_4c = (num_classes == 4)
    # Use mode='test' — the held-out KFold split (~20%, ~120 subjects)
    # This is exactly the same split used by test.py to generate the confusion matrix.
    # mode='valid' is only the ~10% validation set used during training for checkpointing.
    test_dataset = Dataset(mode='test', data_dir=args.data_dir,
                           seed=42, kfold=5, current_fold=args.fold,
                           return_4c=return_4c)
    print(f"  Test samples (held-out fold): {len(test_dataset)}\n")

    label_map = LABEL_MAP_4C if num_classes == 4 else LABEL_MAP_3C

    # ── output directory ─────────────────────────────────────────────────────
    out_root = os.path.join(args.output_dir, args.variant, f"fold{args.fold}")
    os.makedirs(out_root, exist_ok=True)
    mis_dir  = os.path.join(out_root, 'misclassified')
    ok_dir   = os.path.join(out_root, 'correct')
    os.makedirs(mis_dir, exist_ok=True)
    if not args.only_wrong:
        os.makedirs(ok_dir, exist_ok=True)

    # ── run Grad-CAM loop ────────────────────────────────────────────────────
    misclassified_records = []
    n_correct = n_wrong = 0
    limit = args.max_samples if args.max_samples > 0 else len(test_dataset)

    def _to_int(x):
        """Handle both plain Python ints and PyTorch tensor labels."""
        return x.item() if hasattr(x, 'item') else int(x)

    for idx in range(min(limit, len(test_dataset))):
        sample     = test_dataset[idx]
        mri_tensor = sample[0].unsqueeze(0).to(device)   # (1,1,D,H,W)

        # The valid dataset ALWAYS returns 4-class EVAL_MAP labels (0=CN,1=sMCI,2=pMCI,3=AD)
        # regardless of return_4c — see Dataset.py EVAL_MAP
        true_label_4c = _to_int(sample[2]) if (return_4c and len(sample) > 2) \
                        else _to_int(sample[1])

        # For 3-class models: collapse 4-class truth to 3-class for correctness check
        # (pred_idx is in 3-class space: 0=CN, 1=MCI, 2=AD)
        if num_classes == 3:
            true_for_compare = 0 if true_label_4c == 0 else (1 if true_label_4c in [1, 2] else 2)
        else:
            true_for_compare = true_label_4c

        # Always display the full 4-class name for clarity
        true_str = LABEL_MAP_4C.get(true_label_4c, str(true_label_4c))
        true_label_idx = true_label_4c  # keep for any downstream use

        # ── forward to get prediction ────────────────────────────────────────
        with torch.no_grad():
            _, logits, spmci_prob = model(mri_tensor)
            pred_idx = logits.argmax(dim=1).item()

        # If model is 3-class, resolve MCI prediction into sMCI or pMCI using prototypes
        if num_classes == 3:
            if pred_idx == 0:
                final_pred_4c = 0   # CN
            elif pred_idx == 2:
                final_pred_4c = 3   # AD
            else: # pred_idx == 1 (MCI)
                # spmci_prob has shape (1, 2) corresponding to [sMCI, pMCI]
                s = spmci_prob.argmax(dim=1).item()
                final_pred_4c = 1 if s == 0 else 2
        else: # num_classes == 4
            final_pred_4c = pred_idx

        # Both true_str and pred_str are now always in the 4-class label space
        true_str = LABEL_MAP_4C.get(true_label_4c, str(true_label_4c))
        pred_str = LABEL_MAP_4C.get(final_pred_4c, str(final_pred_4c))

        # Correctness is determined by matching the 4-class labels exactly
        correct = (final_pred_4c == true_label_4c)
        if correct:
            n_correct += 1
        else:
            n_wrong  += 1

        if args.only_wrong and correct:
            continue

        # ── compute Grad-CAM for the main predicted class index ────────────────
        cam_vol = grad_cam(mri_tensor, class_idx=pred_idx)
        mri_vol = mri_tensor.squeeze().cpu().numpy()   # (D, H, W)

        # ── save three-plane figure ──────────────────────────────────────────
        save_subdir = ok_dir if correct else mis_dir
        save_path   = os.path.join(
            save_subdir,
            f"s{idx:04d}_true{true_str}_pred{pred_str}.png"
        )
        plot_three_plane(mri_vol, cam_vol, true_str, pred_str,
                         correct, idx, save_path)

        tag = '✓' if correct else '✗'
        print(f"  [{tag}] Sample {idx:4d} | True: {true_str:5s} Pred: {pred_str:5s} → {save_path}")

        if not correct:
            misclassified_records.append({
                'id': idx, 'true': true_str, 'pred': pred_str,
                'mri': mri_vol, 'cam': cam_vol
            })

    # ── misclassification summary grid ──────────────────────────────────────
    grid_path = os.path.join(out_root, 'misclassification_grid.png')
    plot_misclassification_grid(misclassified_records, grid_path, 4)  # Always plot with 4-class maps

    # ── final stats ──────────────────────────────────────────────────────────
    total = n_correct + n_wrong
    print(f"\n{'='*60}")
    print(f"  Done!  Correct: {n_correct}/{total}  Wrong: {n_wrong}/{total}")
    print(f"  Output → {out_root}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
