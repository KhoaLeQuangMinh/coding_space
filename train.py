"""
train.py
========
Entry point for training.  All configuration comes from CLI arguments —
no YAML files required.

After argument parsing the script:
  1.  Creates  outputs/runs/<experiment_name>/run_config.txt
  2.  Writes every argument to the top of that file (run snapshot)
  3.  Attaches a Tee so all subsequent print() output also lands there

Example
-------
python train.py \\
    --experiment_name  mri_pet_concat_ce \\
    --data_root        /data/paired_npz \\
    --fusion_type      concat \\
    --loss             crossentropy \\
    --epochs           40 \\
    --lr               1e-3
"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader, random_split

from src.data    import MRIPETDataset
from src.engine  import train, test_model
from src.utils   import set_global_seed, seed_worker, save_run_config


# ══════════════════════════════════════════════════════════════════════════════
# Argument parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a multimodal MRI+PET Alzheimer classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Identity ──────────────────────────────────────────────────────────
    p.add_argument("--experiment_name", type=str, required=True,
                   help="Unique name; used for log files and checkpoint name")

    # ── Data ──────────────────────────────────────────────────────────────
    p.add_argument("--data_root",       type=str, required=True,
                   help="Path to directory of .npz subject files")
    p.add_argument("--pretrained_path", type=str, default=None,
                   help="Path to pretrained ViT checkpoint (.pth.tar)")
    p.add_argument("--train_ratio",     type=float, default=0.7)
    p.add_argument("--val_ratio",       type=float, default=0.1)
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--num_workers",     type=int,   default=4)

    # ── Model ─────────────────────────────────────────────────────────────
    p.add_argument("--fusion_type",  type=str,  default="concat",
                   choices=["concat", "sum", "film", "gated", "CrossAttention"],
                   help="Fusion module used after the two ViT backbones")
    p.add_argument("--num_classes",  type=int,  default=4)
    p.add_argument("--feature_dim",  type=int,  default=768,
                   help="ViT embedding dimension (must match backbone)")
    p.add_argument("--pretrained",   action="store_true",
                   help="Load pretrained ViT weights from --pretrained_path")
    p.add_argument("--class_names",  type=str,  nargs="+",
                   default=["CN", "sMCI", "pMCI", "AD"])

    # ── Loss ──────────────────────────────────────────────────────────────
    p.add_argument("--loss",            type=str, default="crossentropy",
                   choices=["crossentropy", "mse", "focal"],
                   help="Loss function (crossentropy | mse | focal)")
    p.add_argument("--label_smoothing", type=float, default=0.1,
                   help="Label smoothing for crossentropy loss")
    p.add_argument("--focal_gamma",     type=float, default=2.0,
                   help="Gamma for focal loss")

    # ── Training ──────────────────────────────────────────────────────────
    p.add_argument("--device",  type=str,  default="cuda:0")
    p.add_argument("--epochs",  type=int,  default=40)
    p.add_argument("--seed",    type=int,  default=12345)

    # ── Optimizer ─────────────────────────────────────────────────────────
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--momentum",     type=float, default=0.9)

    # ── Scheduler ─────────────────────────────────────────────────────────
    p.add_argument("--T_0",    type=int,   default=10)
    p.add_argument("--T_mult", type=int,   default=3)
    p.add_argument("--eta_min",type=float, default=1e-5)

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_global_seed(args.seed)

    # ── Run log: captures all CLI args + training stdout ──────────────────
    run_log_path = os.path.join(
        "outputs", "runs", args.experiment_name, "run_config.txt"
    )
    tee = save_run_config(args, run_log_path)  # sys.stdout now mirrors to file

    # ── Dataset & splits ──────────────────────────────────────────────────
    dataset   = MRIPETDataset(root=args.data_root)
    generator = torch.Generator().manual_seed(args.seed)

    train_size = int(args.train_ratio * len(dataset))
    val_size   = int(args.val_ratio   * len(dataset))
    test_size  = len(dataset) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    print(
        f"Split — Train: {len(train_ds)} | "
        f"Val: {len(val_ds)} | Test: {len(test_ds)}"
    )

    g = torch.Generator().manual_seed(args.seed)
    loader_kwargs = dict(
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        worker_init_fn = seed_worker,
        generator   = g,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    # ── Train ─────────────────────────────────────────────────────────────
    train(
        train_loader  = train_loader,
        val_loader    = val_loader,
        args          = args,
        pretrained_path = args.pretrained_path,
    )

    # ── Test ──────────────────────────────────────────────────────────────
    model_path = f"[{args.experiment_name}].pth"
    test_model(
        test_loader = test_loader,
        model_path  = model_path,
        args        = args,
    )

    # ── Restore stdout & close log ────────────────────────────────────────
    sys.stdout = tee.restore()
    print(f"\nRun log saved to: {run_log_path}")


if __name__ == "__main__":
    main()