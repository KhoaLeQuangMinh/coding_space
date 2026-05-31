"""
train.py
========
Entry point for training. All configuration comes from CLI arguments.

Two modes
---------
Default (--kfold 0):
    Standard single train/val/test split. Fast, good for quick iteration.

KFold (--kfold 5):
    Stratified K-fold cross-validation on the train+val pool.
    The held-out test set is carved out first and never touched by any fold.
    At the end, prints mean ± std of val F1 across all folds.
"""

import argparse
import copy
import os
import sys
import json

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, random_split

from src.data import MRIPETDataset, MockDataset, HopeBatchSampler
from src.engine import train
from src.utils import set_global_seed, seed_worker, save_run_config


# ══════════════════════════════════════════════════════════════════════════════
# Argument parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a multimodal MRI+PET Alzheimer classifier or HOPE baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Identity ──────────────────────────────────────────────────────────
    p.add_argument("--experiment_name", type=str, required=True,
                   help="Unique name; used for log files and checkpoint name")

    # ── Mode ──────────────────────────────────────────────────────────────
    p.add_argument("--training_mode", type=str, default="standard",
                   choices=["standard", "hope"],
                   help="standard: Baseline ViT/Fusion models. hope: exact HOPE replication.")
    p.add_argument("--mock_data", action="store_true",
                   help="Use randomly generated tensors instead of actual data for testing.")

    # ── Data ──────────────────────────────────────────────────────────────
    p.add_argument("--data_root",       type=str, default="/data/paired_npz")
    p.add_argument("--pretrained_path", type=str, default=None)
    p.add_argument("--train_ratio",     type=float, default=0.7,
                   help="Ignored when --kfold > 0")
    p.add_argument("--val_ratio",       type=float, default=0.1,
                   help="Ignored when --kfold > 0; test_ratio = 1 - train_ratio - val_ratio")
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--num_workers",     type=int,   default=4)

    # ── Model ─────────────────────────────────────────────────────────────
    p.add_argument("--model_type", type=str, default="fusion",
               choices=["fusion", "mri_only", "pet_only", "hope_resnet"])
    p.add_argument("--fusion_type",  type=str,  default="concat",
                   choices=["concat", "sum", "film", "gated", "CrossAttention"])
    p.add_argument("--num_classes",  type=int,  default=4)
    p.add_argument("--feature_dim",  type=int,  default=768)
    p.add_argument("--pretrained",   action="store_true")
    p.add_argument("--class_names",  type=str,  nargs="+",
                   default=["CN", "sMCI", "pMCI", "AD"])

    # ── Loss ──────────────────────────────────────────────────────────────
    p.add_argument("--loss",            type=str, default="crossentropy",
                   choices=["crossentropy", "mse", "focal", "hope"])
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--focal_gamma",     type=float, default=2.0)
    p.add_argument("--lambda_val",      type=float, default=1.0, help="Lambda for HOPE RankLoss")

    # ── Training ──────────────────────────────────────────────────────────
    p.add_argument("--device",  type=str, default="cuda:0")
    p.add_argument("--epochs",  type=int, default=40)
    p.add_argument("--seed",    type=int, default=12345)
    
    # ── KFold ─────────────────────────────────────────────────────────────
    p.add_argument("--kfold", type=int, default=0,
                   help="Number of folds.  0 = disabled (standard split).")

    # ── Optimizer ─────────────────────────────────────────────────────────
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--momentum",     type=float, default=0.9)

    # ── Scheduler ─────────────────────────────────────────────────────────
    p.add_argument("--T_0",     type=int,   default=10)
    p.add_argument("--T_mult",  type=int,   default=3)
    p.add_argument("--eta_min", type=float, default=1e-5)

    args = p.parse_args()
    
    # Validation and overrides for HOPE
    if args.training_mode == "hope":
        args.model_type = "hope_resnet"
        args.loss = "hope"
        args.num_classes = 3
        args.class_names = ["CN", "MCI", "AD"]
        
    if args.mock_data:
        args.device = "cpu"

    return args


# ══════════════════════════════════════════════════════════════════════════════
# Shared DataLoader builder
# ══════════════════════════════════════════════════════════════════════════════

def make_loader(dataset, indices, shuffle, args):
    """Build a reproducible DataLoader from an explicit index list."""
    subset = Subset(dataset, indices)

    if args.training_mode == "hope" and shuffle:
        # HOPE mode: use custom batch sampler for balanced CN/MCI/AD batches.
        # batch_sampler is incompatible with batch_size, shuffle, and generator.
        all_labels = dataset.get_labels()
        subset_labels = all_labels[indices]
        sampler = HopeBatchSampler(subset_labels, args.batch_size)

        return DataLoader(
            subset,
            batch_sampler  = sampler,
            num_workers    = args.num_workers,
            worker_init_fn = seed_worker,
        )
    else:
        g = torch.Generator().manual_seed(args.seed)
        return DataLoader(
            subset,
            batch_size     = args.batch_size,
            shuffle        = shuffle,
            num_workers    = args.num_workers,
            worker_init_fn = seed_worker,
            generator      = g,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Standard single-split training
# ══════════════════════════════════════════════════════════════════════════════

def run_single(dataset, args):
    generator  = torch.Generator().manual_seed(args.seed)
    train_size = int(args.train_ratio * len(dataset))
    val_size   = int(args.val_ratio   * len(dataset))
    test_size  = len(dataset) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    print(f"Split — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = make_loader(dataset, train_ds.indices, shuffle=True,  args=args)
    val_loader   = make_loader(dataset, val_ds.indices,   shuffle=False, args=args)

    train(
        train_loader    = train_loader,
        val_loader      = val_loader,
        args            = args,
        pretrained_path = args.pretrained_path,
    )

# ══════════════════════════════════════════════════════════════════════════════
# KFold training
# ══════════════════════════════════════════════════════════════════════════════

def run_kfold(dataset, args):
    K = args.kfold
    all_indices = np.arange(len(dataset))
    all_labels = dataset.get_labels()

    test_ratio    = 1.0 - args.train_ratio - args.val_ratio
    test_size     = int(test_ratio * len(dataset))

    rng              = np.random.default_rng(args.seed)
    shuffled         = rng.permutation(all_indices)
    test_indices     = shuffled[:test_size]
    trainval_indices = shuffled[test_size:]
    trainval_labels  = all_labels[trainval_indices]

    print(f"KFold={K} | TrainVal pool: {len(trainval_indices)} | "
          f"Test (held-out): {len(test_indices)}")
    print(f"Each fold — Train: ~{int(len(trainval_indices) * (K-1) / K)} | "
          f"Val: ~{int(len(trainval_indices) / K)}\n")

    skf      = StratifiedKFold(n_splits=K, shuffle=True, random_state=args.seed)
    fold_f1s = []
    best_f1  = -1.0
    best_fold = 0

    for fold, (train_idx_local, val_idx_local) in enumerate(
        skf.split(trainval_indices, trainval_labels), start=1
    ):
        train_global = trainval_indices[train_idx_local]
        val_global   = trainval_indices[val_idx_local]

        print("=" * 62)
        print(f"  FOLD {fold}/{K}  |  train={len(train_global)}  val={len(val_global)}")
        print("=" * 62)

        train_loader = make_loader(dataset, train_global, shuffle=True,  args=args)
        val_loader   = make_loader(dataset, val_global,   shuffle=False, args=args)

        fold_args = copy.copy(args)
        fold_args.experiment_name = f"{args.experiment_name}_fold{fold}"

        train(
            train_loader    = train_loader,
            val_loader      = val_loader,
            args            = fold_args,
            pretrained_path = args.pretrained_path,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_global_seed(args.seed)

    run_log_dir = os.path.join("outputs", "runs", args.experiment_name)
    os.makedirs(run_log_dir, exist_ok=True)
    
    # Save args for analysis
    args_json_path = os.path.join(run_log_dir, "args.json")
    with open(args_json_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    run_log_path = os.path.join(run_log_dir, "run_config.txt")
    tee = save_run_config(args, run_log_path)

    merge_mci = (args.training_mode == "hope")
    if args.mock_data:
        dataset = MockDataset(size=40, merge_mci=merge_mci)
    else:
        dataset = MRIPETDataset(root=args.data_root, merge_mci=merge_mci)
        
    print(f"Dataset: {len(dataset)} subjects\n")

    if args.kfold > 0:
        run_kfold(dataset, args)
    else:
        run_single(dataset, args)

    sys.stdout = tee.restore()
    print(f"\nRun log saved to: {run_log_path}")
    print(f"Args saved to: {args_json_path}")


if __name__ == "__main__":
    main()