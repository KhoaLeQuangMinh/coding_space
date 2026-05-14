"""
train.py
========
Entry point for training.  All configuration comes from CLI arguments.

Two modes
---------
Default (--kfold 0):
    Standard single train/val/test split.  Fast, good for quick iteration.

KFold (--kfold 5):
    Stratified K-fold cross-validation on the train+val pool.
    The held-out test set is carved out first and never touched by any fold.
    At the end, prints mean ± std of val F1 across all folds, then evaluates
    the best fold's checkpoint on the test set.

engine.py is not touched at all — KFold only lives here.
"""

import argparse
import copy
import os
import sys

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, random_split

from src.data   import MRIPETDataset
from src.engine import train, test_model
from src.utils  import set_global_seed, seed_worker, save_run_config


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
    p.add_argument("--data_root",       type=str, required=True)
    p.add_argument("--pretrained_path", type=str, default=None)
    p.add_argument("--train_ratio",     type=float, default=0.7,
                   help="Ignored when --kfold > 0")
    p.add_argument("--val_ratio",       type=float, default=0.1,
                   help="Ignored when --kfold > 0; test_ratio = 1 - train_ratio - val_ratio")
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--num_workers",     type=int,   default=4)

    # ── Model ─────────────────────────────────────────────────────────────
    p.add_argument("--fusion_type",  type=str,  default="concat",
                   choices=["concat", "sum", "film", "gated", "CrossAttention"])
    p.add_argument("--num_classes",  type=int,  default=4)
    p.add_argument("--feature_dim",  type=int,  default=768)
    p.add_argument("--pretrained",   action="store_true")
    p.add_argument("--class_names",  type=str,  nargs="+",
                   default=["CN", "sMCI", "pMCI", "AD"])

    # ── Loss ──────────────────────────────────────────────────────────────
    p.add_argument("--loss",            type=str, default="crossentropy",
                   choices=["crossentropy", "mse", "focal"])
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--focal_gamma",     type=float, default=2.0)

    # ── Training ──────────────────────────────────────────────────────────
    p.add_argument("--device",  type=str, default="cuda:0")
    p.add_argument("--epochs",  type=int, default=40)
    p.add_argument("--seed",    type=int, default=12345)

    # ── KFold ─────────────────────────────────────────────────────────────
    p.add_argument("--kfold", type=int, default=0,
                   help="Number of folds.  0 = disabled (standard split).  "
                        "Recommended: 5.  Multiplies training time by K.")

    # ── Optimizer ─────────────────────────────────────────────────────────
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--momentum",     type=float, default=0.9)

    # ── Scheduler ─────────────────────────────────────────────────────────
    p.add_argument("--T_0",     type=int,   default=10)
    p.add_argument("--T_mult",  type=int,   default=3)
    p.add_argument("--eta_min", type=float, default=1e-5)

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Shared DataLoader builder
# ══════════════════════════════════════════════════════════════════════════════

def make_loader(dataset, indices, shuffle, args):
    """Build a reproducible DataLoader from an explicit index list."""
    g = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        Subset(dataset, indices),
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

    g = torch.Generator().manual_seed(args.seed)
    loader_kw = dict(
        batch_size     = args.batch_size,
        num_workers    = args.num_workers,
        worker_init_fn = seed_worker,
        generator      = g,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    train(
        train_loader    = train_loader,
        val_loader      = val_loader,
        args            = args,
        pretrained_path = args.pretrained_path,
    )

    model_path = f"[{args.experiment_name}].pth"
    test_model(test_loader=test_loader, model_path=model_path, args=args)


# ══════════════════════════════════════════════════════════════════════════════
# KFold training
# ══════════════════════════════════════════════════════════════════════════════

def run_kfold(dataset, args):
    """
    Stratified K-fold cross-validation.

    Strategy
    --------
    1.  Carve out a held-out test set first (same ratio as single-split mode)
        using the global seed.  This test set is identical to what single-split
        would have used, so results are directly comparable.

    2.  The remaining train+val pool is split by StratifiedKFold into K folds.
        Stratified means each fold preserves the class distribution — critical
        with only ~480 samples across 4 classes.

    3.  Each fold trains a completely fresh model and saves its best checkpoint
        as  [experiment_name]_fold{k}.pth

    4.  After all folds, print mean ± std of val F1 across folds.
        Identify the best fold and run test_model once with that checkpoint.

    Why the best fold rather than an ensemble?
    ------------------------------------------
    Ensembling gives a slightly better test number but is harder to deploy and
    harder to compare against single-model baselines in a paper.  The standard
    convention is: report mean ± std from CV, test with the best single fold.
    """
    K = args.kfold

    # ── 1. Carve out test set — identical to single-split, never seen in CV ─
    all_indices = np.arange(len(dataset))
    all_labels  = np.array([dataset[i]["label"].item() for i in all_indices])

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

    # ── 2. Stratified K-fold on the train+val pool ─────────────────────────
    skf      = StratifiedKFold(n_splits=K, shuffle=True, random_state=args.seed)
    fold_f1s = []
    best_f1  = -1.0
    best_fold = 0

    for fold, (train_idx_local, val_idx_local) in enumerate(
        skf.split(trainval_indices, trainval_labels), start=1
    ):
        # Map local fold indices back to global dataset indices
        train_global = trainval_indices[train_idx_local]
        val_global   = trainval_indices[val_idx_local]

        print("=" * 62)
        print(f"  FOLD {fold}/{K}  |  train={len(train_global)}  val={len(val_global)}")
        print("=" * 62)

        train_loader = make_loader(dataset, train_global, shuffle=True,  args=args)
        val_loader   = make_loader(dataset, val_global,   shuffle=False, args=args)

        # Each fold gets a unique experiment name so CSV logs never collide
        fold_args = copy.copy(args)
        fold_args.experiment_name = f"{args.experiment_name}_fold{fold}"

        train(
            train_loader    = train_loader,
            val_loader      = val_loader,
            args            = fold_args,
            pretrained_path = args.pretrained_path,
        )

        # Re-evaluate the saved best checkpoint on this fold's val set
        # to get a clean F1 number for the summary table
        from src.engine import build_model, build_criterion, _evaluate
        criterion, decode_fn = build_criterion(fold_args)
        model = build_model(fold_args)
        model.load_state_dict(
            torch.load(
                f"[{fold_args.experiment_name}].pth",
                map_location=args.device,
            )
        )
        _, _, fold_val_f1, _ = _evaluate(
            model, val_loader, criterion, decode_fn, fold_args
        )
        fold_f1s.append(fold_val_f1)
        print(f"\n  Fold {fold} best checkpoint val F1: {fold_val_f1:.4f}\n")

        if fold_val_f1 > best_f1:
            best_f1   = fold_val_f1
            best_fold = fold

    # ── 3. Cross-validation summary ────────────────────────────────────────
    f1_arr = np.array(fold_f1s)

    print("=" * 62)
    print(f"  CROSS-VALIDATION SUMMARY  ({K} folds)")
    print("-" * 62)
    for i, f1 in enumerate(fold_f1s, start=1):
        marker = "  <-- best" if i == best_fold else ""
        print(f"  Fold {i}:  val F1 = {f1:.4f}{marker}")
    print("-" * 62)
    print(f"  Mean ± Std :  {f1_arr.mean():.4f} ± {f1_arr.std():.4f}")
    print(f"  Best fold  :  {best_fold}  (F1 = {best_f1:.4f})")
    print("=" * 62 + "\n")

    # ── 4. Final test — run once, with the best fold's checkpoint ──────────
    best_checkpoint = f"[{args.experiment_name}_fold{best_fold}].pth"
    test_loader     = make_loader(dataset, test_indices, shuffle=False, args=args)

    print(f"Final test using fold {best_fold} checkpoint: {best_checkpoint}\n")

    # test_model reads args.loss and args.fusion_type to rebuild the model,
    # so we pass the original args (not fold_args) since they are identical
    # except for experiment_name — and model_path is passed explicitly anyway
    test_model(
        test_loader = test_loader,
        model_path  = best_checkpoint,
        args        = args,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_global_seed(args.seed)

    run_log_path = os.path.join(
        "outputs", "runs", args.experiment_name, "run_config.txt"
    )
    tee = save_run_config(args, run_log_path)

    dataset = MRIPETDataset(root=args.data_root)
    print(f"Dataset: {len(dataset)} subjects\n")

    if args.kfold > 0:
        run_kfold(dataset, args)
    else:
        run_single(dataset, args)

    sys.stdout = tee.restore()
    print(f"\nRun log saved to: {run_log_path}")


if __name__ == "__main__":
    main()