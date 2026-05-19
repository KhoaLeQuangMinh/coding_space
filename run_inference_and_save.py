"""
run_inference_and_save.py
=========================
Run inference ONCE and persist everything needed by interactive_viewer.py.

What is saved
-------------
outputs/inference/<experiment_name>/
  results.json          — per-subject: subject_id, true_label, pred_label, conf
  volumes/<subject_id>_mri.npy   — float32 (H,W,D)
  volumes/<subject_id>_pet.npy   — float32 (H,W,D)

Usage (Kaggle / terminal)
--------------------------
python run_inference_and_save.py \\
    --experiment_name  mri_pet_concat_ce \\
    --data_root        /kaggle/input/adni-npz/paired_npz \\
    --fusion_type      concat \\
    --loss             crossentropy \\
    --split            test \\
    --device           cuda:0

Optional flags (same as analyse.py)
-------------------------------------
  --kfold 5 --fold 3       (for k-fold checkpoints)
  --model_type  mri_only   (unimodal baselines)
  --train_ratio 0.7 --val_ratio 0.15
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data    import MRIPETDataset
from src.engine  import build_model
from src.utils   import set_global_seed


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_decode_score(args):
    """Return (decode_fn, score_fn) matching the training loss."""
    if args.loss in ("crossentropy", "focal"):
        def decode_fn(outputs):
            return torch.argmax(outputs, dim=1)
        def score_fn(outputs):
            # returns (N, C) softmax — confidence = prob of predicted class
            return F.softmax(outputs, dim=1).cpu().numpy()
    elif args.loss == "mse":
        def decode_fn(outputs):
            return outputs.squeeze(1).round().long().clamp(0, args.num_classes - 1)
        def score_fn(outputs):
            return outputs.squeeze(1).cpu().numpy()
    else:
        raise ValueError(f"Unknown loss: {args.loss}")
    return decode_fn, score_fn


# ─────────────────────────────────────────────────────────────────────────────
# Core: run inference and stream-save volumes
# ─────────────────────────────────────────────────────────────────────────────

def run_and_save(model, loader, decode_fn, score_fn, args, out_dir: Path):
    """
    Runs inference over `loader`, saves per-subject volumes and a results JSON.

    Returns
    -------
    results : list of dicts  (one per subject)
    """
    vol_dir = out_dir / "volumes"
    vol_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            mri    = batch["mri"].to(args.device)
            pet    = batch["pet"].to(args.device)
            labels = batch["label"]            # CPU LongTensor
            sids   = batch["subject_id"]

            outputs = model(mri, pet)
            preds   = decode_fn(outputs).cpu()
            scores  = score_fn(outputs)        # ndarray  (N,C) or (N,)

            for i in range(len(labels)):
                true_cls = int(labels[i])
                pred_cls = int(preds[i])
                sid      = sids[i].item() if hasattr(sids[i], "item") else str(sids[i])

                # confidence scalar
                if scores.ndim == 2:
                    conf = float(scores[i, pred_cls])
                else:
                    residual = abs(float(scores[i]) - true_cls)
                    conf = max(0.0, 1.0 - residual / max(args.num_classes - 1, 1))

                # ── save volumes ─────────────────────────────────────────────
                mri_vol = mri[i].squeeze(0).cpu().numpy().astype(np.float32)  # (H,W,D)
                pet_vol = pet[i].squeeze(0).cpu().numpy().astype(np.float32)

                mri_path = vol_dir / f"{sid}_mri.npy"
                pet_path = vol_dir / f"{sid}_pet.npy"
                np.save(str(mri_path), mri_vol)
                np.save(str(pet_path), pet_vol)

                results.append({
                    "subject_id":    sid,
                    "true_label":    true_cls,
                    "true_name":     args.class_names[true_cls],
                    "pred_label":    pred_cls,
                    "pred_name":     args.class_names[pred_cls],
                    "confidence":    round(conf, 5),
                    "correct":       true_cls == pred_cls,
                    "mri_path":      str(mri_path),
                    "pet_path":      str(pet_path),
                })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Save inference results for interactive_viewer.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Paths ──────────────────────────────────────────────────────────────
    p.add_argument("--experiment_name", type=str, required=True)
    p.add_argument("--data_root",       type=str, required=True)

    # ── Split ratios (must match train.py) ────────────────────────────────
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio",   type=float, default=0.15)
    p.add_argument("--split",       type=str,   default="test",
                   choices=["train", "val", "test"])

    # ── Data loader ───────────────────────────────────────────────────────
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)

    # ── Model ─────────────────────────────────────────────────────────────
    p.add_argument("--model_type",  type=str, default="fusion",
                   choices=["fusion", "mri_only", "pet_only"])
    p.add_argument("--fusion_type", type=str, default="concat",
                   choices=["concat", "sum", "film", "gated", "CrossAttention"])
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--feature_dim", type=int, default=768)
    p.add_argument("--pretrained",  action="store_true")
    p.add_argument("--class_names", type=str, nargs="+",
                   default=["CN", "sMCI", "pMCI", "AD"])

    # ── Loss ──────────────────────────────────────────────────────────────
    p.add_argument("--loss", type=str, default="crossentropy",
                   choices=["crossentropy", "mse", "focal"])

    # ── KFold ─────────────────────────────────────────────────────────────
    p.add_argument("--kfold", type=int, default=0)
    p.add_argument("--fold",  type=int, default=None)

    # ── Misc ──────────────────────────────────────────────────────────────
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed",   type=int, default=12345)

    return p.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)

    out_dir = Path("outputs") / "inference" / args.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  Experiment : {args.experiment_name}")
    print(f"  Split      : {args.split}")
    print(f"  Output dir : {out_dir}")
    print(f"{'='*62}\n")

    # ── Dataset & split ───────────────────────────────────────────────────
    dataset = MRIPETDataset(root=args.data_root)

    if args.kfold > 0:
        from torch.utils.data import Subset
        all_idx  = np.arange(len(dataset))
        test_ratio = 1.0 - args.train_ratio - args.val_ratio
        test_size  = int(test_ratio * len(dataset))
        rng        = np.random.default_rng(args.seed)
        shuffled   = rng.permutation(all_idx)
        chosen_ds  = Subset(dataset, shuffled[:test_size])
        print(f"  KFold mode — {len(chosen_ds)} test samples\n")
    else:
        gen      = torch.Generator().manual_seed(args.seed)
        train_sz = int(args.train_ratio * len(dataset))
        val_sz   = int(args.val_ratio   * len(dataset))
        test_sz  = len(dataset) - train_sz - val_sz
        splits   = random_split(dataset, [train_sz, val_sz, test_sz], generator=gen)
        chosen_ds = {"train": splits[0], "val": splits[1], "test": splits[2]}[args.split]
        print(f"  Single-split — {args.split}: {len(chosen_ds)} samples\n")

    loader = DataLoader(
        chosen_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
    )

    # ── Checkpoint ────────────────────────────────────────────────────────
    if args.kfold > 0:
        if args.fold is None:
            raise ValueError("Pass --fold N when using --kfold > 0.")
        model_path = f"[{args.experiment_name}_fold{args.fold}].pth"
    else:
        model_path = f"[{args.experiment_name}].pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}\n"
            f"Train first: python train.py --experiment_name {args.experiment_name} ..."
        )

    model = build_model(args)
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    print(f"  Loaded: {model_path}\n")

    # ── Inference ─────────────────────────────────────────────────────────
    decode_fn, score_fn = _make_decode_score(args)
    results = run_and_save(model, loader, decode_fn, score_fn, args, out_dir)

    # ── Save results JSON ─────────────────────────────────────────────────
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump({
            "experiment_name": args.experiment_name,
            "class_names":     args.class_names,
            "split":           args.split,
            "subjects":        results,
        }, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────
    n_correct = sum(r["correct"] for r in results)
    n_wrong   = len(results) - n_correct
    print(f"\n  Saved {len(results)} subjects to {out_dir}/")
    print(f"  Correct: {n_correct}  |  Wrong: {n_wrong}")
    print(f"  Results JSON: {json_path}")
    print(f"\n  Run next:\n  python interactive_viewer.py --results_json {json_path}\n")


if __name__ == "__main__":
    main()