"""
engine.py
=========
Loosely-coupled training / evaluation engine.

Public surface
--------------
build_model(args)                 -> nn.Module
build_optimizer(model, args)      -> Optimizer
build_scheduler(optimizer, args)  -> LRScheduler
build_criterion(args)             -> (criterion_fn, decode_fn)
    criterion_fn(outputs, labels) -> scalar loss
    decode_fn(outputs, args)      -> integer class predictions  (1-D LongTensor)

train(train_loader, val_loader, args, pretrained_path)
test_model(test_loader, model_path, args)
"""

from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    classification_report, confusion_matrix,
)
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.baseline_model import BaselineModel
from src.utils import create_experiment_logger, print_experiment_config


# ══════════════════════════════════════════════════════════════════════════════
# Factory helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_model(args, pretrained_path=None) -> nn.Module:
    """
    Instantiate a BaselineModel from a flat args namespace.

    num_classes is set to 1 when loss=='mse' (scalar regression head)
    and to args.num_classes for every classification loss.
    """
    out_classes = 1 if args.loss == "mse" else args.num_classes

    model = BaselineModel(
        fusion_method  = args.fusion_type,
        out_feature_dim= args.feature_dim,
        class_num      = out_classes,
        pretrained     = args.pretrained,
        pretrained_path= pretrained_path,
    )
    return model.to(args.device)


def build_optimizer(model: nn.Module, args) -> optim.Optimizer:
    """
    Build an SGD optimizer.  Swap the body here to support Adam etc. without
    touching anything else.
    """
    return optim.SGD(
        model.parameters(),
        lr          = args.lr,
        weight_decay= args.weight_decay,
        momentum    = args.momentum,
    )


def build_scheduler(optimizer: optim.Optimizer, args):
    """
    CosineAnnealingWarmRestarts scheduler.
    """
    return optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0   = args.T_0,
        T_mult= args.T_mult,
        eta_min= args.eta_min,
    )


def build_criterion(args):
    """
    Return a ``(criterion_fn, decode_fn)`` pair that matches the chosen loss.

    criterion_fn(outputs, labels) -> scalar tensor loss
    decode_fn(outputs, args)      -> 1-D LongTensor of predicted class indices

    Supported values of args.loss
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    "crossentropy"  CrossEntropyLoss with label smoothing (default 0.1).
                    decode_fn = argmax over class dimension.

    "mse"           MSELoss; model head has 1 output neuron.
                    decode_fn = round scalar to nearest integer, clamp to
                    [0, num_classes-1].

    "focal"         Focal loss (gamma=2) — useful for class-imbalanced data.
                    decode_fn = argmax (same as crossentropy).

    Adding a new loss
    ~~~~~~~~~~~~~~~~~
    Add an ``elif args.loss == "your_name":`` branch, define criterion and
    decode_fn, done.  No other file needs to change.
    """

    if args.loss == "crossentropy":
        smoothing = getattr(args, "label_smoothing", 0.1)
        criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)

        def decode_fn(outputs, _args):
            return torch.argmax(outputs, dim=1)

    elif args.loss == "mse":
        criterion = nn.MSELoss()

        def decode_fn(outputs, _args):
            return (
                outputs.squeeze(1)
                       .round()
                       .long()
                       .clamp(0, _args.num_classes - 1)
            )

    elif args.loss == "focal":
        gamma     = getattr(args, "focal_gamma", 2.0)
        ce_loss   = nn.CrossEntropyLoss(reduction="none")

        def criterion(outputs, labels):
            ce   = ce_loss(outputs, labels)
            pt   = torch.exp(-ce)
            return ((1 - pt) ** gamma * ce).mean()

        def decode_fn(outputs, _args):
            return torch.argmax(outputs, dim=1)

    else:
        raise ValueError(
            f"Unknown loss '{args.loss}'. "
            "Choose from: crossentropy | mse | focal"
        )

    return criterion, decode_fn


# ══════════════════════════════════════════════════════════════════════════════
# Inner loops
# ══════════════════════════════════════════════════════════════════════════════

def _train_one_epoch(model, loader, criterion, decode_fn, optimizer, args):
    model.train()
    total_loss = 0.0

    bar = tqdm(loader, desc="Train", leave=False)
    for batch in bar:
        mri    = batch["mri"].to(args.device)
        pet    = batch["pet"].to(args.device)

        if args.loss == "mse":
            labels = batch["label"].float().to(args.device)
        else:
            labels = batch["label"].to(args.device)

        optimizer.zero_grad()
        outputs = model(mri, pet)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


def _evaluate(model, loader, criterion, decode_fn, args):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            mri    = batch["mri"].to(args.device)
            pet    = batch["pet"].to(args.device)

            if args.loss == "mse":
                labels = batch["label"].float().to(args.device)
            else:
                labels = batch["label"].to(args.device)

            outputs = model(mri, pet)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()

            preds = decode_fn(outputs, args)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(
                labels.long().cpu().numpy()
                if args.loss == "mse"
                else labels.cpu().numpy()
            )

    avg_loss   = total_loss / len(loader)
    acc        = accuracy_score(all_labels, all_preds)
    macro_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_rec  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1, macro_rec


# ══════════════════════════════════════════════════════════════════════════════
# Public: train
# ══════════════════════════════════════════════════════════════════════════════

def train(train_loader, val_loader, args, pretrained_path=None):
    """
    Full training loop.

    Parameters
    ----------
    train_loader, val_loader : DataLoader
    args                     : argparse.Namespace  (flat, all fields present)
    pretrained_path          : str | None

    Returns
    -------
    model : the best-F1 model (weights already saved to disk)
    """
    print_experiment_config(args)
    print(f"[{args.loss.upper()} mode]  device: {args.device}\n")

    log_file, logger = create_experiment_logger(args.experiment_name)
    criterion, decode_fn = build_criterion(args)
    model      = build_model(args, pretrained_path)
    optimizer  = build_optimizer(model, args)
    scheduler  = build_scheduler(optimizer, args)

    best_f1    = 0.0
    model_path = f"[{args.experiment_name}].pth"

    for epoch in range(args.epochs):
        train_loss = _train_one_epoch(
            model, train_loader, criterion, decode_fn, optimizer, args
        )
        scheduler.step()

        val_loss, acc, macro_f1, macro_rec = _evaluate(
            model, val_loader, criterion, decode_fn, args
        )

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), model_path)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:>3}/{args.epochs} | "
            f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
            f"Acc {acc:.4f} | F1 {macro_f1:.4f} | "
            f"Rec {macro_rec:.4f} | LR {current_lr:.6f}"
        )

        logger.writerow([epoch + 1, train_loss, val_loss, acc, macro_f1, macro_rec, current_lr])
        log_file.flush()

    log_file.close()
    print(f"\nBest val F1: {best_f1:.4f}  — weights saved to {model_path}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Public: test
# ══════════════════════════════════════════════════════════════════════════════

def test_model(test_loader, model_path: str, args):
    """
    Load a saved checkpoint and evaluate on the test split.

    Works for any loss variant because it reuses ``build_criterion``.
    """
    criterion, decode_fn = build_criterion(args)
    model = build_model(args)                               # no pretrained path needed
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.eval()

    all_preds, all_labels = [], []

    print(f"\nTesting  |  loss={args.loss}  |  device={args.device}")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            mri    = batch["mri"].to(args.device)
            pet    = batch["pet"].to(args.device)
            labels = batch["label"]

            outputs = model(mri, pet)
            preds   = decode_fn(outputs, args)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc       = accuracy_score(all_labels, all_preds)
    macro_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    cm        = confusion_matrix(all_labels, all_preds)

    print("\n===== TEST RESULTS =====")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  Macro F1     : {macro_f1:.4f}")
    print(f"  Macro Recall : {macro_rec:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=args.class_names,
        zero_division=0,
    ))
    print("Confusion Matrix:")
    print(cm)

    return {
        "accuracy":         acc,
        "macro_f1":         macro_f1,
        "macro_recall":     macro_rec,
        "confusion_matrix": cm,
    }