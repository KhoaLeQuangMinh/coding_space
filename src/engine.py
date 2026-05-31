"""
engine.py
=========
Loosely-coupled training / evaluation engine factory.

Public surface
--------------
build_model(args)                 -> nn.Module
build_optimizer(model, args)      -> Optimizer
build_scheduler(optimizer, args)  -> LRScheduler
build_criterion(args)             -> (criterion_fn, decode_fn) or (criterion_fn)

train(train_loader, val_loader, args, pretrained_path)
"""

import os
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    classification_report, confusion_matrix,
)
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils import create_experiment_logger, print_experiment_config
from src.engine_core.standard import StandardEngine
from src.engine_core.hope import HopeEngine
from src.hope_loss import HopeLossCriterion
from src.models.hope_resnet import resnet18 as hope_resnet18


def build_model(args, pretrained_path=None) -> nn.Module:
    out_classes = 1 if getattr(args, "loss", None) == "mse" else args.num_classes
    model_type  = getattr(args, "model_type", "fusion")
    do_pretrain = getattr(args, "pretrained", False) and (pretrained_path is not None)

    if model_type == "mri_only":
        from src.baseline_model import MriOnlyModel
        model = MriOnlyModel(
            out_feature_dim = args.feature_dim,
            class_num       = out_classes,
            pretrained      = do_pretrain,
            pretrained_path = pretrained_path,
        )

    elif model_type == "pet_only":
        from src.baseline_model import PetOnlyModel
        model = PetOnlyModel(
            out_feature_dim = args.feature_dim,
            class_num       = out_classes,
            pretrained      = do_pretrain,
            pretrained_path = pretrained_path,
        )

    elif model_type == "hope_resnet":
        m_val = getattr(args, "m", 0.9)
        model = hope_resnet18(num_classes=args.num_classes, m=m_val)
        # Note: HOPE model trains from scratch with Kaiming init.

    else:   # "fusion"
        from src.baseline_model import BaselineModel
        model = BaselineModel(
            fusion_method   = args.fusion_type,
            out_feature_dim = args.feature_dim,
            class_num       = out_classes,
            pretrained      = do_pretrain,
            pretrained_path = pretrained_path,
        )

    return model.to(args.device)


def build_optimizer(model: nn.Module, args) -> optim.Optimizer:
    if getattr(args, "training_mode", "standard") == "hope":
        return optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    return optim.SGD(
        model.parameters(),
        lr          = args.lr,
        weight_decay= args.weight_decay,
        momentum    = args.momentum,
    )


def build_scheduler(optimizer: optim.Optimizer, args):
    if getattr(args, "training_mode", "standard") == "hope":
        decay = getattr(args, "lr_decay", 0.95)
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay)
    return optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0   = args.T_0,
        T_mult= args.T_mult,
        eta_min= args.eta_min,
    )


def build_criterion(args):
    if getattr(args, "training_mode", "standard") == "hope":
        criterion = HopeLossCriterion(class_num=args.num_classes, lambda_val=getattr(args, "lambda_val", 1.0))
        return criterion, None

    # Standard losses
    if args.loss == "crossentropy":
        smoothing = getattr(args, "label_smoothing", 0.1)
        criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)

        def decode_fn(outputs, _args):
            return torch.argmax(outputs, dim=1)

    elif args.loss == "mse":
        criterion = nn.MSELoss()

        def decode_fn(outputs, _args):
            return outputs.round().long().clamp(0, _args.num_classes - 1)

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

def train(train_loader, val_loader, args, pretrained_path=None):
    print_experiment_config(args)
    print(f"[{args.training_mode.upper()} mode]  device: {args.device}\n")

    # If starting from scratch (no resume), delete old CSV log to avoid appending to dirty logs
    if not getattr(args, "resume", False):
        print(f"No resume flag provided. Deleting any previous training logs for '{args.experiment_name}' and starting from scratch...")
        log_path = os.path.join("outputs", "logs", f"{args.experiment_name}.csv")
        if os.path.exists(log_path):
            try:
                os.remove(log_path)
            except OSError:
                pass

    log_file, logger = create_experiment_logger(args.experiment_name)
    
    criterion_tuple = build_criterion(args)
    model      = build_model(args, pretrained_path)
    optimizer  = build_optimizer(model, args)
    scheduler  = build_scheduler(optimizer, args)

    if args.training_mode == "hope":
        engine = HopeEngine(criterion_tuple[0])
    else:
        engine = StandardEngine(criterion_tuple[0], criterion_tuple[1])

    best_f1    = 0.0
    model_path = f"[{args.experiment_name}].pth"
    checkpoint_path = os.path.join("outputs", "runs", args.experiment_name, "checkpoint_latest.pth")
    start_epoch = 0

    if getattr(args, "resume", False):
        if os.path.exists(checkpoint_path):
            print(f"Resuming training from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_f1 = checkpoint['best_f1']
            print(f"Successfully resumed from epoch {start_epoch} (Best Val F1 so far: {best_f1:.4f})")
        else:
            print(f"No checkpoint found at {checkpoint_path}. Starting training from scratch...")

    for epoch in range(start_epoch, args.epochs):
        train_loss = engine.train_one_epoch(
            model, train_loader, optimizer, args, current_epoch=epoch+1, total_epochs=args.epochs
        )
        scheduler.step()

        val_loss, acc, macro_f1, macro_rec = engine.evaluate(
            model, val_loader, args
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

        # Save checkpoint at the end of each epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f1': best_f1,
        }
        torch.save(checkpoint, checkpoint_path)

    log_file.close()

    # Clean up checkpoint file upon successful completion of training to save disk space
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except OSError:
            pass

    print(f"\nBest val F1: {best_f1:.4f}  — weights saved to {model_path}")
    return model