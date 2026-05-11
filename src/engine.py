from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix
)
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.baseline_model import BaselineModel
from src.utils import create_experiment_logger, print_experiment_config


# ══════════════════════════════════════════════════════════════════════════════
# DISPATCHER  –  call this from your main script
# ══════════════════════════════════════════════════════════════════════════════

def train_end_to_end(train_dataloader, val_dataloader, config, pretrained_path=None):
    """
    Entry point.  Reads config["training"]["use_mse"] and dispatches to either:
      • train_end_to_end_mse          (MSE regression, scalar output)
      • train_end_to_end_crossentropy (CrossEntropyLoss, num_classes output)
    """
    if config["training"].get("use_mse", False):
        return train_end_to_end_mse(train_dataloader, val_dataloader, config, pretrained_path=pretrained_path)
    else:
        return train_end_to_end_crossentropy(train_dataloader, val_dataloader, config, pretrained_path=pretrained_path)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER  –  build model, optimizer, scheduler  (shared boilerplate)
# ══════════════════════════════════════════════════════════════════════════════

def _build_components(config, num_classes, pretrained_path=None):
    """
    Instantiates model, optimizer, and LR scheduler from config.
    `num_classes` is passed explicitly so callers can set it to 1 (MSE) or
    the real class count (CrossEntropy).
    """
    device = config["training"]["device"]

    model = BaselineModel(
        class_num=num_classes,
        fusion_method=config["model"]["fusion_type"],
        pretrained=config["model"]["pretrained"],
        pretrained_path=pretrained_path
    ).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
        momentum=config["optimizer"]["momentum"]
    )

    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config["scheduler"]["T_0"],
        T_mult=config["scheduler"]["T_mult"],
        eta_min=config["scheduler"]["eta_min"]
    )

    return model, optimizer, lr_scheduler


# ══════════════════════════════════════════════════════════════════════════════
# VARIANT 1  –  MSE regression
# ══════════════════════════════════════════════════════════════════════════════

def train_end_to_end_mse(train_dataloader, val_dataloader, config, pretrained_path=None):
    """
    Trains the model as a scalar regressor.

    • Model output : (B, 1)  – a single continuous value per sample
    • Loss         : MSELoss between the predicted scalar and the integer label
                     (labels are cast to float, e.g. 0.0 / 1.0 / 2.0 / 3.0)
    • Prediction   : round the scalar to the nearest integer, then clamp to
                     [0, num_classes-1] so out-of-range predictions still map
                     to a valid class
    """
    experiment_name = config["experiment"]["name"]
    log_file, logger = create_experiment_logger(experiment_name)
    device = config["training"]["device"]
    epochs = config["training"]["epochs"]
    num_classes = config["model"]["num_classes"]

    # ── model outputs a single scalar ──────────────────────────────────────
    model, optimizer, lr_scheduler = _build_components(config, num_classes=1, pretrained_path=pretrained_path)

    criterion = nn.MSELoss()

    best_f1 = 0.0
    print_experiment_config(config)
    print(f"[MSE mode] Using device: {device}")

    for epoch in range(epochs):

        # ── TRAIN ────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        train_bar = tqdm(
            train_dataloader,
            desc=f"Train Epoch[{epoch+1}/{epochs}]"
        )

        for batch in train_bar:
            mri_data = batch["mri"].to(device)
            pet_data = batch["pet"].to(device)
            # Cast labels to float for MSE  →  shape (B,)
            labels = batch["label"].float().to(device)

            optimizer.zero_grad()

            # outputs shape: (B, 1)  →  squeeze to (B,)
            outputs = model(mri_data, pet_data).squeeze(1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_dataloader)
        lr_scheduler.step()

        # ── VALIDATION ───────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_dataloader:
                mri_data = batch["mri"].to(device)
                pet_data = batch["pet"].to(device)
                labels = batch["label"].float().to(device)

                outputs = model(mri_data, pet_data).squeeze(1)   # (B,)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Nearest-integer prediction, clamped to valid class range
                preds = outputs.round().long().clamp(0, num_classes - 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.long().cpu().numpy())

        val_loss /= len(val_dataloader)

        # ── METRICS ──────────────────────────────────────────────────────
        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

        # ── SAVE BEST MODEL ───────────────────────────────────────────────
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(
                model.state_dict(),
                f"[{config['experiment']['name']}]_mse.pth"
            )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {acc:.4f} | "
            f"F1: {macro_f1:.4f} | "
            f"Recall: {macro_recall:.4f}"
        )

        current_lr = optimizer.param_groups[0]["lr"]
        logger.writerow([epoch + 1, train_loss, val_loss, acc, macro_f1, macro_recall, current_lr])
        log_file.flush()

    log_file.close()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# VARIANT 2  –  CrossEntropy classification  (original logic, unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def train_end_to_end_crossentropy(train_dataloader, val_dataloader, config, pretrained_path=None):
    """
    Original CrossEntropyLoss training, kept intact and renamed so both
    variants live side-by-side.
    """
    experiment_name = config["experiment"]["name"]
    log_file, logger = create_experiment_logger(experiment_name)
    device = config["training"]["device"]
    epochs = config["training"]["epochs"]
    num_classes = config["model"]["num_classes"]

    model, optimizer, lr_scheduler = _build_components(config, num_classes=num_classes, pretrained_path=pretrained_path)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_f1 = 0.0
    print_experiment_config(config)
    print(f"[CrossEntropy mode] Using device: {device}")

    for epoch in range(epochs):

        # ── TRAIN ────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        train_bar = tqdm(
            train_dataloader,
            desc=f"Train Epoch[{epoch+1}/{epochs}]"
        )

        for batch in train_bar:
            mri_data = batch["mri"].to(device)
            pet_data = batch["pet"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(mri_data, pet_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_dataloader)
        lr_scheduler.step()

        # ── VALIDATION ───────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_dataloader:
                mri_data = batch["mri"].to(device)
                pet_data = batch["pet"].to(device)
                labels = batch["label"].to(device)

                outputs = model(mri_data, pet_data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_dataloader)

        # ── METRICS ──────────────────────────────────────────────────────
        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

        # ── SAVE BEST MODEL ───────────────────────────────────────────────
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(
                model.state_dict(),
                f"[{config['experiment']['name']}].pth"
            )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {acc:.4f} | "
            f"F1: {macro_f1:.4f} | "
            f"Recall: {macro_recall:.4f}"
        )

        current_lr = optimizer.param_groups[0]["lr"]
        logger.writerow([epoch + 1, train_loss, val_loss, acc, macro_f1, macro_recall, current_lr])
        log_file.flush()

    log_file.close()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# TEST  –  works for both variants
# ══════════════════════════════════════════════════════════════════════════════

def test_model(test_loader, model_path, config):
    """
    Loads a saved model and evaluates it on the test set.

    Handles both output shapes automatically:
      • MSE mode (use_mse=true)  → model output (B, 1), nearest-integer decoding
      • CE  mode (use_mse=false) → model output (B, C), argmax decoding
    """
    device = config["training"]["device"]
    num_classes = config["model"]["num_classes"]
    use_mse = config["training"].get("use_mse", False)

    # ── BUILD MODEL ──────────────────────────────────────────────────────────
    model = BaselineModel(
        class_num=1 if use_mse else num_classes,
        fusion_method=config["model"]["fusion_type"]
    ).to(device)

    # ── LOAD WEIGHTS ─────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    print(f"Testing on device: {device}  |  mode: {'MSE' if use_mse else 'CrossEntropy'}")

    # ── INFERENCE ────────────────────────────────────────────────────────────
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            mri_data = batch["mri"].to(device)
            pet_data = batch["pet"].to(device)
            labels = batch["label"]

            outputs = model(mri_data, pet_data)

            if use_mse:
                # (B, 1) → (B,) scalar, round to nearest class
                preds = outputs.squeeze(1).round().long().clamp(0, num_classes - 1)
            else:
                # (B, C) → argmax
                preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # ── METRICS ──────────────────────────────────────────────────────────────
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    print("\n===== TEST RESULTS =====")
    print(f"Accuracy:     {acc:.4f}")
    print(f"Macro F1:     {macro_f1:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")

    # ── CLASSIFICATION REPORT ────────────────────────────────────────────────
    class_names = config["model"]["class_names"]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    # ── CONFUSION MATRIX ─────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "confusion_matrix": cm
    }