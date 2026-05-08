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
from src.utils import create_experiment_logger

def train_end_to_end(
    train_dataloader,
    val_dataloader,
    config,
):
    experiment_name = config["experiment"]["name"]

    log_file, logger = (
        create_experiment_logger(
            experiment_name
        )
    )
    device = config["training"]["device"]

    model = BaselineModel(
        class_num=config["model"]["num_classes"],
        fusion_method=config["model"]["fusion_type"]
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

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

    epochs = config["training"]["epochs"]

    best_f1 = 0.0

    print(f"Using device: {device}")

    for epoch in range(epochs):

        # ======================
        # TRAIN
        # ======================
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

            train_bar.set_postfix({
                "loss": f"{loss.item():.4f}"
            })

        train_loss /= len(train_dataloader)

        lr_scheduler.step()

        # ======================
        # VALIDATION
        # ======================
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

                all_preds.extend(
                    preds.cpu().numpy()
                )

                all_labels.extend(
                    labels.cpu().numpy()
                )

        val_loss /= len(val_dataloader)

        # ======================
        # METRICS
        # ======================
        acc = accuracy_score(
            all_labels,
            all_preds
        )

        macro_f1 = f1_score(
            all_labels,
            all_preds,
            average="macro"
        )

        macro_recall = recall_score(
            all_labels,
            all_preds,
            average="macro"
        )

        # ======================
        # SAVE BEST MODEL
        # ======================
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
        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        logger.writerow([
            epoch + 1,
            train_loss,
            val_loss,
            acc,
            macro_f1,
            macro_recall,
            current_lr
        ])

        log_file.flush()
    log_file.close()
    return model

def test_model(
    test_loader,
    model_path,
    config
):

    device = config["training"]["device"]

    # ======================
    # BUILD MODEL
    # ======================
    model = BaselineModel(
        class_num=config["model"]["num_classes"],
        fusion_method=config["model"]["fusion_type"]
    ).to(device)

    # ======================
    # LOAD WEIGHTS
    # ======================
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.eval()

    all_preds = []
    all_labels = []

    print(f"Testing on device: {device}")

    # ======================
    # INFERENCE
    # ======================
    with torch.no_grad():

        for batch in tqdm(
            test_loader,
            desc="Testing"
        ):

            mri_data = batch["mri"].to(device)
            pet_data = batch["pet"].to(device)

            labels = batch["label"]

            outputs = model(
                mri_data,
                pet_data
            )

            preds = torch.argmax(
                outputs,
                dim=1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    # ======================
    # METRICS
    # ======================
    acc = accuracy_score(
        all_labels,
        all_preds
    )

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro"
    )

    macro_recall = recall_score(
        all_labels,
        all_preds,
        average="macro"
    )

    print("\n===== TEST RESULTS =====")

    print(f"Accuracy: {acc:.4f}")

    print(f"Macro F1: {macro_f1:.4f}")

    print(f"Macro Recall: {macro_recall:.4f}")

    # ======================
    # CLASS REPORT
    # ======================
    class_names = config["model"]["class_names"]

    print("\nClassification Report:")

    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            zero_division=0
        )
    )

    # ======================
    # CONFUSION MATRIX
    # ======================
    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "confusion_matrix": cm
    }