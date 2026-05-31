import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report
from tqdm import tqdm
from src.plotting import plot_confusion_matrix, plot_training_curves


class BaseAnalyzer:
    def __init__(self, model, loader, args, csv_log_path, output_dir):
        self.model = model
        self.loader = loader
        self.args = args
        self.csv_log_path = csv_log_path   # CSV from outputs/logs/<experiment>.csv
        self.output_dir = output_dir

    def run(self):
        raise NotImplementedError


class StandardAnalyzer(BaseAnalyzer):
    """Evaluates standard 4-class (CN/sMCI/pMCI/AD) models."""

    def run(self):
        print("\n===== Running Standard Evaluation =====")
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.loader, desc="Evaluation"):
                mri = batch["mri"].to(self.args.device)
                pet = batch["pet"].to(self.args.device)
                labels = batch["label"].cpu().numpy()

                outputs = self.model(mri, pet)
                if self.args.loss == "mse":
                    outputs = outputs.squeeze(1)
                    preds = outputs.round().long().clamp(0, self.args.num_classes - 1)
                else:
                    preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels)

        # Plot training curves from CSV log
        plot_training_curves(self.csv_log_path, f"{self.output_dir}/training_curves.png")

        # Metrics
        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)

        print(f"\n  Accuracy : {acc:.4f}")
        print(f"  Macro F1 : {macro_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds,
                                    target_names=self.args.class_names,
                                    zero_division=0))

        plot_confusion_matrix(cm, self.args.class_names,
                              "Confusion Matrix", f"{self.output_dir}/confusion_matrix.png")


class HopeAnalyzer(BaseAnalyzer):
    """
    Evaluates HOPE models with two outputs:
      1. 3-class prediction (CN/MCI/AD) from the classifier head
      2. 4-class prediction (CN/sMCI/pMCI/AD) using prototype similarity
    """

    # Map original string labels back to 4-class integers
    REVERSE_4CLASS = {"CN": 0, "sMCI": 1, "pMCI": 2, "AD": 3}

    def run(self):
        print("\n===== Running HOPE Evaluation =====")
        self.model.eval()

        preds_3class  = []
        preds_4class  = []
        labels_3class = []   # 0, 1, 2
        labels_4class = []   # 0, 1, 2, 3

        with torch.no_grad():
            for batch in tqdm(self.loader, desc="HOPE Evaluation"):
                mri = batch["mri"].to(self.args.device)
                labels_3c = batch["label"].cpu().numpy()
                original_str_labels = batch["original_label"]

                # Convert string labels back to 4-class
                labels_4c = [self.REVERSE_4CLASS[lbl] for lbl in original_str_labels]

                features, outputs, spmci_prob = self.model(mri)

                # 3-Class prediction from FC head
                p_3c = torch.argmax(outputs, dim=1).cpu().numpy()
                preds_3class.extend(p_3c)
                labels_3class.extend(labels_3c)

                # 4-Class prediction using prototype similarity
                # spmci_prob shape: [B, 2]  →  col 0 = sim_to_CN_proto, col 1 = sim_to_AD_proto
                # For MCI predictions: high col-1 means more like pMCI (progressing toward AD)
                spmci_np = spmci_prob.cpu().numpy()
                for i in range(len(p_3c)):
                    pred_3 = p_3c[i]
                    if pred_3 == 0:
                        preds_4class.append(0)   # CN
                    elif pred_3 == 2:
                        preds_4class.append(3)   # AD
                    else:  # pred_3 == 1 → MCI, split using prototype
                        # spmci_prob[:, 1] = similarity to AD prototype → higher means pMCI
                        if spmci_np[i, 1] > 0.5:
                            preds_4class.append(2)   # pMCI
                        else:
                            preds_4class.append(1)   # sMCI

                labels_4class.extend(labels_4c)

        # Plot training curves
        plot_training_curves(self.csv_log_path, f"{self.output_dir}/training_curves.png")

        # 3-Class report
        print("\n--- HOPE 3-Class Results (Classifier Head) ---")
        acc_3 = accuracy_score(labels_3class, preds_3class)
        f1_3 = f1_score(labels_3class, preds_3class, average="macro", zero_division=0)
        print(f"  Accuracy : {acc_3:.4f}")
        print(f"  Macro F1 : {f1_3:.4f}")
        print(classification_report(labels_3class, preds_3class,
                                    target_names=["CN", "MCI", "AD"],
                                    zero_division=0))
        cm_3 = confusion_matrix(labels_3class, preds_3class)
        plot_confusion_matrix(cm_3, ["CN", "MCI", "AD"],
                              "HOPE 3-Class Confusion Matrix",
                              f"{self.output_dir}/hope_3class_cm.png")

        # 4-Class report
        print("\n--- HOPE 4-Class Results (Prototype Splitting) ---")
        acc_4 = accuracy_score(labels_4class, preds_4class)
        f1_4 = f1_score(labels_4class, preds_4class, average="macro", zero_division=0)
        print(f"  Accuracy : {acc_4:.4f}")
        print(f"  Macro F1 : {f1_4:.4f}")
        print(classification_report(labels_4class, preds_4class,
                                    target_names=["CN", "sMCI", "pMCI", "AD"],
                                    zero_division=0))
        cm_4 = confusion_matrix(labels_4class, preds_4class)
        plot_confusion_matrix(cm_4, ["CN", "sMCI", "pMCI", "AD"],
                              "HOPE 4-Class Confusion Matrix",
                              f"{self.output_dir}/hope_4class_cm.png")
