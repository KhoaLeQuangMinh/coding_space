import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Kaggle/headless
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_confusion_matrix(cm, class_names, title, save_path):
    """Plot and save a confusion matrix heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_curves(csv_path, save_path):
    """
    Plot training curves from the CSV log produced by create_experiment_logger.

    Expected CSV columns: epoch, train_loss, val_loss, accuracy, macro_f1, macro_recall, lr
    """
    import pandas as pd

    if not os.path.exists(csv_path):
        print(f"  CSV log not found at {csv_path}. Skipping curve plotting.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Loss ---
    axes[0].plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(df['epoch'], df['val_loss'],   label='Val Loss',   linewidth=2)
    axes[0].set_title('Loss over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Accuracy & F1 ---
    axes[1].plot(df['epoch'], df['accuracy'],  label='Val Accuracy', linewidth=2)
    axes[1].plot(df['epoch'], df['macro_f1'],  label='Val Macro F1', linewidth=2)
    axes[1].plot(df['epoch'], df['macro_recall'], label='Val Macro Recall', linewidth=2)
    axes[1].set_title('Metrics over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # --- Learning Rate ---
    axes[2].plot(df['epoch'], df['lr'], label='Learning Rate', linewidth=2, color='green')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('LR')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")
