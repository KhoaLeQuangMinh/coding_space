import numpy as np
import argparse
from pathlib import Path
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from src.utils import read_config

def check_volume(tensor):
    """
    pet_tensor: A torch.Tensor of shape (1, H, W, D) or (H, W, D)
    """
    # Remove the channel dimension if it's there (e.g., shape 1, 128, 128, 64 -> 128, 128, 64)
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
        
    # Move to CPU and convert to numpy
    data = tensor.detach().cpu().numpy()
    
    # Get middle slice indices
    mid_idx = [dim // 2 for dim in data.shape]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Axial View (H, W)
    axes[0].imshow(np.rot90(data[:, :, mid_idx[2]]), cmap='hot')
    axes[0].set_title(f'Axial - Slice {mid_idx[2]}')
    
    # Sagittal View (W, D)
    axes[1].imshow(np.rot90(data[mid_idx[0], :, :]), cmap='hot')
    axes[1].set_title(f'Sagittal - Slice {mid_idx[0]}')
    
    # Coronal View (H, D)
    axes[2].imshow(np.rot90(data[:, mid_idx[1], :]), cmap='hot')
    axes[2].set_title(f'Coronal - Slice {mid_idx[1]}')
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()


def plot_experiment(config_path):

    # ======================
    # READ CONFIG
    # ======================
    config = read_config(config_path)

    experiment_name = (
        config["experiment"]["name"]
    )

    # ======================
    # CSV PATH
    # ======================
    csv_path = (
        Path("outputs") /
        "logs" /
        f"{experiment_name}.csv"
    )

    if not csv_path.exists():

        raise FileNotFoundError(
            f"Could not find: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    save_dir = csv_path.parent

    print(
        f"Loading experiment: "
        f"{experiment_name}"
    )

    # ======================
    # LOSS CURVE
    # ======================
    plt.figure(figsize=(8, 5))

    plt.plot(
        df["epoch"],
        df["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        df["epoch"],
        df["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title(
        f"{experiment_name} Loss Curve"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        save_dir /
        f"{experiment_name}_loss_curve.png"
    )

    plt.close()

    # ======================
    # METRICS CURVE
    # ======================
    plt.figure(figsize=(8, 5))

    plt.plot(
        df["epoch"],
        df["acc"],
        label="Accuracy"
    )

    plt.plot(
        df["epoch"],
        df["f1"],
        label="Macro F1"
    )

    plt.plot(
        df["epoch"],
        df["recall"],
        label="Recall"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Score")

    plt.title(
        f"{experiment_name} Validation Metrics"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        save_dir /
        f"{experiment_name}_metrics_curve.png"
    )

    plt.close()

    print(
        "\nSaved:"
    )

    print(
        save_dir /
        f"{experiment_name}_loss_curve.png"
    )

    print(
        save_dir /
        f"{experiment_name}_metrics_curve.png"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to yaml config"
    )

    args = parser.parse_args()

    plot_experiment(
        args.config
    )