import yaml
import os
import csv
import random
import numpy as np
import torch

def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def create_experiment_logger(experiment_name):
    
    os.makedirs(
        "outputs/logs",
        exist_ok=True
    )

    log_path = (
        f"outputs/logs/"
        f"{experiment_name}.csv"
    )

    file_exists = os.path.exists(
        log_path
    )

    file = open(
        log_path,
        mode="a",
        newline=""
    )

    writer = csv.writer(
        file
    )

    if not file_exists:

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "accuracy",
            "macro_f1",
            "macro_recall",
            "lr"
        ])

    return file, writer

def print_experiment_config(config):

    print("\n" + "=" * 60)

    print(
        f"Experiment: "
        f"{config['experiment']['name']}"
    )

    print("-" * 60)

    print(
        f"Model: "
        f"{config['model']['fusion_type']}"
    )

    print(
        f"Classes: "
        f"{config['model']['num_classes']}"
    )

    print("-" * 60)

    print(
        f"Device: "
        f"{config['training']['device']}"
    )

    print(
        f"Epochs: "
        f"{config['training']['epochs']}"
    )

    print("-" * 60)

    print(
        f"Learning Rate: "
        f"{config['optimizer']['lr']}"
    )

    print(
        f"Weight Decay: "
        f"{config['optimizer']['weight_decay']}"
    )

    print(
        f"Momentum: "
        f"{config['optimizer']['momentum']}"
    )

    print("-" * 60)

    print(
        f"Scheduler T0: "
        f"{config['scheduler']['T_0']}"
    )

    print(
        f"Scheduler T_mult: "
        f"{config['scheduler']['T_mult']}"
    )

    print("=" * 60 + "\n")


def set_global_seed(seed: int = 42, deterministic: bool = True):
    """
    Make training as reproducible as possible.
    """

    # Python
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # Torch CPU
    torch.manual_seed(seed)

    # Torch CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:

        # CuBLAS reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Force deterministic algorithms
        torch.use_deterministic_algorithms(True)

    else:
        torch.backends.cudnn.benchmark = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32

    random.seed(worker_seed)
    np.random.seed(worker_seed)