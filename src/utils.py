import yaml
import os
import csv

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