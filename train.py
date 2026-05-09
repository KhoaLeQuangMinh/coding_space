import argparse
import torch
from torch.utils.data import DataLoader, random_split

from src.data import MRIPETDataset
from src.utils import read_config
from src.engine import train_end_to_end, test_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Experiment yaml"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # system config
    config = read_config("configs/kaggle.yaml")

    # experiment config
    config_experiment = read_config(args.config)

    dataset = MRIPETDataset(
        root=config["data"]["root"]
    )

    generator = torch.Generator().manual_seed(12345)

    train_size = int(
        config["split"]["train_ratio"] * len(dataset)
    )

    val_size = int(
        config["split"]["val_ratio"] * len(dataset)
    )

    test_size = (
        len(dataset)
        - train_size
        - val_size
    )

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"]
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"]
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"]
    )

    print(
        f"Train: {len(train_ds)} | "
        f"Val: {len(val_ds)} | "
        f"Test: {len(test_ds)}"
    )

    model = train_end_to_end(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=config_experiment
    )

    results = test_model(
        test_loader=test_loader,
        model_path=f"[{config_experiment['experiment']['name']}].pth",
        config=config_experiment
    )


if __name__ == "__main__":
    main()