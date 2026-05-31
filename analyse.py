"""
analyse.py
==========
Unified analysis script for all training modes.
Automatically loads configuration from the experiment's args.json.
Generates metrics, confusion matrices, and training curves.

Usage:
    python analyse.py --experiment_name <name>
    python analyse.py --experiment_name <name> --device cpu
"""

import argparse
import json
import os
import torch
from torch.utils.data import DataLoader

from src.data import MRIPETDataset, MockDataset
from src.engine import build_model
from src.analyzers import StandardAnalyzer, HopeAnalyzer


def load_args(experiment_name):
    """Load the args.json that was saved during training."""
    json_path = os.path.join("outputs", "runs", experiment_name, "args.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Configuration file not found: {json_path}\n"
            f"Make sure training was run with --experiment_name {experiment_name}"
        )

    with open(json_path, 'r') as f:
        args_dict = json.load(f)

    return argparse.Namespace(**args_dict)


def main():
    parser = argparse.ArgumentParser(
        description="Unified Analysis Script — automatically loads training config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment_name", type=str, required=True,
                        help="Name of the experiment to analyze")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (e.g. 'cpu' for local Mac)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override data_root if different from training")

    cmd_args = parser.parse_args()

    # Load training configuration
    args = load_args(cmd_args.experiment_name)

    # Allow overrides
    if cmd_args.device:
        args.device = cmd_args.device
    if cmd_args.data_root:
        args.data_root = cmd_args.data_root

    print(f"Loaded configuration for experiment: {args.experiment_name}")
    print(f"  Training Mode : {args.training_mode.upper()}")
    print(f"  Model Type    : {args.model_type}")
    print(f"  Num Classes   : {args.num_classes}")
    print(f"  Device        : {args.device}")

    # Setup Paths
    run_dir = os.path.join("outputs", "runs", args.experiment_name)
    model_path = f"[{args.experiment_name}].pth"
    csv_log_path = f"outputs/logs/{args.experiment_name}.csv"

    if not os.path.exists(model_path):
        print(f"\nERROR: Model weights not found at {model_path}")
        print("Make sure training completed successfully.")
        return

    # Load Dataset
    merge_mci = (args.training_mode == "hope")
    mock_data = getattr(args, 'mock_data', False)

    if mock_data:
        print("\nUsing MockData for analysis...")
        dataset = MockDataset(size=40, merge_mci=merge_mci)
    else:
        print(f"\nLoading dataset from: {args.data_root}")
        dataset = MRIPETDataset(root=args.data_root, merge_mci=merge_mci)

    print(f"Dataset size: {len(dataset)} subjects")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=getattr(args, 'num_workers', 0),
    )

    # Load Model
    model = build_model(args)
    state_dict = torch.load(model_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    # Run Analyzer
    if args.training_mode == "hope":
        analyzer = HopeAnalyzer(model, loader, args, csv_log_path, run_dir)
    else:
        analyzer = StandardAnalyzer(model, loader, args, csv_log_path, run_dir)

    analyzer.run()

    print(f"\nAnalysis complete. Visualizations saved to: {run_dir}/")


if __name__ == "__main__":
    main()