import os
import subprocess
import argparse
import sys
from utils.config_loader import load_config, get_variant_params

def main():
    parser = argparse.ArgumentParser(description="Run HOPE EMA Ablation Testing (Table IV)")
    parser.add_argument('--kfold', type=int, default=5, help='Number of folds')
    parser.add_argument('--specific_fold', type=int, default=-1, help='Specific fold to run (distributed mode)')
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/datasets/kisokoghan/paired-npz/paired_npz', help='Path to NPZ data')
    parser.add_argument('--variant', type=str, default=None, help='Variant key from pipeline_config.json (runs only that EMA variant)')
    parser.add_argument('--config', type=str, default=None, help='Path to pipeline_config.json')
    args = parser.parse_args()

    ema_momentums = [None, 0.5, 0.8, 0.9, 0.99, 0.999]

    # If --variant is provided, run only that specific EMA variant
    if args.variant is not None:
        config = load_config(args.config)
        params = get_variant_params(args.variant, config)
        m_value = params.get('m', 0.9)
        # Map 1.0 to None to match original convention
        ema_momentums = [None if m_value == 1.0 else m_value]
    
    for m in ema_momentums:
        variant_name = str(m) if m is not None else "None"
        for test_target in ['2c', '3c', '4c']:
            print(f"\n{'='*60}")
            print(f"TESTING EMA ABLATION: {variant_name} | Target Model: best_{test_target}_net")
            print(f"FOLD(S): {args.specific_fold if args.specific_fold != -1 else 'ALL'}")
            print(f"{'='*60}\n")
            
            cmd = [
                sys.executable, "test.py",
                "--data_dir", args.data_dir,
                "--kfold", str(args.kfold),
                "--specific_fold", str(args.specific_fold),
                "--load_dir", f"./checkpoints/ablation_ema_{variant_name}",
                "--test_target", test_target,
                "--name", f"ablation_ema_{variant_name}",
                "--checkpoints_dir", "./checkpoints",
                "--gpu_ids", "0"
            ]
            
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
