import os
import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Run HOPE EMA Ablation Study (Table IV)")
    parser.add_argument('--kfold', type=int, default=5, help='Number of folds')
    parser.add_argument('--specific_fold', type=int, default=-1, help='Distributed execution: run only this specific fold (1-based)')
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/datasets/kisokoghan/paired-npz/paired_npz', help='Path to NPZ data')
    args = parser.parse_args()

    # Table IV EMA Momentums: None (represented mathematically as 1.0), 0.5, 0.8, 0.9, 0.99, 0.999
    ema_variants = [1.0, 0.5, 0.8, 0.9, 0.99, 0.999]
    
    print(f"Starting EMA Ablation Study (Table IV)")
    print(f"K-Fold Split: {args.kfold}")
    print(f"Specific Fold Target: {args.specific_fold if args.specific_fold != -1 else 'ALL'}")
    
    for m in ema_variants:
        variant_name = "None" if m == 1.0 else str(m)
        print(f"\n{'='*50}")
        print(f"RUNNING EMA VARIANT: m = {variant_name}")
        print(f"{'='*50}\n")
        
        # Build command for train.py
        cmd = [
            sys.executable, "train.py",
            "--data_dir", args.data_dir,
            "--kfold", str(args.kfold),
            "--specific_fold", str(args.specific_fold),
            "--m", str(m),
            "--ablation_loss", "full", # Lock loss to full for EMA ablation
            "--gpu_ids", "0",
            "--name", f"ablation_ema_{variant_name}"
        ]
        
        # Execute the training run
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
