import os
import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Run HOPE Loss Ablation Testing (Table III)")
    parser.add_argument('--kfold', type=int, default=5, help='Number of folds')
    parser.add_argument('--specific_fold', type=int, default=-1, help='Specific fold to run (distributed mode)')
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/datasets/kisokoghan/paired-npz/paired_npz', help='Path to NPZ data')
    args = parser.parse_args()

    losses = ['ce', 'ins2ins', 'ins2cls', 'full']
    
    for loss_type in losses:
        print(f"\n{'='*60}")
        print(f"TESTING LOSS ABLATION: {loss_type.upper()}")
        print(f"FOLD(S): {args.specific_fold if args.specific_fold != -1 else 'ALL'}")
        print(f"{'='*60}\n")
        
        cmd = [
            sys.executable, "test.py",
            "--data_dir", args.data_dir,
            "--kfold", str(args.kfold),
            "--specific_fold", str(args.specific_fold),
            "--load_dir", f"./checkpoints/ablation_loss_{loss_type}",
            "--epoch_count", "20",
            "--name", f"ablation_loss_{loss_type}",
            "--checkpoints_dir", "./checkpoints",
            "--gpu_ids", "0"
        ]
        
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
