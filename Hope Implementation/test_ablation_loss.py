import os
import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Run HOPE Loss Ablation Testing (Table III)")
    parser.add_argument('--kfold', type=int, default=5, help='Number of folds')
    parser.add_argument('--specific_fold', type=int, default=-1, help='Distributed execution: test only this specific fold (1-based)')
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/datasets/kisokoghan/paired-npz/paired_npz', help='Path to NPZ data')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes for classification')
    parser.add_argument('--target_loss', type=str, default='all', help='Specific loss variant to run. Defaults to all variants.')
    args = parser.parse_args()

    # losses = ['ce', 'ins2ins', 'ins2cls', 'full']
    # loss_variants = ['ce', 'ins2ins', 'ins2cls', 'full', 'exclude_ins2ins', 'exclude_ins2cls']
    loss_variants = ['full', 'exp_triplet_ins2cls']
    if args.target_loss != 'all':
        loss_variants = [args.target_loss]
    
    for loss_type in loss_variants:
        for test_target in ['2c', '3c', '4c']:
            print(f"\n{'='*60}")
            print(f"TESTING LOSS ABLATION: {loss_type.upper()} | Target Model: best_{test_target}_net")
            print(f"FOLD(S): {args.specific_fold if args.specific_fold != -1 else 'ALL'}")
            print(f"{'='*60}\n")
            
            cmd = [
                sys.executable, "test.py",
                "--data_dir", args.data_dir,
                "--kfold", str(args.kfold),
                "--specific_fold", str(args.specific_fold),
                "--num_classes", str(args.num_classes),
                "--load_dir", f"./checkpoints/ablation_loss_{loss_type}",
                "--test_target", test_target,
                "--name", f"ablation_loss_{loss_type}",
                "--checkpoints_dir", "./checkpoints",
                "--gpu_ids", "0"
            ]
            
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
