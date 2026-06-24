import os
import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Run HOPE Loss Ablation Testing (Table III)")
    parser.add_argument('--kfold', type=int, default=5, help='Number of folds')
    parser.add_argument('--specific_fold', type=int, default=-1, help='Distributed execution: test only this specific fold (1-based)')
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/datasets/kisokoghan/paired-npz/paired_npz', help='Path to NPZ data')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes for classification (e.g., 3 or 4)')
    parser.add_argument('--target_loss', type=str, required=True, help='Specific loss variant to run (e.g., ce, full, exp_triplet_ins2cls)')
    parser.add_argument('--triplet_margin', type=float, default=0.3, help='Margin for triplet relative losses')
    parser.add_argument('--m', type=float, default=0.9, help='EMA momentum')
    args = parser.parse_args()

    loss_type = args.target_loss
    class_num = args.num_classes
    
    for test_target in ['2c', '3c', '4c']:
        print(f"\n{'='*60}")
        print(f"TESTING LOSS ABLATION: {loss_type.upper()} | CLASSES: {class_num} | Target Model: best_{test_target}_net")
        print(f"FOLD(S): {args.specific_fold if args.specific_fold != -1 else 'ALL'}")
        print(f"{'='*60}\n")
        
        name_suffix = "_4class" if class_num == 4 else ""
        ema_suffix = f"_ema{args.m}" if args.m != 0.9 else ""
        margin_suffix = f"_margin{args.triplet_margin}" if args.triplet_margin != 0.3 else ""
        cmd = [
            sys.executable, "test.py",
            "--data_dir", args.data_dir,
            "--kfold", str(args.kfold),
            "--specific_fold", str(args.specific_fold),
            "--class_num", str(class_num),
            "--load_dir", f"./checkpoints/ablation_loss_{loss_type}{name_suffix}{ema_suffix}{margin_suffix}",
            "--test_target", test_target,
            "--name", f"ablation_loss_{loss_type}{name_suffix}{ema_suffix}{margin_suffix}",
            "--checkpoints_dir", "./checkpoints",
            "--gpu_ids", "0"
        ]
        
        if class_num == 4:
            cmd.extend(["--dataset", "all", "--group", "all"])
        
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
