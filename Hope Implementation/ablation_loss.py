import os
import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Run HOPE Loss Ablation Study (Table III)")
    parser.add_argument('--kfold', type=int, default=5, help='Number of folds')
    parser.add_argument('--specific_fold', type=int, default=-1, help='Distributed execution: run only this specific fold (1-based)')
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/datasets/kisokoghan/paired-npz/paired_npz', help='Path to NPZ data')
    args = parser.parse_args()

    # Table III Loss Variants
    # 'ce'      = L_CE only
    # 'ins2ins' = L_CE + L_Ins2Ins
    # 'ins2cls' = L_CE + L_Ins2Ins + L_Ins2Cls
    # 'full'    = L_CE + L_Ins2Ins + L_Ins2Cls + L_Cls2Cls (RankLoss)
    loss_variants = ['ce', 'ins2ins', 'ins2cls', 'full']
    
    print(f"Starting Loss Ablation Study (Table III)")
    print(f"K-Fold Split: {args.kfold}")
    print(f"Specific Fold Target: {args.specific_fold if args.specific_fold != -1 else 'ALL'}")
    
    for loss_type in loss_variants:
        print(f"\n{'='*50}")
        print(f"RUNNING LOSS VARIANT: {loss_type.upper()}")
        print(f"{'='*50}\n")
        
        # Build command for train.py
        cmd = [
            sys.executable, "train.py",
            "--data_dir", args.data_dir,
            "--kfold", str(args.kfold),
            "--specific_fold", str(args.specific_fold),
            "--m", "0.9", # Lock EMA momentum to 0.9 as requested by the user
            "--ablation_loss", loss_type,
            "--gpu_ids", "0",
            "--epoch_count", "30",
            "--checkpoints_dir", "./checkpoints",
            "--name", f"ablation_loss_{loss_type}"
        ]
        
        # Execute the training run
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
