import os
import subprocess
import argparse
import sys
from utils.config_loader import load_config, get_variant_params

def main():
    parser = argparse.ArgumentParser(description="Run HOPE Loss Ablation Testing (Table III)")
    parser.add_argument('--kfold', type=int, default=5, help='Number of folds')
    parser.add_argument('--specific_fold', type=int, default=-1, help='Distributed execution: test only this specific fold (1-based)')
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/datasets/kisokoghan/paired-npz/paired_npz', help='Path to NPZ data')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes for classification (e.g., 3 or 4)')
    parser.add_argument('--target_loss', type=str, default=None, help='Specific loss variant to run (e.g., ce, full, exp_triplet_ins2cls)')
    parser.add_argument('--triplet_margin', type=float, default=0.3, help='Margin for triplet relative losses')
    parser.add_argument('--m', type=float, default=0.9, help='EMA momentum')
    parser.add_argument('--variant', type=str, default=None, help='Variant key from pipeline_config.json (overrides manual args)')
    parser.add_argument('--config', type=str, default=None, help='Path to pipeline_config.json')
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU IDs (e.g. 0, use empty string or -1 for CPU)')
    args = parser.parse_args()

    # Resolve variant from config if provided
    no_classifier = False
    if args.variant is not None:
        config = load_config(args.config)
        params = get_variant_params(args.variant, config)
        args.target_loss = params.get('loss_type', params.get('target_loss', args.target_loss))
        args.num_classes = params.get('num_classes', args.num_classes)
        args.triplet_margin = params.get('triplet_margin', args.triplet_margin)
        args.m = params.get('m', args.m)
        no_classifier = params.get('no_classifier', False)

    if args.target_loss is None:
        parser.error("--target_loss is required (or use --variant to load from config)")

    loss_type = args.target_loss
    class_num = args.num_classes
    
    for test_target in ['2c', '3c', '4c']:
        print(f"\n{'='*60}")
        print(f"TESTING LOSS ABLATION: {loss_type.upper()} | CLASSES: {class_num} | Target Model: best_{test_target}_net")
        print(f"FOLD(S): {args.specific_fold if args.specific_fold != -1 else 'ALL'}")
        print(f"{'='*60}\n")
        
        if args.variant is not None:
            expr_name = f"ablation_loss_{args.variant}"
        else:
            name_suffix = "_4class" if class_num == 4 else ""
            ema_suffix = f"_ema{args.m}" if args.m != 0.9 else ""
            margin_suffix = f"_margin{args.triplet_margin}" if args.triplet_margin != 0.3 else ""
            proto_suffix = "_proto" if no_classifier else ""
            
            expr_name = f"ablation_loss_{loss_type}{name_suffix}{ema_suffix}{margin_suffix}{proto_suffix}"
            if loss_type == 'triplet_only_collinear':
                expr_name = f"ablation_loss_triplet_only{name_suffix}{ema_suffix}_margin{args.triplet_margin}_collinear{proto_suffix}"
            
        cmd = [
            sys.executable, "test.py",
            "--data_dir", args.data_dir,
            "--kfold", str(args.kfold),
            "--specific_fold", str(args.specific_fold),
            "--class_num", str(class_num),
            "--load_dir", f"./checkpoints/{expr_name}",
            "--test_target", test_target,
            "--name", expr_name,
            "--checkpoints_dir", "./checkpoints",
            "--gpu_ids", args.gpu_ids
        ]
        if no_classifier:
            cmd.append("--no_classifier")
        
        if class_num == 4:
            cmd.extend(["--dataset", "all", "--group", "all"])
        
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
