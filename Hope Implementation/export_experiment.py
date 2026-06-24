"""
Export experiment results for a given variant into a ZIP archive.

Usage:
    python export_experiment.py --variant loss_full_3c
    python export_experiment.py --variant loss_full_3c --checkpoints_dir ./checkpoints --out_dir ./analysis_output
"""
import os
import glob
import zipfile
import argparse
from utils.config_loader import load_config, get_variant_params


def collect_files(pattern, base_dir="."):
    """Collect files matching a glob pattern relative to base_dir."""
    full_pattern = os.path.join(base_dir, pattern)
    return glob.glob(full_pattern, recursive=True)


def build_experiment_name(params):
    """Build the experiment name string from variant params (matching ablation_loss.py convention)."""
    loss_type = params.get('loss_type', params.get('target_loss', 'full'))
    class_num = params.get('num_classes', 3)
    m = params.get('m', 0.9)
    triplet_margin = params.get('triplet_margin', 0.3)

    name_suffix = "_4class" if class_num == 4 else ""
    ema_suffix = f"_ema{m}" if m != 0.9 else ""
    margin_suffix = f"_margin{triplet_margin}" if triplet_margin != 0.3 else ""

    return f"ablation_loss_{loss_type}{name_suffix}{ema_suffix}{margin_suffix}"


def main():
    parser = argparse.ArgumentParser(description="Export experiment results into a ZIP archive")
    parser.add_argument('--variant', type=str, required=True, help='Variant key from pipeline_config.json')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='Path to checkpoints directory')
    parser.add_argument('--out_dir', type=str, default='./analysis_output', help='Base output directory')
    parser.add_argument('--config', type=str, default=None, help='Path to pipeline_config.json')
    parser.add_argument('--kfold', type=int, default=5, help='Number of folds')
    args = parser.parse_args()

    config = load_config(args.config)
    params = get_variant_params(args.variant, config)
    experiment_name = build_experiment_name(params)

    loss_type = params.get('loss_type', params.get('target_loss', 'full'))
    class_num = params.get('num_classes', 3)
    m = params.get('m', 0.9)
    triplet_margin = params.get('triplet_margin', 0.3)
    name_suffix = "_4class" if class_num == 4 else ""
    ema_suffix = f"_ema{m}" if m != 0.9 else ""
    margin_suffix = f"_margin{triplet_margin}" if triplet_margin != 0.3 else ""
    variant_file_prefix = f"{loss_type}{name_suffix}{ema_suffix}{margin_suffix}"

    zip_name = f"{args.variant}_results.zip"
    zip_path = os.path.join(args.out_dir, zip_name)
    os.makedirs(args.out_dir, exist_ok=True)

    files_added = 0

    print(f"Exporting experiment: {args.variant}")
    print(f"  Experiment name: {experiment_name}")
    print(f"  Checkpoints dir: {args.checkpoints_dir}")
    print(f"  Output ZIP: {zip_path}")
    print()

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. Extracted features CSVs
        features_dir = os.path.join(args.out_dir, "extracted_features")
        if os.path.isdir(features_dir):
            for test_target in ['2c', '3c', '4c']:
                for fold in range(1, args.kfold + 1):
                    csv_name = f"{variant_file_prefix}_best_{test_target}_net_fold{fold}.csv"
                    csv_path = os.path.join(features_dir, csv_name)
                    if os.path.isfile(csv_path):
                        arcname = f"extracted_features/{csv_name}"
                        zf.write(csv_path, arcname)
                        files_added += 1
                        print(f"  + {arcname}")

        # 2. Prototype JSON files
        # Search in checkpoints fold dirs and analysis output for matching JSONs
        for fold in range(1, args.kfold + 1):
            fold_dir = os.path.join(args.checkpoints_dir, f"{experiment_name}_fold{fold}")
            if os.path.isdir(fold_dir):
                for json_file in glob.glob(os.path.join(fold_dir, "*.json")):
                    arcname = f"prototypes/{os.path.basename(json_file)}"
                    zf.write(json_file, arcname)
                    files_added += 1
                    print(f"  + {arcname}")

        # Also search in analysis_output for prototype JSONs matching variant
        prototype_patterns = [
            os.path.join(args.out_dir, f"*{variant_file_prefix}*.json"),
            os.path.join(args.out_dir, "prototypes", f"*{variant_file_prefix}*.json"),
        ]
        for pattern in prototype_patterns:
            for json_file in glob.glob(pattern):
                arcname = f"prototypes/{os.path.basename(json_file)}"
                # Avoid duplicates
                if arcname not in [info.filename for info in zf.infolist()]:
                    zf.write(json_file, arcname)
                    files_added += 1
                    print(f"  + {arcname}")

        # 3. Test logs (PDFs from checkpoint fold dirs)
        for fold in range(1, args.kfold + 1):
            fold_dir = os.path.join(args.checkpoints_dir, f"{experiment_name}_fold{fold}")
            if os.path.isdir(fold_dir):
                for pdf_file in glob.glob(os.path.join(fold_dir, "*.pdf")):
                    arcname = f"test_logs/{experiment_name}_fold{fold}/{os.path.basename(pdf_file)}"
                    zf.write(pdf_file, arcname)
                    files_added += 1
                    print(f"  + {arcname}")

    if files_added == 0:
        print(f"\nWARNING: No files found for variant '{args.variant}' (experiment: {experiment_name}).")
        print("Make sure training, testing, and feature extraction have been run first.")
        # Remove empty zip
        os.remove(zip_path)
    else:
        print(f"\nExported {files_added} files -> {zip_path}")


if __name__ == '__main__':
    main()
