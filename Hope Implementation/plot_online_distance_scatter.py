import os
import json
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_key(key):
    """
    Parses a JSON key like 'ablation_loss_3pole_global_only_fold5/best_2c_net.pth'
    Returns (variant, fold, model_name)
    """
    parts = key.split('/')
    if len(parts) != 2:
        return None
    
    dir_name, model_file = parts[0], parts[1]
    
    # Extract fold from folder name (e.g., _fold5)
    fold_match = re.search(r'_fold(\d+)$', dir_name)
    if not fold_match:
        return None
    fold = int(fold_match.group(1))
    
    # Strip prefix 'ablation_loss_' and suffix '_foldX' to get variant name
    variant = dir_name
    if variant.startswith('ablation_loss_'):
        variant = variant[len('ablation_loss_'):]
    variant = re.sub(r'_fold\d+$', '', variant)
    
    # Strip '.pth' from model file to get model name (e.g. best_2c_net)
    model_name = os.path.splitext(model_file)[0]
    
    return variant, fold, model_name

def main():
    parser = argparse.ArgumentParser(description="Plot CN vs AD prototype distances using online prototypes from JSON.")
    parser.add_argument('--json_dir', type=str, default='/Users/khoale/Downloads', help='Directory containing the downloaded JSON files')
    parser.add_argument('--csv_dir', type=str, default='/Users/khoale/Downloads/analysis_output_tSNE/extracted_features', help='Directory containing patient feature CSVs')
    parser.add_argument('--out_dir', type=str, default='/Users/khoale/Downloads/plot_display/plots/online_distance_scatter', help='Directory to save output plots')
    parser.add_argument('--model', type=str, default='best_2c_net', help='Model checkpoint to plot (best_2c_net, best_3c_net, or best_4c_net)')
    parser.add_argument('--normalized', type=bool, default=True, help='L2 normalize features and prototypes (matching cosine space)')
    parser.add_argument('--metric', type=str, default='distance', choices=['distance', 'cosine'], help='Metric to plot (distance or cosine)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Find all prototype JSON files in json_dir
    json_files = sorted([f for f in os.listdir(args.json_dir) if f.endswith('.json') and ('3 poles' in f or 'Triplets' in f or 'Triplet' in f)])
    if not json_files:
        # Fallback to look at all JSON files if name filtering is too strict
        json_files = sorted([f for f in os.listdir(args.json_dir) if f.endswith('.json') and f != 'settings.json'])
        
    if not json_files:
        print(f"Error: No JSON files found in {args.json_dir}")
        return

    print(f"Found {len(json_files)} prototype JSON files to process:")
    for jf in json_files:
        print(f"  - {jf}")

    color_palette = {'sMCI': '#fc8d62', 'pMCI': '#8da0cb'}

    for jf in json_files:
        json_path = os.path.join(args.json_dir, jf)
        try:
            with open(json_path, 'r') as f:
                proto_data = json.load(f)
        except Exception as e:
            print(f"Failed to read {jf}: {e}")
            continue

        # Extract name of the experiment from the JSON filename
        exp_name = os.path.splitext(jf)[0]
        print(f"\nProcessing experiment: {exp_name}...")

        # We will create one comparative grid for all 5 folds for this experiment
        fig, axes = plt.subplots(1, 5, figsize=(25, 5.5), sharey=True, sharex=True)
        metric_title = "Cosine Similarity" if args.metric == 'cosine' else "Distance"
        fig.suptitle(f"Online Prototype {metric_title} Scatter (sMCI vs pMCI) - {exp_name} ({args.model})", fontsize=16, fontweight='bold', y=1.02)

        plot_count = 0

        # Sort keys so folds are processed in order (Fold 1 to Fold 5)
        sorted_keys = sorted(list(proto_data.keys()))

        for key in sorted_keys:
            # We only plot the specific model of interest (e.g. best_2c_net)
            if args.model not in key:
                continue

            parsed = parse_key(key)
            if not parsed:
                continue

            variant, fold, model_name = parsed
            
            # Normalise margin_X.X to marginX.X to match the CSV filename format
            variant_normalized = variant.replace("margin_", "margin") if "margin_" in variant else variant
            
            csv_path = os.path.join(args.csv_dir, f"{variant_normalized}_{model_name}_fold{fold}.csv")
            if not os.path.exists(csv_path):
                # Fallback to the raw variant name
                csv_path = os.path.join(args.csv_dir, f"{variant}_{model_name}_fold{fold}.csv")

            if not os.path.exists(csv_path):
                print(f"  [Warning] Missing CSV for fold {fold}: {csv_path}")
                continue

            # Load features
            df = pd.read_csv(csv_path)
            feature_cols = [c for c in df.columns if c.startswith('feature_')]
            if not feature_cols:
                continue

            # Get online prototypes from JSON
            protos = np.array(proto_data[key])
            num_classes = protos.shape[0]

            # CN is always index 0, AD is always the last index
            proto_cn = protos[0]
            proto_ad = protos[-1]

            # Filter patient data for sMCI and pMCI
            df_mci = df[df['True Label'].isin(['sMCI', 'pMCI'])]
            if len(df_mci) == 0:
                continue

            mci_feats = df_mci[feature_cols].values

            # Perform computations based on metric
            if args.metric == 'cosine':
                # Normalizing prototypes (cosine similarity always requires L2 normalization)
                proto_cn_norm = proto_cn / np.linalg.norm(proto_cn)
                proto_ad_norm = proto_ad / np.linalg.norm(proto_ad)
                
                # Normalizing patient features
                mci_feats_norm = mci_feats / np.linalg.norm(mci_feats, axis=1, keepdims=True)

                # Compute Cosine Similarity
                sim_to_cn = np.dot(mci_feats_norm, proto_cn_norm)
                sim_to_ad = np.dot(mci_feats_norm, proto_ad_norm)
                
                x_val = sim_to_ad
                y_val = sim_to_cn
                x_label = "Cosine Similarity to AD Prototype"
                y_label = "Cosine Similarity to CN Prototype"
            else:
                # Perform L2 normalization if requested
                if args.normalized:
                    # Normalizing prototypes
                    proto_cn_norm = proto_cn / np.linalg.norm(proto_cn)
                    proto_ad_norm = proto_ad / np.linalg.norm(proto_ad)
                    
                    # Normalizing patient features
                    mci_feats_norm = mci_feats / np.linalg.norm(mci_feats, axis=1, keepdims=True)

                    # Compute Euclidean distance in the L2-normalized space
                    dist_to_cn = np.linalg.norm(mci_feats_norm - proto_cn_norm, axis=1)
                    dist_to_ad = np.linalg.norm(mci_feats_norm - proto_ad_norm, axis=1)
                else:
                    # Compute raw Euclidean distance
                    dist_to_cn = np.linalg.norm(mci_feats - proto_cn, axis=1)
                    dist_to_ad = np.linalg.norm(mci_feats - proto_ad, axis=1)
                
                x_val = dist_to_ad
                y_val = dist_to_cn
                x_label = "Distance to AD Prototype"
                y_label = "Distance to CN Prototype"

            df_plot = pd.DataFrame({
                x_label: x_val,
                y_label: y_val,
                'True Label': df_mci['True Label'].values
            })

            # Plot on the corresponding fold subplot
            ax = axes[fold - 1]
            sns.scatterplot(data=df_plot, x=x_label, y=y_label, 
                            hue='True Label', palette=color_palette, ax=ax, s=45, alpha=0.8, edgecolor='w', linewidth=0.3)

            # Draw y=x diagonal reference line
            lims = [
                min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])
            ]
            ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0)
            
            ax.set_title(f"Fold {fold}", fontsize=13, fontweight='bold')
            ax.set_xlabel(x_label, fontsize=11)
            ax.set_ylabel(y_label, fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Hide individual legends to show a single unified one
            if ax.legend_:
                ax.legend_.remove()

            plot_count += 1

        if plot_count > 0:
            # Add a unified legend to the right of the subplots
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.99, 0.5), 
                       title='Diagnosis', title_fontsize=12, fontsize=11)
            
            plt.tight_layout(rect=[0, 0, 0.92, 0.95])
            if args.metric == 'cosine':
                suffix = "_cosine"
            else:
                suffix = "_normalized" if args.normalized else "_raw"
            out_path = os.path.join(args.out_dir, f"online_scatter_{exp_name}_{args.model}{suffix}.png")
            plt.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"  -> Generated: {out_path}")
        else:
            plt.close()
            print(f"  -> No matching checkpoints found for model {args.model}")

if __name__ == '__main__':
    main()
