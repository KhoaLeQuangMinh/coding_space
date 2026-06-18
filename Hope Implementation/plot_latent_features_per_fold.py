import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str, default='/Users/khoale/Downloads/analysis_output_tSNE/extracted_features', help='path to the downloaded CSV files')
    parser.add_argument('--out_dir', type=str, default='/Users/khoale/Downloads/analysis_output_tSNE/plot_kde_folds', help='where to save the final plots')
    opt = parser.parse_args()

    os.makedirs(opt.out_dir, exist_ok=True)

    LOSS_VARIANTS = [
        'ce', 
        'ins2ins', 
        'ins2cls', 
        'full',
        'exclude_ins2ins',
        'exclude_ins2cls',
        'exp_triplet_ins2cls',
        'triplet_only',
        'hierarchical_triplet_only',
        'exp_hierarchical_triplet_ins2cls',
        'full_4class',
        'exp_triplet_ins2cls_4class',
        'hierarchical_triplet_only_4class',
        'qwk_hierarchical_triplet_4class',
        'exp_3pole_local',
        'exp_3pole_global'
    ]
    
    CHECKPOINTS = ['best_2c_net', 'best_3c_net', 'best_4c_net']
    FOLDS = [1, 2, 3, 4, 5]

    for variant in LOSS_VARIANTS:
        for ckpt in CHECKPOINTS:
            for fold in FOLDS:
                csv_path = os.path.join(opt.csv_dir, f"{variant}_{ckpt}_fold{fold}.csv")
                
                if not os.path.exists(csv_path):
                    continue
                    
                print(f"Plotting: {variant} | {ckpt} | Fold {fold}")
                
                df = pd.read_csv(csv_path)
                
                # Check for mode collapse (zero variance) which causes blank KDE plots
                # If a class or the entire dataset has 0 variance, seaborn silently fails.
                for label in df['True Label'].unique():
                    mask = df['True Label'] == label
                    if df.loc[mask, 'Severity'].std() == 0.0 or pd.isna(df.loc[mask, 'Severity'].std()):
                        # Add a tiny amount of noise to allow KDE to draw a massive spike
                        df.loc[mask, 'Severity'] += np.random.normal(0, 1e-4, mask.sum())
                
                # Plotting
                plt.figure(figsize=(10, 6))
                sns.kdeplot(data=df, x='Severity', hue='True Label', fill=True, common_norm=False, 
                            palette={'CN':'blue', 'sMCI':'green', 'pMCI':'orange', 'AD':'red'},
                            alpha=0.4, linewidth=2.5)
                
                plt.title(f"Latent KDE | {variant} | {ckpt} | Fold {fold}", fontsize=14, pad=10)
                plt.xlabel("Latent 1D Severity: (Sim to AD) - (Sim to CN)", fontsize=12)
                plt.ylabel("Density", fontsize=12)
                
                plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Neutral Boundary')
                plt.legend(title='Clinical Diagnosis', loc='upper left', fontsize=10, title_fontsize=10)
                
                plt.tight_layout()
                
                plot_filename = f"latent_kde_{variant}_{ckpt}_fold{fold}.png"
                plot_path = os.path.join(opt.out_dir, plot_filename)
                plt.savefig(plot_path, dpi=200, bbox_inches='tight')
                plt.close()
                
                print(f"  -> Saved {plot_path}")

if __name__ == '__main__':
    main()
