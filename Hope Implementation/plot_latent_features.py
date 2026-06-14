import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from glob import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str, default='./analysis_output/extracted_features', help='path to the downloaded CSV files')
    parser.add_argument('--out_dir', type=str, default='./analysis_output/latent_plots', help='where to save the final plots')
    opt = parser.parse_args()

    os.makedirs(opt.out_dir, exist_ok=True)

    LOSS_VARIANTS = ['ce', 'ins2ins', 'ins2cls', 'full', 'exclude_ins2ins', 'exclude_ins2cls', 'exp_triplet_ins2cls']
    CHECKPOINTS = ['best_2c_net', 'best_3c_net', 'best_4c_net']

    for variant in LOSS_VARIANTS:
        for ckpt in CHECKPOINTS:
            # Find all CSVs for this specific variant and checkpoint across all folds
            search_pattern = os.path.join(opt.csv_dir, f"{variant}_{ckpt}_fold*.csv")
            csv_files = glob(search_pattern)
            
            if not csv_files:
                continue
                
            print(f"Plotting: {variant} | {ckpt} | Found {len(csv_files)} folds")
            
            # Combine all folds
            df_list = [pd.read_csv(f) for f in csv_files]
            combined_df = pd.concat(df_list, ignore_index=True)
            
            # Plotting
            plt.figure(figsize=(12, 7))
            sns.kdeplot(data=combined_df, x='Severity', hue='True Label', fill=True, common_norm=False, 
                        palette={'CN':'blue', 'sMCI':'green', 'pMCI':'orange', 'AD':'red'},
                        alpha=0.4, linewidth=2.5)
            
            plt.title(f"Latent Feature Distribution\nVariant: {variant}  |  Checkpoint: {ckpt}.pth\n(Aggregated {len(csv_files)} Folds | {len(combined_df)} Patients)", fontsize=16, pad=15)
            plt.xlabel("Latent 1D Severity: (Sim to AD Prototype) - (Sim to CN Prototype)", fontsize=14)
            plt.ylabel("Density", fontsize=14)
            
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Neutral Boundary')
            plt.legend(title='Clinical Diagnosis', loc='upper left', fontsize=12, title_fontsize=12)
            
            plt.tight_layout()
            
            plot_filename = f"latent_kde_{variant}_{ckpt}.png"
            plot_path = os.path.join(opt.out_dir, plot_filename)
            plt.savefig(plot_path, dpi=200)
            plt.close()
            
            print(f"  -> Saved {plot_path}")

if __name__ == '__main__':
    main()
