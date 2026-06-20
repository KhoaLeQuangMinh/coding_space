import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_euclidean_scatter():
    data_dir = '/Users/khoale/Downloads/analysis_output_tSNE/extracted_features/'
    out_dir = '/Users/khoale/Downloads/ablation_result/proposed/'
    os.makedirs(out_dir, exist_ok=True)
    
    variants = {
        'ce': 'CE (Baseline)',
        'full': 'HOPE (Full)',
        'triplet_only': 'Proposed (Triplet Only)',
        'exp_3pole_local': '3-Pole Triplet (Local)',
        'exp_3pole_global': '3-Pole Triplet (Global)'
    }
    
    # User specifically requested this only for the 2 class task (sMCI vs pMCI)
    net = 'best_2c_net'
    folds = range(1, 6)
    
    color_palette = {'sMCI': '#fc8d62', 'pMCI': '#8da0cb'}

    for fold in folds:
        fig, axes = plt.subplots(1, 5, figsize=(30, 6))
        fig.suptitle(f'Distance to CN vs AD Prototype (2-Class: sMCI vs pMCI) - Fold {fold}', fontsize=16)
        
        for ax, (var_key, var_name) in zip(axes, variants.items()):
            csv_path = os.path.join(data_dir, f"{var_key}_{net}_fold{fold}.csv")
            if not os.path.exists(csv_path):
                ax.set_title(f"{var_name} (Data Missing)")
                continue
            
            df = pd.read_csv(csv_path)
            
            # Filter for only sMCI and pMCI
            df = df[df['True Label'].isin(['sMCI', 'pMCI'])]
            
            # To calculate distance to CN and AD prototypes, we need to extract them first.
            # We assume the prototype is the mean of all training samples, but since we only have test samples here,
            # we will re-read the full CSV and calculate the mean of CN and AD from the test set as an approximation.
            # Wait, the test set might not have many CN or AD. 
            # In your pipeline, the features were normalized in some cases, but Euclidean distance can be calculated directly.
            
            df_full = pd.read_csv(csv_path)
            feature_cols = [c for c in df.columns if c.startswith('feature_')]
            
            cn_feats = df_full[df_full['True Label'] == 'CN'][feature_cols].values
            ad_feats = df_full[df_full['True Label'] == 'AD'][feature_cols].values
            
            if len(cn_feats) > 0 and len(ad_feats) > 0:
                proto_cn = np.mean(cn_feats, axis=0)
                proto_ad = np.mean(ad_feats, axis=0)
            else:
                print(f"Warning: Missing CN or AD in fold {fold} for {var_key}. Skipping scatter.")
                continue
            
            mci_feats = df[feature_cols].values
            
            dist_to_cn = np.linalg.norm(mci_feats - proto_cn, axis=1)
            dist_to_ad = np.linalg.norm(mci_feats - proto_ad, axis=1)
            
            df_plot = pd.DataFrame({
                'Distance to AD Prototype (X)': dist_to_ad,
                'Distance to CN Prototype (Y)': dist_to_cn,
                'True Label': df['True Label'].values
            })
            
            sns.scatterplot(data=df_plot, x='Distance to AD Prototype (X)', y='Distance to CN Prototype (Y)', 
                            hue='True Label', palette=color_palette, ax=ax, s=50, alpha=0.8, edgecolor='w')
            
            # Draw y=x reference line
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  
                np.max([ax.get_xlim(), ax.get_ylim()]),  
            ]
            ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0)
            
            ax.set_title(var_name, fontsize=14)
            
            if ax != axes[-1]:
                ax.get_legend().remove()
            else:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        out_path = os.path.join(out_dir, f"Distance_Scatter_{net}_fold{fold}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    plot_euclidean_scatter()
