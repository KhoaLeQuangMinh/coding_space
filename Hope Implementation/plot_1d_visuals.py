import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

def plot_1d_projections():
    data_dir = '/Users/khoale/Downloads/analysis_output_tSNE/extracted_features/'
    out_dir = '/Users/khoale/Downloads/ablation_result/proposed/'
    os.makedirs(out_dir, exist_ok=True)
    
    # We restrict to CE, HOPE (full), Proposed, and 3-Pole variants with margins
    variants = {
        'ce': 'CE (Baseline)',
        'full': 'HOPE (Full)',
        'triplet_only_margin0.3': 'Proposed (Triplet Only) (Margin 0.3)',
        'triplet_only_margin3.0': 'Proposed (Triplet Only) (Margin 3.0)',
        'triplet_only_ema0.5_margin0.0': 'Proposed (Triplet Only) (EMA 0.5, Margin 0.0)',
        '3pole_local_only': '3-Pole Local (Margin 0.3)',
        '3pole_local_only_margin0.0': '3-Pole Local (Margin 0.0)',
        '3pole_global_only': '3-Pole Global (Margin 0.3)',
        '3pole_global_only_margin0.0': '3-Pole Global (Margin 0.0)'
    }
    
    networks = ['best_2c_net', 'best_3c_net', 'best_4c_net']
    folds = range(1, 6)
    
    color_palette = {'CN': '#66c2a5', 'sMCI': '#fc8d62', 'pMCI': '#8da0cb', 'AD': '#e78ac3'}
    class_order = ['CN', 'sMCI', 'pMCI', 'AD']

    for net in networks:
        for fold in folds:
            fig, axes = plt.subplots(1, 9, figsize=(45, 5))
            fig.suptitle(f'1D PCA Projection - Fold {fold} ({net})', fontsize=16)
            
            for ax, (var_key, var_name) in zip(axes, variants.items()):
                csv_path = os.path.join(data_dir, f"{var_key}_{net}_fold{fold}.csv")
                if not os.path.exists(csv_path):
                    ax.set_title(f"{var_name}\n(Data Missing)")
                    continue
                
                df = pd.read_csv(csv_path)
                feature_cols = [c for c in df.columns if c.startswith('feature_')]
                X = df[feature_cols].values
                y = df['True Label'].values
                
                # Apply 1D PCA
                pca = PCA(n_components=1)
                X_1d = pca.fit_transform(X).flatten()
                
                df_plot = pd.DataFrame({'PCA_1D': X_1d, 'True Label': y, 'Y_jitter': np.random.normal(0, 0.05, size=len(y))})
                
                sns.scatterplot(data=df_plot, x='PCA_1D', y='Y_jitter', hue='True Label', 
                                palette=color_palette, hue_order=class_order, 
                                ax=ax, s=30, alpha=0.7, edgecolor='w')
                
                ax.set_title(var_name, fontsize=12)
                ax.set_yticks([])  # Hide Y axis since it's just jitter
                ax.set_ylabel('')
                ax.axhline(0, color='black', linewidth=0.8, alpha=0.5, zorder=0)
                
                # Cleanup legend
                if ax != axes[-1]:
                    ax.get_legend().remove()
                else:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout(rect=[0, 0, 0.92, 0.95])
            out_path = os.path.join(out_dir, f"1D_PCA_{net}_fold{fold}.png")
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {out_path}")

if __name__ == "__main__":
    plot_1d_projections()
