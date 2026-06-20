import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap

def plot_umap_visuals():
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
    
    networks = ['best_2c_net', 'best_3c_net', 'best_4c_net']
    folds = range(1, 6)
    
    color_palette = {'CN': '#66c2a5', 'sMCI': '#fc8d62', 'pMCI': '#8da0cb', 'AD': '#e78ac3'}
    class_order = ['CN', 'sMCI', 'pMCI', 'AD']

    for net in networks:
        for fold in folds:
            fig, axes = plt.subplots(1, 5, figsize=(30, 6))
            fig.suptitle(f'UMAP Projection - Fold {fold} ({net})', fontsize=16)
            
            for ax, (var_key, var_name) in zip(axes, variants.items()):
                csv_path = os.path.join(data_dir, f"{var_key}_{net}_fold{fold}.csv")
                if not os.path.exists(csv_path):
                    ax.set_title(f"{var_name} (Data Missing)")
                    continue
                
                df = pd.read_csv(csv_path)
                feature_cols = [c for c in df.columns if c.startswith('feature_')]
                X = df[feature_cols].values
                y = df['True Label'].values
                
                # Apply UMAP
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
                X_umap = reducer.fit_transform(X)
                
                df_plot = pd.DataFrame({'UMAP_1': X_umap[:, 0], 'UMAP_2': X_umap[:, 1], 'True Label': y})
                
                sns.scatterplot(data=df_plot, x='UMAP_1', y='UMAP_2', hue='True Label', 
                                palette=color_palette, hue_order=class_order, 
                                ax=ax, s=40, alpha=0.8, edgecolor='w')
                
                ax.set_title(var_name, fontsize=14)
                
                # Cleanup legend
                if ax != axes[-1]:
                    ax.get_legend().remove()
                else:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout(rect=[0, 0, 0.9, 0.95])
            out_path = os.path.join(out_dir, f"UMAP_{net}_fold{fold}.png")
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {out_path}")

if __name__ == "__main__":
    plot_umap_visuals()
