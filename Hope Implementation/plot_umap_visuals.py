import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap

def plot_umap_visuals():
    data_dir = '/Users/khoale/Downloads/analysis_output_tSNE/extracted_features/'
    out_dir = '/Users/khoale/Downloads/plot_display/plots/'
    os.makedirs(out_dir, exist_ok=True)
    
    variants = {
        'ce': 'L_CE',
        'ins2ins': 'L_CE + L_Ins2Ins',
        'ins2cls': 'L_CE + L_Ins2Ins + L_Ins2Cls',
        'full': 'L_CE + L_Ins2Ins + L_Ins2Cls + L_Cls2Cls (HOPE)',
        'exclude_ins2ins': 'Exclude Ins2Ins',
        'exclude_ins2cls': 'Exclude Ins2Cls',
        'exp_triplet_ins2cls': 'Triplet Ins2Cls (Poles)',
        'triplet_only': 'CE + Triplet Only',
        'triplet_only_margin0.3': 'CE + Triplet Only (Margin 0.3)',
        'triplet_only_margin3.0': 'CE + Triplet Only (Margin 3.0)',
        'triplet_only_ema0.5_margin0.0': 'CE + Triplet Only (EMA 0.5, Margin 0.0)',
        'hierarchical_triplet_only': 'CE + Hierarchical Triplet',
        'exp_hierarchical_triplet_ins2cls': 'Hierarchical Triplet Ins2Cls',
        'full_4class': 'HOPE 4-Class (No EMA)',
        'exp_triplet_ins2cls_4class': 'Triplet Ins2Cls 4-Class',
        'hierarchical_triplet_only_4class': 'Hierarchical Triplet 4-Class',
        'qwk_hierarchical_triplet_4class': 'QWK Hierarchical Triplet 4c',
        'exp_3pole_local': '3-Pole Triplet (Local)',
        'exp_3pole_global': '3-Pole Triplet (Global)',
        '3pole_local_only': '3-Pole (Local) Only (Margin 0.3)',
        '3pole_global_only': '3-Pole (Global) Only (Margin 0.3)',
        '3pole_local_only_margin0.0': '3-Pole (Local) Only (Margin 0.0)',
        '3pole_global_only_margin0.0': '3-Pole (Global) Only (Margin 0.0)'
    }
    
    networks = ['best_2c_net', 'best_3c_net', 'best_4c_net']
    folds = range(1, 6)
    
    color_palette = {'CN': '#66c2a5', 'sMCI': '#fc8d62', 'pMCI': '#8da0cb', 'AD': '#e78ac3'}
    class_order = ['CN', 'sMCI', 'pMCI', 'AD']

    # Subdirectory for individual plots
    umap_ind_dir = os.path.join(out_dir, 'umap_individual')
    os.makedirs(umap_ind_dir, exist_ok=True)
    
    comp_dir = os.path.join(out_dir, 'comparative')
    os.makedirs(comp_dir, exist_ok=True)

    for net in networks:
        for fold in folds:
            fig, axes = plt.subplots(4, 6, figsize=(36, 20))
            axes_flat = axes.flatten()
            fig.suptitle(f'UMAP Projections for All 23 Loss Variants - Fold {fold} ({net})', fontsize=20, y=0.98, fontweight='bold')
            
            handles = None
            labels = None
            
            for idx, (var_key, var_name) in enumerate(variants.items()):
                ax = axes_flat[idx]
                csv_path = os.path.join(data_dir, f"{var_key}_{net}_fold{fold}.csv")
                if not os.path.exists(csv_path):
                    ax.set_title(f"{var_name} (Data Missing)", fontsize=11, color='red')
                    ax.axis('off')
                    continue
                
                df = pd.read_csv(csv_path)
                feature_cols = [c for c in df.columns if c.startswith('feature_')]
                X = df[feature_cols].values
                y = df['True Label'].values
                
                # Apply UMAP
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
                X_umap = reducer.fit_transform(X)
                
                df_plot = pd.DataFrame({'UMAP_1': X_umap[:, 0], 'UMAP_2': X_umap[:, 1], 'True Label': y})
                
                # Subplot for Grid
                sns.scatterplot(data=df_plot, x='UMAP_1', y='UMAP_2', hue='True Label', 
                                palette=color_palette, hue_order=class_order, 
                                ax=ax, s=35, alpha=0.8, edgecolor='w', linewidth=0.3)
                
                ax.set_title(var_name, fontsize=12, fontweight='bold')
                ax.set_xlabel("UMAP 1", fontsize=9)
                ax.set_ylabel("UMAP 2", fontsize=9)
                
                # Extract legend handles and then remove local legend
                if ax.legend_:
                    if handles is None:
                        handles, labels = ax.get_legend_handles_labels()
                    ax.legend_.remove()
                
                # Save individual UMAP plot
                fig_ind, ax_ind = plt.subplots(figsize=(7, 6))
                sns.scatterplot(data=df_plot, x='UMAP_1', y='UMAP_2', hue='True Label', 
                                palette=color_palette, hue_order=class_order, 
                                ax=ax_ind, s=50, alpha=0.8, edgecolor='w', linewidth=0.3)
                ax_ind.set_title(f"{var_name}\nUMAP Projection - Fold {fold} ({net})", fontsize=12, fontweight='bold')
                ax_ind.set_xlabel("UMAP 1", fontsize=10)
                ax_ind.set_ylabel("UMAP 2", fontsize=10)
                ax_ind.legend(title="Clinical Diagnosis", bbox_to_anchor=(1.05, 1), loc='upper left')
                fig_ind.savefig(os.path.join(umap_ind_dir, f"umap_{var_key}_{net}_fold{fold}.png"), dpi=150, bbox_inches='tight')
                plt.close(fig_ind)
            
            # Hide any unused axes
            for idx in range(len(variants), len(axes_flat)):
                axes_flat[idx].axis('off')
                
            # Add a single unified global legend
            if handles is not None:
                fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.95), 
                           fontsize=14, title='Clinical Diagnosis', title_fontsize=15)

            plt.tight_layout(rect=[0, 0, 0.9, 0.95])
            out_path = os.path.join(comp_dir, f"UMAP_{net}_fold{fold}.png")
            plt.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved: {out_path}")

if __name__ == "__main__":
    plot_umap_visuals()
