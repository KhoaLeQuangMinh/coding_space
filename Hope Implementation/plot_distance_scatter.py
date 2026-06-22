import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_euclidean_scatter():
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
        'hierarchical_triplet_only': 'CE + Hierarchical Triplet',
        'exp_hierarchical_triplet_ins2cls': 'Hierarchical Triplet Ins2Cls',
        'full_4class': 'HOPE 4-Class (No EMA)',
        'exp_triplet_ins2cls_4class': 'Triplet Ins2Cls 4-Class',
        'hierarchical_triplet_only_4class': 'Hierarchical Triplet 4-Class',
        'qwk_hierarchical_triplet_4class': 'QWK Hierarchical Triplet 4c',
        'exp_3pole_local': '3-Pole Triplet (Local)',
        'exp_3pole_global': '3-Pole Triplet (Global)',
        '3pole_local_only': '3-Pole (Local) Only',
        '3pole_global_only': '3-Pole (Global) Only'
    }
    
    net = 'best_2c_net'
    folds = range(1, 6)
    color_palette = {'sMCI': '#fc8d62', 'pMCI': '#8da0cb'}

    # Subdirectory for individual plots
    scatter_ind_dir = os.path.join(out_dir, 'distance_scatter_individual')
    os.makedirs(scatter_ind_dir, exist_ok=True)
    
    comp_dir = os.path.join(out_dir, 'comparative')
    os.makedirs(comp_dir, exist_ok=True)

    for fold in folds:
        fig, axes = plt.subplots(3, 6, figsize=(28, 14))
        axes_flat = axes.flatten()
        fig.suptitle(f'Distance to CN vs AD Prototype (2-Class: sMCI vs pMCI) - Fold {fold}', fontsize=20, y=0.98, fontweight='bold')
        
        handles = None
        labels = None
        
        for idx, (var_key, var_name) in enumerate(variants.items()):
            ax = axes_flat[idx]
            csv_path = os.path.join(data_dir, f"{var_key}_{net}_fold{fold}.csv")
            if not os.path.exists(csv_path):
                ax.set_title(f"{var_name} (Data Missing)", fontsize=11, color='red')
                ax.axis('off')
                continue
            
            df_full = pd.read_csv(csv_path)
            feature_cols = [c for c in df_full.columns if c.startswith('feature_')]
            
            cn_feats = df_full[df_full['True Label'] == 'CN'][feature_cols].values
            ad_feats = df_full[df_full['True Label'] == 'AD'][feature_cols].values
            
            if len(cn_feats) > 0 and len(ad_feats) > 0:
                proto_cn = np.mean(cn_feats, axis=0)
                proto_ad = np.mean(ad_feats, axis=0)
            else:
                ax.set_title(f"{var_name} (Proto Missing)", fontsize=11, color='orange')
                ax.axis('off')
                continue
            
            # Filter for only sMCI and pMCI
            df_mci = df_full[df_full['True Label'].isin(['sMCI', 'pMCI'])]
            if len(df_mci) == 0:
                ax.set_title(f"{var_name} (No MCI)", fontsize=11, color='orange')
                ax.axis('off')
                continue
                
            mci_feats = df_mci[feature_cols].values
            dist_to_cn = np.linalg.norm(mci_feats - proto_cn, axis=1)
            dist_to_ad = np.linalg.norm(mci_feats - proto_ad, axis=1)
            
            df_plot = pd.DataFrame({
                'Distance to AD Prototype': dist_to_ad,
                'Distance to CN Prototype': dist_to_cn,
                'True Label': df_mci['True Label'].values
            })
            
            # Plot for comparative grid
            sns.scatterplot(data=df_plot, x='Distance to AD Prototype', y='Distance to CN Prototype', 
                            hue='True Label', palette=color_palette, ax=ax, s=40, alpha=0.8, edgecolor='w', linewidth=0.3)
            
            # Draw y=x reference line
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  
                np.max([ax.get_xlim(), ax.get_ylim()]),  
            ]
            ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0)
            
            ax.set_title(var_name, fontsize=12, fontweight='bold')
            ax.set_xlabel("Distance to AD Prototype", fontsize=9)
            ax.set_ylabel("Distance to CN Prototype", fontsize=9)
            ax.tick_params(axis='both', which='major', labelsize=9)
            
            # Extract legend handles and then remove local legend
            if ax.legend_:
                if handles is None:
                    handles, labels = ax.get_legend_handles_labels()
                ax.legend_.remove()
            
            # Save individual scatter plot
            fig_ind, ax_ind = plt.subplots(figsize=(7, 6))
            sns.scatterplot(data=df_plot, x='Distance to AD Prototype', y='Distance to CN Prototype', 
                            hue='True Label', palette=color_palette, ax=ax_ind, s=50, alpha=0.8, edgecolor='w', linewidth=0.3)
            lims_ind = [
                np.min([ax_ind.get_xlim(), ax_ind.get_ylim()]),  
                np.max([ax_ind.get_xlim(), ax_ind.get_ylim()]),  
            ]
            ax_ind.plot(lims_ind, lims_ind, 'k--', alpha=0.3, zorder=0)
            ax_ind.set_title(f"{var_name}\nDistance to CN vs AD Prototype (Fold {fold})", fontsize=12, fontweight='bold')
            ax_ind.set_xlabel("Distance to AD Prototype", fontsize=10)
            ax_ind.set_ylabel("Distance to CN Prototype", fontsize=10)
            ax_ind.legend(title="Clinical Diagnosis", loc='upper left')
            fig_ind.savefig(os.path.join(scatter_ind_dir, f"distance_scatter_{var_key}_fold{fold}.png"), dpi=150, bbox_inches='tight')
            plt.close(fig_ind)

        # Hide any unused axes
        for idx in range(len(variants), len(axes_flat)):
            axes_flat[idx].axis('off')
            
        # Add unified legend
        if handles is not None:
            fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.95), 
                       fontsize=14, title='Clinical Diagnosis', title_fontsize=15)

        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        out_path = os.path.join(comp_dir, f"Distance_Scatter_{net}_fold{fold}.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    plot_euclidean_scatter()
