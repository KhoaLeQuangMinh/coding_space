import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_quantitative_distances():
    data_dir = "/Users/khoale/Downloads/analysis_output_tSNE/extracted_features/"
    out_dir = "/Users/khoale/Downloads/plot_display/plots/"
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

    features_cols = [f'feature_{i}' for i in range(128)]
    labels_order = ['CN', 'sMCI', 'pMCI', 'AD']

    # Subdirectory for individual boxplots
    box_ind_dir = os.path.join(out_dir, 'distance_box_individual')
    os.makedirs(box_ind_dir, exist_ok=True)
    
    comp_dir = os.path.join(out_dir, 'comparative')
    os.makedirs(comp_dir, exist_ok=True)

    fig, axes = plt.subplots(4, 6, figsize=(30, 20))
    axes_flat = axes.flatten()
    fig.suptitle('Quantitative Euclidean Distance to AD Prototype for All 23 Loss Variants (best_2c_net)', fontsize=20, y=0.98, fontweight='bold')

    for idx, (var_key, var_name) in enumerate(variants.items()):
        ax = axes_flat[idx]
        all_distances = []
        
        for fold in range(1, 6):
            csv_path = os.path.join(data_dir, f"{var_key}_best_2c_net_fold{fold}.csv")
            if not os.path.exists(csv_path):
                continue
            
            df = pd.read_csv(csv_path)
            
            # Calculate empirical AD centroid for this fold
            ad_data = df[df['True Label'] == 'AD']
            if len(ad_data) == 0:
                continue
                
            ad_centroid = ad_data[features_cols].mean(axis=0).values
            
            # Calculate Euclidean distance to AD centroid for all patients in this fold
            feats = df[features_cols].values
            distances = np.linalg.norm(feats - ad_centroid, axis=1)
            
            for dist, lbl in zip(distances, df['True Label']):
                all_distances.append({'Distance': dist, 'Label': lbl, 'Fold': fold})
                
        df_dist = pd.DataFrame(all_distances)
        
        if len(df_dist) > 0:
            # Subplot for Grid
            sns.boxplot(data=df_dist, x='Label', y='Distance', order=labels_order, ax=ax, palette='Set2')
            sns.stripplot(data=df_dist, x='Label', y='Distance', order=labels_order, ax=ax, color='black', alpha=0.3, size=2.5)
            ax.set_title(var_name, fontsize=12, fontweight='bold')
            ax.set_ylabel("Distance to AD Prototype", fontsize=9)
            ax.set_xlabel("Clinical Diagnosis", fontsize=9)
            ax.tick_params(axis='both', which='major', labelsize=9)
            
            # Save individual boxplot
            fig_ind, ax_ind = plt.subplots(figsize=(7, 6))
            sns.boxplot(data=df_dist, x='Label', y='Distance', order=labels_order, ax=ax_ind, palette='Set2')
            sns.stripplot(data=df_dist, x='Label', y='Distance', order=labels_order, ax=ax_ind, color='black', alpha=0.3, size=3)
            ax_ind.set_title(f"{var_name}\nEuclidean Distance to AD Prototype", fontsize=12, fontweight='bold')
            ax_ind.set_ylabel("Distance to AD Prototype", fontsize=10)
            ax_ind.set_xlabel("Clinical Diagnosis", fontsize=10)
            fig_ind.savefig(os.path.join(box_ind_dir, f"distance_box_{var_key}.png"), dpi=150, bbox_inches='tight')
            plt.close(fig_ind)
        else:
            ax.set_title(f"{var_name} (Data Missing)", fontsize=11, color='red')
            ax.axis('off')

    # Hide any unused axes
    for idx in range(len(variants), len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(comp_dir, "quantitative_distance_boxplots.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    plot_quantitative_distances()
