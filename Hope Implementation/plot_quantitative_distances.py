import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = "/Users/khoale/Downloads/analysis_output_tSNE/extracted_features/"
out_dir = "/Users/khoale/Downloads/ablation_result/statistical_proof/"

variants = {
    'ce': 'CE (Baseline)',
    'full': 'HOPE (Full)',
    'triplet_only': 'Proposed (Triplet Only)',
    'exp_3pole_local': '3-Pole Triplet (Local)',
    'exp_3pole_global': '3-Pole Triplet (Global)'
}

features_cols = [f'feature_{i}' for i in range(128)]
labels_order = ['CN', 'sMCI', 'pMCI', 'AD']

fig, axes = plt.subplots(1, 5, figsize=(30, 6))

for ax, (var_key, var_name) in zip(axes, variants.items()):
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
        sns.boxplot(data=df_dist, x='Label', y='Distance', order=labels_order, ax=ax, palette='Set2')
        sns.stripplot(data=df_dist, x='Label', y='Distance', order=labels_order, ax=ax, color='black', alpha=0.3, size=3)
        
    ax.set_title(var_name, fontsize=14, fontweight='bold')
    ax.set_ylabel("Euclidean Distance to AD Prototype")
    ax.set_xlabel("True Label")

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "quantitative_distance_boxplots.png"), dpi=300)
print(f"Saved quantitative distance boxplots to {out_dir}")
