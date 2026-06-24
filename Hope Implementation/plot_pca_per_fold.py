import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
csv_dir = '/Users/khoale/Downloads/analysis_output_tSNE/extracted_features'
out_dir = '/Users/khoale/Downloads/plot_display/plots/pca_folds'
os.makedirs(out_dir, exist_ok=True)

VARIANTS_TO_PLOT = [
    'ce', 
    'ins2ins', 
    'ins2cls', 
    'full',
    'exclude_ins2ins',
    'exclude_ins2cls',
    'exp_triplet_ins2cls',
    'triplet_only',
    'triplet_only_margin0.3',
    'triplet_only_margin3.0',
    'triplet_only_ema0.5_margin0.0',
    'hierarchical_triplet_only',
    'exp_hierarchical_triplet_ins2cls',
    'full_4class',
    'exp_triplet_ins2cls_4class',
    'qwk_hierarchical_triplet_4class',
    'exp_3pole_local',
    'exp_3pole_global',
    '3pole_local_only',
    '3pole_global_only',
    '3pole_local_only_margin0.0',
    '3pole_global_only_margin0.0',
    'hierarchical_triplet_only_4class'
]

CHECKPOINTS = ['best_2c_net', 'best_3c_net', 'best_4c_net']
FOLDS = [1, 2, 3, 4, 5]

palette = {
    'CN': '#1f77b4',     # Blue
    'sMCI': '#2ca02c',   # Green
    'pMCI': '#ff7f0e',   # Orange
    'AD': '#d62728'      # Red
}

def plot_pca_per_fold(variant, checkpoint_name, fold):
    csv_path = os.path.join(csv_dir, f"{variant}_{checkpoint_name}_fold{fold}.csv")
    if not os.path.exists(csv_path):
        return # Skip silently if the fold doesn't exist
        
    df = pd.read_csv(csv_path)
    
    # Extract the 512 feature columns
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    if not feature_cols:
        print(f"Warning: Fold {fold} for {variant} does not contain raw features. Skipping.")
        return
        
    X = df[feature_cols].values
    y = df['True Label'].values
    
    if X.shape[0] < 5: # Safety check
        return
        
    print(f"  -> {variant} ({checkpoint_name}) Fold {fold}: {X.shape[0]} patients. Running PCA...")
    
    # Run PCA Optimization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    var_ratio = pca.explained_variance_ratio_
    
    plot_df = pd.DataFrame({
        'PCA 1': X_pca[:, 0],
        'PCA 2': X_pca[:, 1],
        'Diagnosis': y
    })
    
    hue_order = ['CN', 'sMCI', 'pMCI', 'AD']
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=plot_df,
        x='PCA 1',
        y='PCA 2',
        hue='Diagnosis',
        hue_order=hue_order,
        palette=palette,
        alpha=0.7,
        s=60,
        edgecolor='k',
        linewidth=0.5
    )
    
    plt.title(f'PCA Latent Features - Fold {fold}\nModel: {variant} ({checkpoint_name})', fontsize=14, fontweight='bold')
    plt.xlabel(f'Principal Component 1 ({var_ratio[0]*100:.1f}% Variance)')
    plt.ylabel(f'Principal Component 2 ({var_ratio[1]*100:.1f}% Variance)')
    plt.legend(title='Disease Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, f"pca_{variant}_{checkpoint_name}_fold{fold}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print(f"Starting Per-Fold PCA Generation Pipeline (Output to: {out_dir})")
    for ckpt in CHECKPOINTS:
        for variant in VARIANTS_TO_PLOT:
            for fold in FOLDS:
                plot_pca_per_fold(variant, ckpt, fold)
    print("\nFinished generating all PCA per-fold plots!")

if __name__ == '__main__':
    main()
