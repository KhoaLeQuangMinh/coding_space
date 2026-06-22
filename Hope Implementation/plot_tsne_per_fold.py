import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
csv_dir = '/Users/khoale/Downloads/analysis_output_tSNE/extracted_features'
out_dir = '/Users/khoale/Downloads/plot_display/plots/tsne_folds'
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
    'hierarchical_triplet_only',
    'exp_hierarchical_triplet_ins2cls',
    'full_4class',
    'exp_triplet_ins2cls_4class',
    'qwk_hierarchical_triplet_4class',
    'exp_3pole_local',
    'exp_3pole_global',
    '3pole_local_only',
    '3pole_global_only',
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

def plot_tsne_per_fold(variant, checkpoint_name, fold):
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
        
    print(f"  -> {variant} ({checkpoint_name}) Fold {fold}: {X.shape[0]} patients. Running t-SNE...")
    
    # Run t-SNE Optimization with lower perplexity for smaller single-fold datasets
    tsne = TSNE(n_components=2, random_state=42, perplexity=20, method='exact')
    X_tsne = tsne.fit_transform(X)
    
    plot_df = pd.DataFrame({
        't-SNE 1': X_tsne[:, 0],
        't-SNE 2': X_tsne[:, 1],
        'Diagnosis': y
    })
    
    hue_order = ['CN', 'sMCI', 'pMCI', 'AD']
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=plot_df,
        x='t-SNE 1',
        y='t-SNE 2',
        hue='Diagnosis',
        hue_order=hue_order,
        palette=palette,
        alpha=0.7,
        s=60,
        edgecolor='k',
        linewidth=0.5
    )
    
    plt.title(f't-SNE Latent Features - Fold {fold}\nModel: {variant} ({checkpoint_name})', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Disease Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, f"tsne_{variant}_{checkpoint_name}_fold{fold}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print(f"Starting Per-Fold t-SNE Generation Pipeline (Output to: {out_dir})")
    for ckpt in CHECKPOINTS:
        for variant in VARIANTS_TO_PLOT:
            for fold in FOLDS:
                plot_tsne_per_fold(variant, ckpt, fold)
    print("\nFinished generating all per-fold plots!")

if __name__ == '__main__':
    main()
