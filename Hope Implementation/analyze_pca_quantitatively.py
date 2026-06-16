import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr

csv_dir = '/Users/khoale/Downloads/analysis_output_tSNE/extracted_features'

# We will analyze the 4-class checkpoint which is the hardest
ckpt = 'best_4c_net'

VARIANTS = [
    'ce', 
    'full',
    'hierarchical_triplet_only',
    'exp_hierarchical_triplet_ins2cls'
]

# Map string labels to numeric ordinal values for correlation
label_to_ordinal = {
    'CN': 0,
    'sMCI': 1,
    'pMCI': 2,
    'AD': 3
}

print(f"--- Quantitative PCA Analysis ({ckpt}) ---")
print(f"{'Variant':<35} | {'Avg Silhouette':<15} | {'Avg PC1-Severity Correlation':<30}")
print("-" * 85)

for variant in VARIANTS:
    fold_silhouettes = []
    fold_correlations = []
    
    for fold in range(1, 6):
        csv_path = os.path.join(csv_dir, f"{variant}_{ckpt}_fold{fold}.csv")
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        if not feature_cols:
            continue
            
        X = df[feature_cols].values
        labels_str = df['True Label'].values
        
        # Convert string labels to ordinal numbers
        y_ordinal = np.array([label_to_ordinal[lbl] for lbl in labels_str])
        
        # PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        
        # 1. Silhouette Score (Measures how well clusters are separated)
        # Note: silhouette score isn't perfect for ordinal data, but gives a rough idea of spread
        if len(np.unique(y_ordinal)) > 1:
            sil_score = silhouette_score(X_pca, y_ordinal)
            fold_silhouettes.append(sil_score)
        
        # 2. Pearson Correlation between Principal Component 1 and Disease Severity
        # We take the absolute value because PCA direction (sign) is arbitrary
        corr, _ = pearsonr(X_pca[:, 0], y_ordinal)
        fold_correlations.append(abs(corr))
        
    if fold_silhouettes and fold_correlations:
        avg_sil = np.mean(fold_silhouettes)
        avg_corr = np.mean(fold_correlations)
        print(f"{variant:<35} | {avg_sil:+.4f}         | {avg_corr:.4f}")
    else:
        print(f"{variant:<35} | N/A             | N/A")
