import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
# Matches the extracted features from your Kaggle output
csv_dir = '/Users/khoale/Downloads/analysis_output/extracted_features'
out_dir = './analysis_output/tsne_plots'
os.makedirs(out_dir, exist_ok=True)

# Define the models we want to plot (e.g., ce and your triplet method)
VARIANTS_TO_PLOT = ['ce', 'exp_triplet_ins2cls', 'triplet_only']

# We are analyzing the 'best_2c_net'
checkpoint_name = 'best_2c_net'

# Strict coloring to maintain consistency with previous plots
palette = {
    'CN': '#1f77b4',     # Blue
    'sMCI': '#2ca02c',   # Green
    'pMCI': '#ff7f0e',   # Orange
    'AD': '#d62728'      # Red
}

def plot_tsne_for_variant(variant):
    print(f"\nProcessing t-SNE for {variant}...")
    
    # 1. Load data from all 5 folds
    all_features = []
    all_labels = []
    
    for fold in range(1, 6):
        csv_path = os.path.join(csv_dir, f"{variant}_{checkpoint_name}_fold{fold}.csv")
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
        
        # Extract the 512 feature columns
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        if not feature_cols:
            print(f"Warning: Fold {fold} for {variant} does not contain raw features. Skipping.")
            continue
            
        all_features.append(df[feature_cols].values)
        all_labels.extend(df['True Label'].tolist())
        
    if not all_features:
        print(f"No feature data found for {variant}. Did you run the updated extract_latent_features.py?")
        return
        
    # Combine all folds into a single massive array
    X = np.vstack(all_features)
    y = np.array(all_labels)
    
    print(f"  -> Total patients loaded: {X.shape[0]}, Features: {X.shape[1]}")
    
    # 2. Run t-SNE Optimization
    print("  -> Running t-SNE optimization (this might take a few seconds)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X)
    
    # 3. Create DataFrame for plotting
    plot_df = pd.DataFrame({
        't-SNE 1': X_tsne[:, 0],
        't-SNE 2': X_tsne[:, 1],
        'Diagnosis': y
    })
    
    # Order the legend logically
    hue_order = ['CN', 'sMCI', 'pMCI', 'AD']
    
    # 4. Plot using Seaborn
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
    
    plt.title(f't-SNE Visualization of Latent Features\nModel: {variant} ({checkpoint_name})', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Disease Stage', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save high-res plot
    save_path = os.path.join(out_dir, f"tsne_{variant}_{checkpoint_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  -> Saved plot to {save_path}")

def main():
    print("Starting t-SNE Generation Pipeline...")
    for variant in VARIANTS_TO_PLOT:
        plot_tsne_for_variant(variant)
    print("\nFinished!")

if __name__ == '__main__':
    main()
