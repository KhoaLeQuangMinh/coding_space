import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def main():
    parser = argparse.ArgumentParser(description="Analyze learned prototypes and visualize them in t-SNE space.")
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the model checkpoint (.pth file)')
    parser.add_argument('--csv', type=str, required=True, help='Path to the extracted features CSV file')
    parser.add_argument('--out_img', type=str, default='prototype_tsne_comparison.png', help='Path to save the output t-SNE image')
    parser.add_argument('--perplexity', type=int, default=30, help='t-SNE perplexity (default: 30)')
    opt = parser.parse_args()

    # 1. Load Checkpoint & Extract Prototypes
    if not os.path.exists(opt.ckpt):
        print(f"Error: Checkpoint file not found at: {opt.ckpt}")
        return
        
    print(f"Loading checkpoint from: {opt.ckpt}...")
    checkpoint = torch.load(opt.ckpt, map_location='cpu')
    
    prototypes = None
    if 'prototypes' in checkpoint:
        prototypes = checkpoint['prototypes']
    elif 'state_dict' in checkpoint and 'prototypes' in checkpoint['state_dict']:
        prototypes = checkpoint['state_dict']['prototypes']
        
    if prototypes is None:
        print("Error: Could not find 'prototypes' key in checkpoint.")
        print("Available keys in checkpoint:", list(checkpoint.keys()))
        return
        
    # Convert to numpy array
    if isinstance(prototypes, torch.Tensor):
        prototypes = prototypes.detach().numpy()
        
    num_classes, dim = prototypes.shape
    print(f"Successfully loaded prototypes! Shape: {num_classes} classes x {dim} dimensions.")

    # 2. Compute Norms & Pairwise Cosine Similarity
    norms = np.linalg.norm(prototypes, axis=1)
    print("\n--- Prototype L2 Norms ---")
    for i, norm in enumerate(norms):
        print(f"  Class {i} Prototype Norm: {norm:.4f}")

    # L2 normalize the prototypes to compute cosine similarity
    protos_norm = prototypes / norms[:, np.newaxis]
    cosine_sim_matrix = np.dot(protos_norm, protos_norm.T)

    print("\n--- Pairwise Cosine Similarity Matrix ---")
    header = "          " + "".join([f"Class {j:<8}" for j in range(num_classes)])
    print(header)
    for i in range(num_classes):
        row_str = f"Class {i:<4} | "
        for j in range(num_classes):
            row_str += f"{cosine_sim_matrix[i, j]:8.4f} "
        print(row_str)

    # Check for collapse
    # Average off-diagonal similarity
    off_diag_vals = []
    for i in range(num_classes):
        for j in range(i+1, num_classes):
            off_diag_vals.append(cosine_sim_matrix[i, j])
            
    avg_off_diag = np.mean(off_diag_vals) if off_diag_vals else 0.0
    print(f"\nAverage off-diagonal cosine similarity: {avg_off_diag:.4f}")
    if avg_off_diag > 0.95:
        print("WARNING: Prototypes show extremely high cosine similarity (> 0.95). They might have collapsed!")
    elif avg_off_diag < 0.1:
        print("EXCELLENT: Prototypes are highly orthogonal/distinct (< 0.1 similarity).")
    else:
        print("GOOD: Prototypes are moderately separated.")

    # 3. Load Extracted Features
    if not os.path.exists(opt.csv):
        print(f"\nError: CSV file not found at: {opt.csv}")
        return
        
    print(f"\nLoading extracted features from: {opt.csv}...")
    df = pd.read_csv(opt.csv)
    
    feature_cols = [c for c in df.columns if c.startswith('feature_')]
    if len(feature_cols) == 0:
        print("Error: No feature columns found in the CSV (should start with 'feature_').")
        return
        
    print(f"Found {len(feature_cols)} feature dimensions and {len(df)} test samples.")
    
    feats = df[feature_cols].values
    true_labels = df['True Label'].values
    
    # 4. L2 Normalize Features
    feats_norm = feats / np.linalg.norm(feats, axis=1, keepdims=True)

    # 5. Combine Features and Prototypes for Joint t-SNE
    # We concatenate the normalized features and the normalized prototypes
    combined_data = np.vstack([feats_norm, protos_norm])
    print(f"Running joint t-SNE on combined array of shape {combined_data.shape}...")
    
    tsne = TSNE(n_components=2, perplexity=opt.perplexity, random_state=42, init='pca', learning_rate='auto')
    combined_tsne = tsne.fit_transform(combined_data)
    
    # Separate back into features and prototypes
    tsne_feats = combined_tsne[:-num_classes]
    tsne_protos = combined_tsne[-num_classes:]
    
    # 6. Plotting
    plt.figure(figsize=(10, 8))
    
    # Define color scheme matching the dashboard
    color_palette = {'CN': '#66c2a5', 'sMCI': '#fc8d62', 'pMCI': '#8da0cb', 'AD': '#e78ac3'}
    class_order = ['CN', 'sMCI', 'pMCI', 'AD']
    
    # Map prototype indices to their biological class labels
    proto_labels = {}
    if num_classes == 3:
        proto_labels = {0: 'CN Prototype', 1: 'MCI Prototype', 2: 'AD Prototype'}
    elif num_classes == 4:
        proto_labels = {0: 'CN Prototype', 1: 'sMCI Prototype', 2: 'pMCI Prototype', 3: 'AD Prototype'}
    else:
        proto_labels = {i: f'Class {i} Prototype' for i in range(num_classes)}
        
    # Plot test features
    df_tsne = pd.DataFrame({
        't-SNE 1': tsne_feats[:, 0],
        't-SNE 2': tsne_feats[:, 1],
        'Diagnosis': true_labels
    })
    
    # Check what labels are actually present in the dataset
    present_classes = [c for c in class_order if c in df_tsne['Diagnosis'].unique()]
    
    sns.scatterplot(data=df_tsne, x='t-SNE 1', y='t-SNE 2', hue='Diagnosis', 
                    palette=color_palette, hue_order=present_classes, alpha=0.7, s=40, edgecolor='w', linewidth=0.3)
    
    # Plot Prototypes as large distinct stars
    proto_colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a'] # High contrast colors
    for i in range(num_classes):
        label_text = proto_labels[i]
        plt.scatter(tsne_protos[i, 0], tsne_protos[i, 1], 
                    color=proto_colors[i % len(proto_colors)], 
                    marker='*', s=350, edgecolor='black', linewidth=1.5,
                    label=label_text, zorder=10)
        # Add text label next to the prototype star
        plt.text(tsne_protos[i, 0] + 0.5, tsne_protos[i, 1] + 0.5, 
                 f"  {label_text}", fontsize=11, fontweight='bold', 
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2', edgecolor='grey'), zorder=11)

    plt.title(f"Joint t-SNE of Latent Space Features & Learned Prototypes\nCheckpoint: {os.path.basename(opt.ckpt)}", fontsize=14, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=11)
    plt.ylabel("t-SNE Dimension 2", fontsize=11)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(opt.out_img, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSuccessfully generated joint t-SNE plot and saved to: {opt.out_img}")

if __name__ == "__main__":
    main()
