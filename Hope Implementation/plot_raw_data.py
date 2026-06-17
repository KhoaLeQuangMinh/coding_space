import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.Dataset import Dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/datasets/kisokoghan/paired-npz/paired_npz', help='path to the dataset')
    parser.add_argument('--out_dir', type=str, default='./raw_plots', help='where to save the plots')
    parser.add_argument('--sample_size', type=int, default=1000, help='Max patients to sample (to prevent Kaggle OOM)')
    opt = parser.parse_args()

    os.makedirs(opt.out_dir, exist_ok=True)

    print("Initializing Dataset...")
    # 'valid' or 'test' mode loads all 4 classes for evaluation.
    # We use valid to grab 10% of the data or we can just use total files
    dataset = Dataset(mode="test", data_dir=opt.data_dir, seed=42)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    label_map_inv = {0: "CN", 1: "sMCI", 2: "pMCI", 3: "AD"}
    
    X = []
    y = []

    print(f"Loading up to {opt.sample_size} raw MRI scans from dataloader...")
    for i, (img, label) in enumerate(loader):
        if i >= opt.sample_size:
            break
            
        # img shape is [1, 1, 128, 128, 128]
        # Flattening without downsampling (Warning: Very high RAM usage)
        flat_img = img.view(-1).numpy()
        
        X.append(flat_img)
        y.append(label_map_inv[label.item()])
        
        if (i+1) % 50 == 0:
            print(f"  Processed {i+1} scans...")

    if not X:
        print("No data found! Check your data_dir path.")
        return

    X = np.stack(X)
    print(f"\nFinal feature matrix shape: {X.shape}")
    
    print("Running PCA projection on raw voxels...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    print("Running t-SNE projection on raw voxels...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)

    df = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        't-SNE1': X_tsne[:, 0],
        't-SNE2': X_tsne[:, 1],
        'Diagnosis': y
    })
    
    hue_order = ['CN', 'sMCI', 'pMCI', 'AD']
    palette = {'CN': '#1f77b4', 'sMCI': '#2ca02c', 'pMCI': '#ff7f0e', 'AD': '#d62728'}

    print("Generating and saving plots...")
    
    # PCA Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Diagnosis', hue_order=hue_order, palette=palette, alpha=0.7)
    plt.title('PCA of Raw MRI Images\n(Before Neural Network Feature Extraction)')
    plt.savefig(os.path.join(opt.out_dir, 'raw_data_pca.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # t-SNE Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='t-SNE1', y='t-SNE2', hue='Diagnosis', hue_order=hue_order, palette=palette, alpha=0.7)
    plt.title('t-SNE of Raw MRI Images\n(Before Neural Network Feature Extraction)')
    plt.savefig(os.path.join(opt.out_dir, 'raw_data_tsne.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\nSuccessfully saved raw distribution plots to: {opt.out_dir}")

if __name__ == '__main__':
    main()
