import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

# Try to use existing fast resizing
try:
    from utils.Dataset import resize_volume_fast, TARGET_SHAPE
except ImportError:
    TARGET_SHAPE = (128, 128, 128)
    def resize_volume_fast(volume: np.ndarray, target_shape=TARGET_SHAPE) -> np.ndarray:
        if volume.shape == target_shape:
            return volume.astype(np.float32)
        tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(tensor, size=target_shape, mode='trilinear', align_corners=True)
        return resized.squeeze(0).squeeze(0).numpy().astype(np.float32)

def load_all_data(data_dir):
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
    data_list = []
    labels_list_4class = []
    
    print(f"Loading {len(files)} files from {data_dir}...")
    for f in tqdm(files):
        path = os.path.join(data_dir, f)
        sample = np.load(path, allow_pickle=True)
        string_label = sample["label"].item()
        
        if string_label == "CN":
            lbl_4 = 0
        elif string_label == "sMCI":
            lbl_4 = 1
        elif string_label == "pMCI":
            lbl_4 = 2
        elif string_label == "AD":
            lbl_4 = 3
        else:
            continue
            
        mwp1 = sample["mwp1"]
        mwp1 = np.nan_to_num(mwp1, nan=0.0)
        mwp1 = resize_volume_fast(mwp1, TARGET_SHAPE)
        
        flat_tensor = torch.from_numpy(mwp1.flatten()).float()
        data_list.append(flat_tensor)
        labels_list_4class.append(lbl_4)
        
    X = torch.stack(data_list)
    y_4class = torch.tensor(labels_list_4class)
    return X, y_4class

def compute_similarities(X, mean_CN, mean_AD):
    # Process in chunks to save GPU memory
    chunk_size = 50
    N = X.shape[0]
    X_norm = F.normalize(X, p=2, dim=1)
    mean_CN_norm = F.normalize(mean_CN, p=2, dim=1)
    mean_AD_norm = F.normalize(mean_AD, p=2, dim=1)
    
    sim_to_CN_list = []
    sim_to_AD_list = []
    
    for i in range(0, N, chunk_size):
        X_chunk = X_norm[i:i+chunk_size]
        sim_to_CN_list.append(torch.matmul(X_chunk, mean_CN_norm.T).squeeze(1))
        sim_to_AD_list.append(torch.matmul(X_chunk, mean_AD_norm.T).squeeze(1))

    sim_to_CN = torch.cat(sim_to_CN_list).cpu().numpy()
    sim_to_AD = torch.cat(sim_to_AD_list).cpu().numpy()
    return np.column_stack((sim_to_CN, sim_to_AD))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data', help='path to npz files')
    parser.add_argument('--out_dir', type=str, default='./analysis_output', help='where to save plots')
    parser.add_argument('--components', type=int, default=4, help='Number of GMM clusters')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size ratio')
    opt = parser.parse_args()

    os.makedirs(opt.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    X, y_4 = load_all_data(opt.data_dir)
    
    # 2. Train/Test Split (Stratified to maintain class ratios)
    print(f"\nSplitting data into Train ({(1-opt.test_size)*100:.0f}%) and Test ({opt.test_size*100:.0f}%)...")
    indices = np.arange(X.shape[0])
    train_idx, test_idx = train_test_split(indices, test_size=opt.test_size, stratify=y_4.numpy(), random_state=42)
    
    X_train, y_train = X[train_idx].to(device), y_4[train_idx].to(device)
    X_test, y_test = X[test_idx].to(device), y_4[test_idx].to(device)
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 3. Calculate Mean CN and AD strictly from Training Set
    print("\nCalculating Mean Profiles from Training Set only...")
    mean_CN_train = X_train[y_train == 0].mean(dim=0).unsqueeze(0)
    mean_AD_train = X_train[y_train == 3].mean(dim=0).unsqueeze(0)
    
    # 4. Compute 2D Features
    print("Computing Similarities...")
    train_features = compute_similarities(X_train, mean_CN_train, mean_AD_train)
    test_features = compute_similarities(X_test, mean_CN_train, mean_AD_train)
    
    # 5. Fit GMM on Train, Predict on Test
    print(f"\nFitting GMM with {opt.components} components on Training Set...")
    gmm = GaussianMixture(n_components=opt.components, covariance_type='full', random_state=42)
    gmm.fit(train_features)
    
    print("Predicting Clusters for Test Set...")
    test_preds = gmm.predict(test_features)
    
    # 6. Results on Test Set
    print("\n--- TEST SET GMM Clustering Results (Cross-Tabulation) ---")
    class_names = {0: 'CN', 1: 'sMCI', 2: 'pMCI', 3: 'AD'}
    y_test_cpu = y_test.cpu().numpy()
    
    df_test = pd.DataFrame({
        'True Label': [class_names[lbl] for lbl in y_test_cpu],
        'GMM Cluster': [f"Cluster {c}" for c in test_preds],
        'Sim_to_CN': test_features[:, 0],
        'Sim_to_AD': test_features[:, 1]
    })
    
    crosstab = pd.crosstab(df_test['True Label'], df_test['GMM Cluster'])
    crosstab = crosstab.reindex(['CN', 'sMCI', 'pMCI', 'AD'])
    print(crosstab)
    
    # 7. Plotting Test Set
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df_test, x='Sim_to_CN', y='Sim_to_AD', hue='True Label', 
                    palette={'CN':'blue', 'sMCI':'green', 'pMCI':'orange', 'AD':'red'}, alpha=0.7, s=60)
    plt.title('TEST SET: True Clinical Labels')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df_test, x='Sim_to_CN', y='Sim_to_AD', hue='GMM Cluster', 
                    palette='tab10', alpha=0.7, s=60)
    plt.title(f'TEST SET: GMM Predictions (Trained on Train Set)')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plot_path = os.path.join(opt.out_dir, 'gmm_2d_scatter_testset.png')
    plt.savefig(plot_path)
    print(f"\nSaved Test Set scatter plot to {plot_path}")

if __name__ == '__main__':
    main()
