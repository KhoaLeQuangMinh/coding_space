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

def compute_1d_severity(X, mean_CN, mean_AD):
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
    
    # 1D Severity: Similarity to AD - Similarity to CN
    severity = sim_to_AD - sim_to_CN
    return severity.reshape(-1, 1)

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
    
    # 2. Train/Test Split
    print(f"\nSplitting data into Train ({(1-opt.test_size)*100:.0f}%) and Test ({opt.test_size*100:.0f}%)...")
    indices = np.arange(X.shape[0])
    train_idx, test_idx = train_test_split(indices, test_size=opt.test_size, stratify=y_4.numpy(), random_state=42)
    
    X_train, y_train = X[train_idx].to(device), y_4[train_idx].to(device)
    X_test, y_test = X[test_idx].to(device), y_4[test_idx].to(device)
    
    # 3. Calculate Means from Train Set
    mean_CN_train = X_train[y_train == 0].mean(dim=0).unsqueeze(0)
    mean_AD_train = X_train[y_train == 3].mean(dim=0).unsqueeze(0)
    
    # 4. Compute 1D Features (Sim to AD - Sim to CN)
    print("Computing 1D Severity Scores...")
    train_features = compute_1d_severity(X_train, mean_CN_train, mean_AD_train)
    test_features = compute_1d_severity(X_test, mean_CN_train, mean_AD_train)
    
    # 5. Fit GMM on 1D feature
    print(f"\nFitting GMM with {opt.components} components on 1D Severity...")
    gmm = GaussianMixture(n_components=opt.components, covariance_type='full', random_state=42)
    gmm.fit(train_features)
    
    print("Predicting Clusters for Test Set...")
    test_preds = gmm.predict(test_features)
    
    # Sort clusters logically from left (CN-like) to right (AD-like) based on cluster mean
    cluster_means = gmm.means_.flatten()
    sorted_cluster_idx = np.argsort(cluster_means)
    cluster_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_cluster_idx)}
    test_preds_sorted = np.array([cluster_mapping[c] for c in test_preds])
    
    # 6. Results
    print("\n--- TEST SET 1D GMM Clustering Results ---")
    class_names = {0: 'CN', 1: 'sMCI', 2: 'pMCI', 3: 'AD'}
    y_test_cpu = y_test.cpu().numpy()
    
    df_test = pd.DataFrame({
        'True Label': [class_names[lbl] for lbl in y_test_cpu],
        'GMM Cluster': [f"Cluster {c}" for c in test_preds_sorted],
        'Severity': test_features.flatten()
    })
    
    crosstab = pd.crosstab(df_test['True Label'], df_test['GMM Cluster'])
    crosstab = crosstab.reindex(['CN', 'sMCI', 'pMCI', 'AD'])
    print(crosstab)
    
    # 7. Plotting Test Set
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.kdeplot(data=df_test, x='Severity', hue='True Label', fill=True, common_norm=False, 
                palette={'CN':'blue', 'sMCI':'green', 'pMCI':'orange', 'AD':'red'})
    plt.title('TEST SET: True Clinical Labels (1D Severity)')
    
    plt.subplot(1, 2, 2)
    sns.kdeplot(data=df_test, x='Severity', hue='GMM Cluster', fill=True, common_norm=False, 
                palette='tab10')
    plt.title(f'TEST SET: GMM Predictions ({opt.components} Components)')
    
    plt.tight_layout()
    plot_path = os.path.join(opt.out_dir, 'gmm_1d_severity.png')
    plt.savefig(plot_path)
    print(f"\nSaved 1D GMM plot to {plot_path}")

if __name__ == '__main__':
    main()
