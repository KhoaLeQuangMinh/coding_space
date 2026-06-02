import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Attempt to use the existing fast resizing from Dataset
try:
    from utils.Dataset import resize_volume_fast, TARGET_SHAPE
except ImportError:
    # Fallback in case run from elsewhere
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
    labels_list = [] # 0: CN, 1: MCI, 2: AD
    
    print(f"Loading {len(files)} files from {data_dir}...")
    for f in tqdm(files):
        path = os.path.join(data_dir, f)
        sample = np.load(path, allow_pickle=True)
        string_label = sample["label"].item()
        
        if string_label == "CN":
            lbl = 0
        elif string_label in ["sMCI", "pMCI"]:
            lbl = 1
        elif string_label == "AD":
            lbl = 2
        else:
            continue
            
        mwp1 = sample["mwp1"]
        mwp1 = np.nan_to_num(mwp1, nan=0.0)
        mwp1 = resize_volume_fast(mwp1, TARGET_SHAPE)
        
        # Flatten and convert to tensor
        flat_tensor = torch.from_numpy(mwp1.flatten()).float()
        data_list.append(flat_tensor)
        labels_list.append(lbl)
        
    X = torch.stack(data_list)
    y = torch.tensor(labels_list)
    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data', help='path to npz files')
    parser.add_argument('--out_dir', type=str, default='./analysis_output', help='where to save plots')
    opt = parser.parse_args()

    os.makedirs(opt.out_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    X, y = load_all_data(opt.data_dir)
    N = X.shape[0]
    
    # Push to device
    X = X.to(device)
    y = y.to(device)
    
    print(f"Loaded feature matrix: {X.shape}")
    
    # 2. Compute similarity matrices
    print("Computing L2 Distances...")
    # L2 distance: ||A-B||^2 = ||A||^2 + ||B||^2 - 2AB
    dist_L2 = torch.cdist(X, X, p=2.0)
    
    print("Computing Cosine Similarities...")
    # Cosine Similarity
    X_norm = F.normalize(X, p=2, dim=1)
    sim_Cos = torch.matmul(X_norm, X_norm.T)
    
    # 3. Triplet Violations Count
    print("\n--- Triplet Violations ---")
    
    def check_triplets(ref_class, class_A, class_B, name_ref, name_A, name_B):
        # A should be closer to Ref than B is to Ref.
        # e.g., Ref=AD, A=AD, B=MCI. AD should be closer to AD than MCI is to AD.
        # If MCI (B) is closer to AD (Ref) than AD (A) is, it's a violation.
        
        refs = torch.nonzero(y == ref_class).squeeze(1) if y.dim() > 1 else torch.nonzero(y == ref_class).squeeze()
        As = torch.nonzero(y == class_A).squeeze(1) if y.dim() > 1 else torch.nonzero(y == class_A).squeeze()
        Bs = torch.nonzero(y == class_B).squeeze(1) if y.dim() > 1 else torch.nonzero(y == class_B).squeeze()
        
        # Handle cases with single element to avoid 0-dim tensor crashes
        if refs.dim() == 0: refs = refs.unsqueeze(0)
        if As.dim() == 0: As = As.unsqueeze(0)
        if Bs.dim() == 0: Bs = Bs.unsqueeze(0)
        
        violations_L2 = 0
        violations_Cos = 0
        total_triplets = 0
        
        for r in refs:
            dist_r_A = dist_L2[r, As] # shape [len(As)]
            dist_r_B = dist_L2[r, Bs] # shape [len(Bs)]
            
            # Count how many B's are smaller than A's
            viol_L2 = (dist_r_B.unsqueeze(1) < dist_r_A.unsqueeze(0)).sum().item()
            violations_L2 += viol_L2
            
            cos_r_A = sim_Cos[r, As]
            cos_r_B = sim_Cos[r, Bs]
            
            # For cosine, larger is closer.
            # Count how many B's are greater than A's
            viol_Cos = (cos_r_B.unsqueeze(1) > cos_r_A.unsqueeze(0)).sum().item()
            violations_Cos += viol_Cos
            
            total_triplets += len(As) * len(Bs)
            
        print(f"[{name_ref} Ref] How many times is {name_B} closer than {name_A}?")
        if total_triplets > 0:
            print(f"  L2 Distance : {violations_L2}/{total_triplets} ({violations_L2/total_triplets*100:.2f}%)")
            print(f"  Cosine Sim  : {violations_Cos}/{total_triplets} ({violations_Cos/total_triplets*100:.2f}%)")
        
    check_triplets(ref_class=2, class_A=2, class_B=1, name_ref="AD", name_A="AD", name_B="MCI")
    check_triplets(ref_class=0, class_A=0, class_B=1, name_ref="CN", name_A="CN", name_B="MCI")
    check_triplets(ref_class=2, class_A=2, class_B=0, name_ref="AD", name_A="AD", name_B="CN")
    check_triplets(ref_class=0, class_A=0, class_B=2, name_ref="CN", name_A="CN", name_B="AD")
    
    # 4. Global Severity Scoring
    print("\nComputing Global Severity Scores...")
    # Calculate Mean CN and Mean AD
    mean_CN = X[y == 0].mean(dim=0).unsqueeze(0) # [1, D]
    mean_AD = X[y == 2].mean(dim=0).unsqueeze(0) # [1, D]
    
    # Global Severity (L2): Distance to CN - Distance to AD 
    dist_to_CN_L2 = torch.cdist(X, mean_CN, p=2.0).squeeze()
    dist_to_AD_L2 = torch.cdist(X, mean_AD, p=2.0).squeeze()
    severity_L2 = dist_to_CN_L2 - dist_to_AD_L2
    
    # Global Severity (Cosine): Similarity to AD - Similarity to CN
    mean_CN_norm = F.normalize(mean_CN, p=2, dim=1)
    mean_AD_norm = F.normalize(mean_AD, p=2, dim=1)
    sim_to_CN_Cos = torch.matmul(X_norm, mean_CN_norm.T).squeeze()
    sim_to_AD_Cos = torch.matmul(X_norm, mean_AD_norm.T).squeeze()
    severity_Cos = sim_to_AD_Cos - sim_to_CN_Cos
    
    y_cpu = y.cpu().numpy()
    sev_L2_cpu = severity_L2.cpu().numpy()
    sev_Cos_cpu = severity_Cos.cpu().numpy()
    
    class_names = {0: 'CN', 1: 'MCI', 2: 'AD'}
    labels_str = [class_names[lbl] for lbl in y_cpu]
    
    df = pd.DataFrame({
        'Label': labels_str,
        'Severity_L2': sev_L2_cpu,
        'Severity_Cos': sev_Cos_cpu
    })
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='Severity_L2', hue='Label', fill=True, common_norm=False, palette={'CN':'blue', 'MCI':'green', 'AD':'red'})
    plt.title('Global Severity Score Distribution (L2 Distance)')
    plt.xlabel('Severity (Distance to CN - Distance to AD)')
    plt.ylabel('Density')
    l2_path = os.path.join(opt.out_dir, 'severity_dist_L2.png')
    plt.savefig(l2_path)
    print(f"Saved L2 plot to {l2_path}")
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='Severity_Cos', hue='Label', fill=True, common_norm=False, palette={'CN':'blue', 'MCI':'green', 'AD':'red'})
    plt.title('Global Severity Score Distribution (Cosine Similarity)')
    plt.xlabel('Severity (Sim to AD - Sim to CN)')
    plt.ylabel('Density')
    cos_path = os.path.join(opt.out_dir, 'severity_dist_Cosine.png')
    plt.savefig(cos_path)
    print(f"Saved Cosine plot to {cos_path}")

if __name__ == '__main__':
    main()
