import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from tqdm import tqdm
from utils.Dataset import Dataset
from models.Resnet import resnet18

def compute_1d_severity(features, proto_CN, proto_AD):
    """
    Computes severity based on Cosine Similarity to the prototypes.
    Severity = Cosine_Sim(Feature, AD_Prototype) - Cosine_Sim(Feature, CN_Prototype)
    Higher value means more AD-like. Lower value means more CN-like.
    """
    # Normalize features and prototypes to compute cosine similarity
    feat_norm = F.normalize(features, p=2, dim=1)
    cn_norm = F.normalize(proto_CN.unsqueeze(0), p=2, dim=1)
    ad_norm = F.normalize(proto_AD.unsqueeze(0), p=2, dim=1)
    
    sim_to_cn = torch.matmul(feat_norm, cn_norm.T).squeeze(1)
    sim_to_ad = torch.matmul(feat_norm, ad_norm.T).squeeze(1)
    
    severity = sim_to_ad - sim_to_cn
    return severity.cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data', help='path to npz files')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='path to checkpoints')
    parser.add_argument('--out_dir', type=str, default='./analysis_output/latent_features', help='where to save plots')
    opt = parser.parse_args()

    os.makedirs(opt.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Configuration matching the ablation analysis
    LOSS_VARIANTS = ['ce', 'ins2ins', 'ins2cls', 'full', 'exclude_ins2ins', 'exclude_ins2cls', 'exp_triplet_ins2cls', 'exp_3pole_local', 'exp_3pole_global', '3pole_local_only', '3pole_global_only']
    CHECKPOINTS = ['best_2c_net.pth', 'best_3c_net.pth', 'best_4c_net.pth']
    N_FOLDS = 5

    class_names = {0: 'CN', 1: 'sMCI', 2: 'pMCI', 3: 'AD'}

    # Cache test loaders because loading volumetric MRI from disk takes a while
    print("Pre-loading the 5-fold Test datasets (This captures 100% of the data across all folds)...")
    test_loaders = {}
    for fold in range(1, N_FOLDS + 1):
        # We need return_4c=True to get the ground truth 4-class labels for plotting
        dataset = Dataset(mode="test", data_dir=opt.data_dir, seed=42, kfold=N_FOLDS, current_fold=fold, return_4c=True)
        # Drop_last=False because we want every single patient in the dataset
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        test_loaders[fold] = loader

    # Main iteration
    for variant in LOSS_VARIANTS:
        for ckpt_name in CHECKPOINTS:
            print(f"\n{'='*70}")
            print(f"ANALYZING: Variant [{variant}] | Checkpoint [{ckpt_name}]")
            print(f"{'='*70}")
            
            all_severities = []
            all_labels = []
            missing_folds = 0
            
            # Extract features across all 5 folds
            for fold in range(1, N_FOLDS + 1):
                ckpt_path = os.path.join(opt.checkpoints_dir, f"ablation_loss_{variant}_fold{fold}", ckpt_name)
                
                if not os.path.exists(ckpt_path):
                    print(f"  [WARN] Missing {ckpt_path}")
                    missing_folds += 1
                    continue
                    
                # Load model architecture
                model = resnet18(class_num=3)
                state_dict = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)
                
                # Check if model has prototypes initialized (CE might not have meaningful ones, but they exist in state_dict)
                if 'prototypes' in state_dict:
                    model.prototypes = state_dict['prototypes'].to(device)
                else:
                    model.prototypes = model.prototypes.to(device)
                    
                model.to(device)
                model.eval()
                
                # We need the learned CN prototype (index 0) and AD prototype (index 2)
                proto_cn = model.prototypes[0]
                proto_ad = model.prototypes[2]
                
                loader = test_loaders[fold]
                
                with torch.no_grad():
                    for batch in loader:
                        imgs = batch[0].to(device)
                        labels_4c = batch[2].numpy() # we requested return_4c=True
                        
                        # Forward pass to get latent features (x_ori is the 512-d feature before classifier)
                        x_ori, _, _ = model(imgs)
                        
                        # Calculate Severity based on distance to the learned prototypes
                        severity = compute_1d_severity(x_ori, proto_cn, proto_ad)
                        
                        all_severities.extend(severity.tolist())
                        all_labels.extend(labels_4c.tolist())
                        
            if missing_folds == N_FOLDS:
                print("  -> Skipping plot (no folds found).")
                continue
                
            # --- Plotting ---
            df = pd.DataFrame({
                'Severity': all_severities,
                'True Label': [class_names[lbl] for lbl in all_labels]
            })
            
            # Reorder for proper plotting stack sequence if desired, or let seaborn handle it
            plt.figure(figsize=(12, 7))
            sns.kdeplot(data=df, x='Severity', hue='True Label', fill=True, common_norm=False, 
                        palette={'CN':'blue', 'sMCI':'green', 'pMCI':'orange', 'AD':'red'},
                        alpha=0.4, linewidth=2.5)
            
            plt.title(f"Latent Feature Distribution\nVariant: {variant}  |  Checkpoint: {ckpt_name}\n(Aggregated 100% of Dataset across {N_FOLDS - missing_folds} Folds)", fontsize=16, pad=15)
            plt.xlabel("Latent 1D Severity: (Sim to AD Prototype) - (Sim to CN Prototype)", fontsize=14)
            plt.ylabel("Density", fontsize=14)
            
            # Add vertical lines at 0 to show the neutral boundary
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Neutral Boundary')
            
            # Tweak legend
            plt.legend(title='Clinical Diagnosis', loc='upper left', fontsize=12, title_fontsize=12)
            
            plt.tight_layout()
            
            # Save
            plot_filename = f"latent_kde_{variant}_{ckpt_name.split('.')[0]}.png"
            plot_path = os.path.join(opt.out_dir, plot_filename)
            plt.savefig(plot_path, dpi=200)
            plt.close()
            
            print(f"  -> Successfully generated {plot_path}")
            print(f"  -> Aggregated {len(all_severities)} patient MRI features.")

if __name__ == '__main__':
    main()
