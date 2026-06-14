import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from utils.Dataset import Dataset
from models.Resnet import resnet18

def compute_1d_severity(features, proto_CN, proto_AD):
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
    parser.add_argument('--out_dir', type=str, default='./analysis_output/extracted_features', help='where to save CSVs')
    opt = parser.parse_args()

    os.makedirs(opt.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    LOSS_VARIANTS = ['ce', 'ins2ins', 'ins2cls', 'full', 'exclude_ins2ins', 'exclude_ins2cls', 'exp_triplet_ins2cls']
    CHECKPOINTS = ['best_2c_net.pth', 'best_3c_net.pth', 'best_4c_net.pth']
    N_FOLDS = 5
    class_names = {0: 'CN', 1: 'sMCI', 2: 'pMCI', 3: 'AD'}

    print("Pre-loading Test datasets...")
    test_loaders = {}
    for fold in range(1, N_FOLDS + 1):
        dataset = Dataset(mode="test", data_dir=opt.data_dir, seed=42, kfold=N_FOLDS, current_fold=fold, return_4c=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        test_loaders[fold] = loader

    for variant in LOSS_VARIANTS:
        for ckpt_name in CHECKPOINTS:
            for fold in range(1, N_FOLDS + 1):
                ckpt_path = os.path.join(opt.checkpoints_dir, f"ablation_loss_{variant}_fold{fold}", ckpt_name)
                
                if not os.path.exists(ckpt_path):
                    continue
                    
                print(f"Extracting: {variant} | {ckpt_name} | Fold {fold}")
                
                model = resnet18(class_num=3)
                state_dict = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)
                
                if 'prototypes' in state_dict:
                    model.prototypes = state_dict['prototypes'].to(device)
                else:
                    model.prototypes = model.prototypes.to(device)
                    
                model.to(device)
                model.eval()
                
                proto_cn = model.prototypes[0]
                proto_ad = model.prototypes[2]
                
                loader = test_loaders[fold]
                fold_severities = []
                fold_labels = []
                
                with torch.no_grad():
                    for batch in loader:
                        imgs = batch[0].to(device)
                        labels_4c = batch[2].numpy()
                        x_ori, _, _ = model(imgs)
                        severity = compute_1d_severity(x_ori, proto_cn, proto_ad)
                        fold_severities.extend(severity.tolist())
                        fold_labels.extend(labels_4c.tolist())
                
                # Save this specific fold's data to a tiny CSV
                df = pd.DataFrame({
                    'Severity': fold_severities,
                    'True Label': [class_names[lbl] for lbl in fold_labels]
                })
                csv_name = f"{variant}_{ckpt_name.split('.')[0]}_fold{fold}.csv"
                df.to_csv(os.path.join(opt.out_dir, csv_name), index=False)
                print(f"  -> Saved {csv_name}")

if __name__ == '__main__':
    main()
