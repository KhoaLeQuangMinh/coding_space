import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from utils.Dataset import Dataset
from models.Resnet import resnet18
from utils.config_loader import load_config, get_variant_params

def compute_1d_severity(features, proto_CN, proto_AD):
    feat_norm = F.normalize(features, p=2, dim=1)
    cn_norm = F.normalize(proto_CN.unsqueeze(0), p=2, dim=1)
    ad_norm = F.normalize(proto_AD.unsqueeze(0), p=2, dim=1)
    
    sim_to_cn = torch.matmul(feat_norm, cn_norm.T).squeeze(1)
    sim_to_ad = torch.matmul(feat_norm, ad_norm.T).squeeze(1)
    
    severity = sim_to_ad - sim_to_cn
    return severity.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Extract and save latent features for visualization/t-SNE.")
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/datasets/kisokoghan/paired-npz/paired_npz', help='Path to NPZ data')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='path to checkpoints')
    parser.add_argument('--out_dir', type=str, default='./analysis_output/extracted_features', help='where to save CSVs')
    parser.add_argument('--kfold', type=int, default=5, help='Number of folds used during training')
    parser.add_argument('--specific_fold', type=int, default=-1, help='Extract features for only this specific fold (1-based)')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of classes for classification (e.g., 3 or 4)')
    parser.add_argument('--target_loss', type=str, default=None, help='Specific loss variant to run (e.g., ce, full, exp_triplet_ins2cls)')
    parser.add_argument('--triplet_margin', type=float, default=0.3, help='Margin for triplet relative losses')
    parser.add_argument('--m', type=float, default=0.9, help='EMA momentum')
    parser.add_argument('--variant', type=str, default=None, help='Variant key from pipeline_config.json (overrides manual args)')
    parser.add_argument('--config', type=str, default=None, help='Path to pipeline_config.json')
    opt = parser.parse_args()

    # Resolve variant from config if provided
    no_classifier = False
    if opt.variant is not None:
        config = load_config(opt.config)
        params = get_variant_params(opt.variant, config)
        opt.target_loss = params.get('loss_type', params.get('target_loss', opt.target_loss))
        opt.num_classes = params.get('num_classes', opt.num_classes)
        opt.triplet_margin = params.get('triplet_margin', opt.triplet_margin)
        opt.m = params.get('m', opt.m)
        no_classifier = params.get('no_classifier', False)

    if opt.target_loss is None:
        parser.error("--target_loss is required (or use --variant to load from config)")
    if opt.num_classes is None:
        parser.error("--num_classes is required (or use --variant to load from config)")

    os.makedirs(opt.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    variant = opt.target_loss
    class_num = opt.num_classes
    name_suffix = "_4class" if class_num == 4 else ""
    ema_suffix = f"_ema{opt.m}" if opt.m != 0.9 else ""
    margin_suffix = f"_margin{opt.triplet_margin}" if opt.triplet_margin != 0.3 else ""
    proto_suffix = "_proto" if no_classifier else ""
    prefix = "ablation_loss"
    
    CHECKPOINTS = ['best_2c_net.pth', 'best_3c_net.pth', 'best_4c_net.pth']
    N_FOLDS = opt.kfold
    class_names = {0: 'CN', 1: 'sMCI', 2: 'pMCI', 3: 'AD'}

    folds_to_run = range(1, N_FOLDS + 1) if opt.specific_fold == -1 else [opt.specific_fold]
    
    for fold in folds_to_run:
        print(f"\nProcessing Fold {fold}...")
        dataset = Dataset(mode="test", data_dir=opt.data_dir, seed=42, kfold=N_FOLDS, current_fold=fold, return_4c=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        
        for ckpt_name in CHECKPOINTS:
            ckpt_path = os.path.join(opt.checkpoints_dir, f"{prefix}_{variant}{name_suffix}{ema_suffix}{margin_suffix}{proto_suffix}_fold{fold}", ckpt_name)
            
            if not os.path.exists(ckpt_path):
                print(f"Skipping {ckpt_name} (Not found: {ckpt_path})")
                continue
            
            print(f"Extracting: {variant}{name_suffix} | {ckpt_name}")
            
            model = resnet18(spatial_size=128, sample_duration=128, num_classes=class_num, m=0.99, no_classifier=no_classifier)
            state_dict = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            
            if 'prototypes' in state_dict:
                model.prototypes = state_dict['prototypes'].to(device)
            else:
                model.prototypes = model.prototypes.to(device)
                
            model.to(device)
            model.eval()
            
            proto_cn = model.prototypes[0]
            proto_ad = model.prototypes[class_num - 1]
            
            fold_severities = []
            fold_labels = []
            fold_features = []
            
            with torch.no_grad():
                for batch in loader:
                    imgs = batch[0].to(device)
                    labels_4c = batch[2].numpy()
                    x_ori, _, _ = model(imgs)
                    severity = compute_1d_severity(x_ori, proto_cn, proto_ad)
                    fold_severities.extend(severity.tolist())
                    fold_labels.extend(labels_4c.tolist())
                    fold_features.extend(x_ori.cpu().detach().numpy().tolist())
            
            df_data = {
                'Severity': fold_severities,
                'True Label': [class_names[lbl] for lbl in fold_labels]
            }
            
            fold_features_np = np.array(fold_features)
            for i in range(fold_features_np.shape[1]):
                df_data[f'feature_{i}'] = fold_features_np[:, i]
                
            df = pd.DataFrame(df_data)
            csv_name = f"{variant}{name_suffix}{ema_suffix}{margin_suffix}{proto_suffix}_{ckpt_name.split('.')[0]}_fold{fold}.csv"
            df.to_csv(os.path.join(opt.out_dir, csv_name), index=False)
            print(f"  -> Saved {csv_name}")

if __name__ == '__main__':
    main()
