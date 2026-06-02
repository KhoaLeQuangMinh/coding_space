import sys
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import nibabel as nib
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# IMPORT pySuStaIn FROM THE CLONED REPOSITORY
# ---------------------------------------------------------
# We dynamically add the SOFTX directory to the Python path
# so we don't modify the original code at all.
SUSTAIN_REPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'SOFTX-D-21-00098')
sys.path.append(SUSTAIN_REPO_DIR)

from kde_ebm.mixture_model import fit_all_gmm_models
from pySuStaIn.MixtureSustain import MixtureSustain
from kde_ebm import plotting

# ---------------------------------------------------------
# DATA RESIZING LOGIC
# ---------------------------------------------------------
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

def extract_aal_features(data_dir, aal_path):
    """
    Overlays the AAL template on the 3D MRIs and extracts the mean 
    grey matter density per brain region for every subject.
    """
    print(f"Loading AAL template from {aal_path}...")
    aal_img = nib.load(aal_path)
    aal_data = aal_img.get_fdata()
    
    # Resize AAL template using Nearest Neighbor to preserve discrete integer labels
    aal_tensor = torch.from_numpy(aal_data).unsqueeze(0).unsqueeze(0).float()
    aal_resized = F.interpolate(aal_tensor, size=TARGET_SHAPE, mode='nearest').squeeze(0).squeeze(0).numpy()
    
    region_ids = np.unique(aal_resized)
    region_ids = region_ids[region_ids > 0] # Remove background (label 0)
    region_ids = np.sort(region_ids)
    
    print(f"Found {len(region_ids)} regions in the AAL template.")
    
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
    features_list = []
    labels_list_4class = []
    subj_ids = []
    
    print(f"Extracting regional volumes for {len(files)} files...")
    for f in tqdm(files):
        path = os.path.join(data_dir, f)
        sample = np.load(path, allow_pickle=True)
        string_label = sample["label"].item()
        
        # 4-class labels
        if string_label == "CN":     lbl_4 = 0
        elif string_label == "sMCI": lbl_4 = 1
        elif string_label == "pMCI": lbl_4 = 2
        elif string_label == "AD":   lbl_4 = 3
        else: continue
            
        mwp1 = sample["mwp1"]
        mwp1 = np.nan_to_num(mwp1, nan=0.0)
        mwp1 = resize_volume_fast(mwp1, TARGET_SHAPE)
        
        # Calculate mean grey matter density per AAL region
        regional_volumes = []
        for rid in region_ids:
            mask = (aal_resized == rid)
            if np.sum(mask) > 0:
                mean_vol = np.mean(mwp1[mask])
            else:
                mean_vol = 0.0
            regional_volumes.append(mean_vol)
            
        features_list.append(regional_volumes)
        labels_list_4class.append(lbl_4)
        subj_ids.append(f.split('.')[0])
        
    X = np.array(features_list)
    y = np.array(labels_list_4class)
    return X, y, subj_ids, region_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='path to npz files')
    parser.add_argument('--aal_path', type=str, required=True, help='path to AAL template .nii file')
    parser.add_argument('--out_dir', type=str, default='./sustain_output', help='where to save SuStaIn outputs')
    parser.add_argument('--n_subtypes', type=int, default=3, help='Maximum number of subtypes (N_S_max)')
    opt = parser.parse_args()

    os.makedirs(opt.out_dir, exist_ok=True)
    
    # 1. Feature Extraction
    X, y, subj_ids, region_ids = extract_aal_features(opt.data_dir, opt.aal_path)
    print(f"\nExtracted Feature Matrix Shape: {X.shape} (Subjects x Biomarkers)")
    
    # 2. Prepare SuStaIn Labels
    # SuStaIn expects: 0 for healthy controls, 1 for diseased cases, 2 for intermediate (ignored during GMM fitting)
    sustain_labels = np.ones(X.shape[0], dtype=int) * 2 
    sustain_labels[y == 0] = 0 # CN -> Control (0)
    sustain_labels[y == 3] = 1 # AD -> Case (1)
    
    data_case_control = X[sustain_labels != 2, :]
    labels_case_control = sustain_labels[sustain_labels != 2]
    
    # Fit Gaussian Mixture Models to determine Normal vs Abnormal distributions per biomarker
    print("\nFitting GMMs on CN and AD groups for each biomarker...")
    mixtures = fit_all_gmm_models(X, sustain_labels)
    
    biomarker_names = [f"Region_{int(rid)}" for rid in region_ids]
    
    print("Plotting GMM fits (CN vs AD) per biomarker...")
    fig, ax = plotting.mixture_model_grid(data_case_control, labels_case_control, mixtures, biomarker_names)
    fig.savefig(os.path.join(opt.out_dir, 'gmm_fits.png'))
    plt.close(fig)
    
    # Calculate likelihood matrices (Probability of being Abnormal vs Normal) for ALL subjects
    print("\nCalculating likelihood matrices (L_yes, L_no) for all patients...")
    L_yes = np.zeros(X.shape)
    L_no = np.zeros(X.shape)
    for i in range(len(region_ids)):
        L_no[:, i], L_yes[:, i] = mixtures[i].pdf(None, X[:, i])
        
    # 3. Initialize and Run SuStaIn
    dataset_name = 'alzheimers_aal'
    N_startpoints = 25
    N_S_max = opt.n_subtypes
    N_iterations_MCMC = int(1e5)
    use_parallel_startpoints = True
    
    print(f"\nInitializing MixtureSustain with {N_S_max} max subtypes...")
    sustain = MixtureSustain(L_yes, L_no, biomarker_names, N_startpoints, N_S_max, 
                             N_iterations_MCMC, opt.out_dir, dataset_name, use_parallel_startpoints)
                             
    print("Running SuStaIn MCMC Algorithm... (Note: This will take a significant amount of time)")
    samples_sequence, samples_f, ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype_stage = sustain.run_sustain_algorithm(plot=True)
    
    # 4. Save Subject-level Outputs
    df = pd.DataFrame()
    df['subj_id'] = subj_ids
    df['true_clinical_label'] = [['CN', 'sMCI', 'pMCI', 'AD'][lbl] for lbl in y]
    df['ml_subtype'] = ml_subtype
    df['prob_ml_subtype'] = prob_ml_subtype
    df['ml_stage'] = ml_stage
    df['prob_ml_stage'] = prob_ml_stage
    
    out_csv = os.path.join(opt.out_dir, 'sustain_subject_estimates.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nSaved subject-level subtype and stage estimates to {out_csv}")
    print("SuStaIn modeling completed successfully!")

if __name__ == '__main__':
    main()
