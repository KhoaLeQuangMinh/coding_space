import os
import argparse
import torch
import numpy as np

def analyze_prototype(ckpt_path):
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        return f"Error loading: {e}"
        
    prototypes = None
    if 'prototypes' in checkpoint:
        prototypes = checkpoint['prototypes']
    elif 'state_dict' in checkpoint and 'prototypes' in checkpoint['state_dict']:
        prototypes = checkpoint['state_dict']['prototypes']
        
    if prototypes is None:
        return "No prototypes key found"
        
    if isinstance(prototypes, torch.Tensor):
        prototypes = prototypes.detach().numpy()
        
    num_classes, dim = prototypes.shape
    norms = np.linalg.norm(prototypes, axis=1)
    
    # Safe division: replace zero norms with 1.0 to avoid division by zero warnings
    safe_norms = np.where(norms == 0, 1.0, norms)
    protos_norm = prototypes / safe_norms[:, np.newaxis]
    cosine_sim = np.dot(protos_norm, protos_norm.T)
    
    off_diag_vals = []
    for i in range(num_classes):
        for j in range(i+1, num_classes):
            off_diag_vals.append(cosine_sim[i, j])
            
    avg_sim = np.mean(off_diag_vals) if off_diag_vals else 0.0
    
    if avg_sim > 0.95:
        status = "COLLAPSED"
    elif avg_sim < 0.1:
        status = "ORTHOGONAL"
    else:
        status = "GOOD"
        
    norms_str = ", ".join([f"{n:.2f}" for n in norms])
    return {
        'shape': f"{num_classes}x{dim}",
        'norms': norms_str,
        'avg_sim': avg_sim,
        'status': status
    }

def main():
    parser = argparse.ArgumentParser(description="Check learned prototypes across all checkpoints in a directory.")
    parser.add_argument('--checkpoints_dir', type=str, default='/kaggle/working/checkpoints', help='Root checkpoints directory')
    opt = parser.parse_args()

    root_dir = opt.checkpoints_dir
    if not os.path.exists(root_dir):
        # Fallback to local checkpoints if they exist
        root_dir = './checkpoints'
        if not os.path.exists(root_dir):
            print(f"Error: Directory not found at: {opt.checkpoints_dir}")
            return
            
    print(f"Scanning checkpoints recursively in: {root_dir}...")
    
    records = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.pth'):
                path = os.path.join(root, f)
                res = analyze_prototype(path)
                
                # Extract relative folder names for classification
                rel_path = os.path.relpath(path, root_dir)
                parts = rel_path.split(os.sep)
                folder = parts[0] if len(parts) > 1 else ""
                
                if isinstance(res, dict):
                    records.append({
                        'rel_path': rel_path,
                        'folder': folder,
                        'file': f,
                        'shape': res['shape'],
                        'norms': res['norms'],
                        'avg_sim': res['avg_sim'],
                        'status': res['status']
                    })

    if len(records) == 0:
        print("No PyTorch checkpoints (.pth) with learned prototypes found.")
        return
        
    # Print formatted output
    print(f"\nFound {len(records)} checkpoints with prototypes. Summary:\n")
    print(f"{'Folder / Loss Variant':<50} | {'Model File':<16} | {'Shape':<8} | {'Norms':<12} | {'Avg CosSim':<10} | {'Status':<11}")
    print("-" * 120)
    
    # Sort records by folder and file name
    records_sorted = sorted(records, key=lambda x: (x['folder'], x['file']))
    
    for r in records_sorted:
        print(f"{r['folder']:<50} | {r['file']:<16} | {r['shape']:<8} | {r['norms']:<12} | {r['avg_sim']:10.4f} | {r['status']:<11}")

if __name__ == "__main__":
    main()
