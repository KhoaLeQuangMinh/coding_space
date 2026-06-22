import os
import torch
import json
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Extract only prototypes from PyTorch checkpoints and save them to a single tiny file.")
    parser.add_argument('--checkpoints_dir', type=str, default='/kaggle/working/checkpoints', help='Directory containing trained checkpoints')
    parser.add_argument('--out_file', type=str, default='extracted_prototypes.json', help='Output JSON file path')
    args = parser.parse_args()

    root_dir = args.checkpoints_dir
    if not os.path.exists(root_dir):
        # Fallback to local checkpoints if run locally
        root_dir = './checkpoints'
        if not os.path.exists(root_dir):
            print(f"Error: Directory not found at '{args.checkpoints_dir}' or '{root_dir}'")
            return

    print(f"Scanning checkpoints recursively in: {root_dir}...")
    extracted = {}

    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.pth'):
                ckpt_path = os.path.join(root, f)
                rel_path = os.path.relpath(ckpt_path, root_dir)
                
                try:
                    # Load checkpoint on CPU to avoid CUDA dependency
                    checkpoint = torch.load(ckpt_path, map_location='cpu')
                    
                    prototypes = None
                    if 'prototypes' in checkpoint:
                        prototypes = checkpoint['prototypes']
                    elif 'state_dict' in checkpoint and 'prototypes' in checkpoint['state_dict']:
                        prototypes = checkpoint['state_dict']['prototypes']
                        
                    if prototypes is not None:
                        if isinstance(prototypes, torch.Tensor):
                            proto_list = prototypes.detach().numpy().tolist()
                        else:
                            proto_list = np.array(prototypes).tolist()
                            
                        # Use relative path as the key (e.g. "ablation_loss_full_fold1/best_3c_net.pth")
                        extracted[rel_path] = proto_list
                        print(f"  [Extracted] {rel_path} (Shape: {len(proto_list)}x{len(proto_list[0])})")
                except Exception as e:
                    print(f"  [Failed] {rel_path}: {e}")

    if not extracted:
        print("No prototypes found in any of the checkpoints.")
        return

    # Save to JSON file
    with open(args.out_file, 'w') as out_f:
        json.dump(extracted, out_f, indent=2)
        
    file_size_kb = os.path.getsize(args.out_file) / 1024.0
    print(f"\nSuccessfully extracted prototypes from {len(extracted)} checkpoints!")
    print(f"Saved to: {os.path.abspath(args.out_file)} ({file_size_kb:.2f} KB)")
    print("\nYou can now download this single tiny file from Kaggle instead of downloading gigabytes of checkpoints!")

if __name__ == '__main__':
    main()
