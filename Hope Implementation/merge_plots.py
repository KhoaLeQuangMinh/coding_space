import os
from PIL import Image

# Directories
tsne_dir = '/Users/khoale/Downloads/analysis_output_tSNE'
kde_dir = '/Users/khoale/Downloads/analysis_output_tSNE/plot_kde_folds'
out_dir = '/Users/khoale/Downloads/ablation_result/plots'

os.makedirs(out_dir, exist_ok=True)

VARIANTS_TO_PLOT = [
    'ce', 
    'ins2ins', 
    'ins2cls', 
    'full',
    'exclude_ins2ins',
    'exclude_ins2cls',
    'exp_triplet_ins2cls',
    'triplet_only',
    'hierarchical_triplet_only',
    'exp_hierarchical_triplet_ins2cls',
    'full_4class',
    'exp_triplet_ins2cls_4class',
    'hierarchical_triplet_only_4class',
    'qwk_hierarchical_triplet_4class',
    'exp_3pole_local',
    'exp_3pole_global'
]

CHECKPOINTS = ['best_2c_net', 'best_3c_net', 'best_4c_net']

print("Starting to merge t-SNE and KDE plots...")

for ckpt in CHECKPOINTS:
    for variant in VARIANTS_TO_PLOT:
        tsne_path = os.path.join(tsne_dir, f"tsne_{variant}_{ckpt}.png")
        kde_path = os.path.join(kde_dir, f"latent_kde_{variant}_{ckpt}.png")
        
        if not os.path.exists(tsne_path) or not os.path.exists(kde_path):
            print(f"Skipping {variant} {ckpt}: One or both plots missing.")
            continue
            
        # Load images
        img_tsne = Image.open(tsne_path)
        img_kde = Image.open(kde_path)
        
        # Determine the maximum height and total width for side-by-side merging
        # If one is taller than the other, we can pad with white or resize proportionally
        # For simplicity, let's just make them the exact same height by resizing the KDE plot
        # to match the t-SNE plot's height, keeping its aspect ratio.
        
        target_height = img_tsne.height
        aspect_ratio = img_kde.width / img_kde.height
        new_kde_width = int(target_height * aspect_ratio)
        
        img_kde_resized = img_kde.resize((new_kde_width, target_height), Image.Resampling.LANCZOS)
        
        # Create a new blank canvas (white background)
        total_width = img_tsne.width + new_kde_width
        merged_img = Image.new('RGB', (total_width, target_height), color=(255, 255, 255))
        
        # Paste the images
        merged_img.paste(img_tsne, (0, 0))
        merged_img.paste(img_kde_resized, (img_tsne.width, 0))
        
        # Save
        out_path = os.path.join(out_dir, f"merged_{variant}_{ckpt}.png")
        merged_img.save(out_path)
        print(f"  -> Saved merged plot: {out_path}")

print("\nFinished merging all plots!")
