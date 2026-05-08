import matplotlib.pyplot as plt
import numpy as np
import torch


def check_volume(tensor):
    """
    pet_tensor: A torch.Tensor of shape (1, H, W, D) or (H, W, D)
    """
    # Remove the channel dimension if it's there (e.g., shape 1, 128, 128, 64 -> 128, 128, 64)
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
        
    # Move to CPU and convert to numpy
    data = tensor.detach().cpu().numpy()
    
    # Get middle slice indices
    mid_idx = [dim // 2 for dim in data.shape]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Axial View (H, W)
    axes[0].imshow(np.rot90(data[:, :, mid_idx[2]]), cmap='hot')
    axes[0].set_title(f'Axial - Slice {mid_idx[2]}')
    
    # Sagittal View (W, D)
    axes[1].imshow(np.rot90(data[mid_idx[0], :, :]), cmap='hot')
    axes[1].set_title(f'Sagittal - Slice {mid_idx[0]}')
    
    # Coronal View (H, D)
    axes[2].imshow(np.rot90(data[:, mid_idx[1], :]), cmap='hot')
    axes[2].set_title(f'Coronal - Slice {mid_idx[1]}')
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()
