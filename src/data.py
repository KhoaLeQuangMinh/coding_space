import torch
import nibabel as nib
from scipy.ndimage import zoom
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

 
LABEL_MAP = {"CN": 0, "sMCI": 1, "pMCI": 2, "AD": 3}
TARGET_SHAPE = (128, 128, 128)  # model expected input
 
 
def resize_volume(volume: np.ndarray, target_shape=TARGET_SHAPE) -> np.ndarray:
    """
    Resize a 3D volume to target_shape using cubic spline interpolation (order=3).
 
    Why order=3 (cubic spline)?
    - order=0 (nearest): fastest but blocky — loses structural detail.
    - order=1 (trilinear): fast but blurs fine anatomical features.
    - order=3 (cubic spline): best quality/speed trade-off for MRI/PET;
      preserves edges and intensity gradients; recommended by SciPy docs
      and standard in neuroimaging preprocessing pipelines.
    - order=5: slightly sharper but much slower with negligible gain.
 
    Approach: compute per-axis zoom factors from actual vs. target shape,
    then apply scipy.ndimage.zoom. This preserves the full field-of-view
    rather than cropping, which is important when the subject brain may be
    positioned differently across scans.
    """
    current_shape = np.array(volume.shape, dtype=float)
    zoom_factors = np.array(target_shape, dtype=float) / current_shape
    resized = zoom(volume, zoom=zoom_factors, order=3, prefilter=True)
    return resized.astype(np.float32)

class MRIPETDataset(Dataset):
    def __init__(self, root, mri_transform=None, pet_transform = None):
        # Load the dataframe and take the first N subjects
        self.root = root
        self.subjects = os.listdir(root)
        self.mri_transform = mri_transform
        self.pet_transform = pet_transform

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        sample = np.load(os.path.join(self.root, self.subjects[idx]))
        
        string_label = sample["label"].item()
        mri_volume = sample["mwp1"]
        pet_volume = sample["nPET"]
        ptid = sample["ptid"].item()

        mri_volume = np.nan_to_num(mri_volume, nan=0.0)
        pet_volume = np.nan_to_num(pet_volume, nan=0.0)

        mri_volume = resize_volume(mri_volume, TARGET_SHAPE)
        pet_volume = resize_volume(pet_volume, TARGET_SHAPE)

        label = LABEL_MAP[string_label]

        if self.mri_transform:
            mri_volume = self.mri_transform(mri_volume)
        if self.pet_transform:
            pet_volume = self.pet_transform(pet_volume)

        output = {
            'mri': torch.from_numpy(mri_volume),
            'pet': torch.from_numpy(pet_volume),
            'label': torch.tensor(label, dtype=torch.long),
            'subject_id': ptid
        }

        return output