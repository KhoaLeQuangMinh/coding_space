import torch
import torch.nn.functional as F
import nibabel as nib
from scipy.ndimage import zoom
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import os
import random

LABEL_MAP = {"CN": 0, "sMCI": 1, "pMCI": 2, "AD": 3}
HOPE_LABEL_MAP = {"CN": 0, "sMCI": 1, "pMCI": 1, "AD": 2}
TARGET_SHAPE = (128, 128, 128)  # model expected input


def resize_volume(volume: np.ndarray, target_shape=TARGET_SHAPE) -> np.ndarray:
    """
    Resize a 3D volume to target_shape using PyTorch's C++ vectorized backend (trilinear).
    This is ~5x to 10x faster than SciPy's zoom on the CPU, preventing DataLoader starvation.
    """
    if volume.shape == target_shape:
        return volume.astype(np.float32)
    # Convert to PyTorch tensor [1, 1, D, H, W]
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
    # Trilinear interpolation
    resized = F.interpolate(tensor, size=target_shape, mode='trilinear', align_corners=True)
    # Return as numpy array
    return resized.squeeze(0).squeeze(0).numpy().astype(np.float32)


class MRIPETDataset(Dataset):
    def __init__(self, root, mri_transform=None, pet_transform=None, merge_mci=False):
        self.root = root
        # Filter to only .npz files — avoids .DS_Store, .gitkeep, etc.
        self.subjects = sorted([f for f in os.listdir(root) if f.endswith('.npz')])
        self.mri_transform = mri_transform
        self.pet_transform = pet_transform
        self.merge_mci = merge_mci

        # Pre-load and cache all labels once during init to avoid slow Disk I/O calls later
        print(f"Loading and caching {len(self.subjects)} labels from disk...")
        labels = []
        for subj in self.subjects:
            sample = np.load(os.path.join(self.root, subj))
            string_label = sample["label"].item()
            label = HOPE_LABEL_MAP[string_label] if self.merge_mci else LABEL_MAP[string_label]
            labels.append(label)
        self._cached_labels = np.array(labels)

    def __len__(self):
        return len(self.subjects)

    def get_labels(self):
        """Return cached integer labels (extremely fast)."""
        return self._cached_labels

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

        mri_volume = np.expand_dims(mri_volume.astype(np.float32), axis=0)
        pet_volume = np.expand_dims(pet_volume.astype(np.float32), axis=0)

        label = HOPE_LABEL_MAP[string_label] if self.merge_mci else LABEL_MAP[string_label]

        if self.mri_transform:
            mri_volume = self.mri_transform(mri_volume)
        if self.pet_transform:
            pet_volume = self.pet_transform(pet_volume)

        output = {
            'mri': torch.from_numpy(mri_volume),
            'pet': torch.from_numpy(pet_volume),
            'label': torch.tensor(label, dtype=torch.long),
            'subject_id': ptid,
            'original_label': string_label,
        }

        return output


class MockDataset(Dataset):
    """Generates random 128x128x128 volumes for local testing without real data."""
    def __init__(self, size=40, merge_mci=False):
        self.size = size
        self.merge_mci = merge_mci
        # Pre-generate labels and original_label strings
        self._labels = []
        self._original_labels = []
        for _ in range(size):
            if self.merge_mci:
                lbl = random.choice([0, 1, 2])
                self._labels.append(lbl)
                # For MCI (label=1), randomly assign sMCI or pMCI as original
                if lbl == 0:
                    self._original_labels.append("CN")
                elif lbl == 1:
                    self._original_labels.append(random.choice(["sMCI", "pMCI"]))
                else:
                    self._original_labels.append("AD")
            else:
                lbl = random.choice([0, 1, 2, 3])
                self._labels.append(lbl)
                self._original_labels.append(["CN", "sMCI", "pMCI", "AD"][lbl])

    def __len__(self):
        return self.size

    def get_labels(self):
        return np.array(self._labels)

    def __getitem__(self, idx):
        return {
            'mri': torch.randn(1, *TARGET_SHAPE),
            'pet': torch.randn(1, *TARGET_SHAPE),
            'label': torch.tensor(self._labels[idx], dtype=torch.long),
            'subject_id': f"mock_{idx:03d}",
            'original_label': self._original_labels[idx],
        }


class HopeBatchSampler(Sampler):
    """
    Custom batch sampler for HOPE training.
    Ensures every batch contains CN, MCI, and AD in roughly 1:2:1 ratio
    (e.g. batch_size=8 → 2 CN, 4 MCI, 2 AD).
    """
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.batch_size = batch_size

        self.idx_cn  = np.where(self.labels == 0)[0].tolist()
        self.idx_mci = np.where(self.labels == 1)[0].tolist()
        self.idx_ad  = np.where(self.labels == 2)[0].tolist()

        self.n_cn  = max(1, batch_size // 4)
        self.n_ad  = max(1, batch_size // 4)
        self.n_mci = batch_size - self.n_cn - self.n_ad

        self.num_batches = len(self.labels) // batch_size

    def __iter__(self):
        # Shuffle copies at the start of each epoch
        cn_pool  = self.idx_cn.copy();  random.shuffle(cn_pool)
        mci_pool = self.idx_mci.copy(); random.shuffle(mci_pool)
        ad_pool  = self.idx_ad.copy();  random.shuffle(ad_pool)

        cn_iter  = self._cycle(cn_pool)
        mci_iter = self._cycle(mci_pool)
        ad_iter  = self._cycle(ad_pool)

        for _ in range(self.num_batches):
            batch = []
            batch.extend([next(cn_iter)  for _ in range(self.n_cn)])
            batch.extend([next(mci_iter) for _ in range(self.n_mci)])
            batch.extend([next(ad_iter)  for _ in range(self.n_ad)])
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches

    @staticmethod
    def _cycle(lst):
        """Infinite iterator that reshuffles when exhausted."""
        saved = list(lst)
        while True:
            for element in saved:
                yield element
            random.shuffle(saved)