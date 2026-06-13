import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import KFold

# Train: CN=0, MCI=1, AD=2
# Valid/Test: CN=0, sMCI=1, pMCI=2, AD=3
TRAIN_MAP = {"CN": 0, "sMCI": 1, "pMCI": 1, "AD": 2}
EVAL_MAP = {"CN": 0, "sMCI": 1, "pMCI": 2, "AD": 3}
TARGET_SHAPE = (128, 128, 128)

def resize_volume_fast(volume: np.ndarray, target_shape=TARGET_SHAPE) -> np.ndarray:
    if volume.shape == target_shape:
        return volume.astype(np.float32)
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=target_shape, mode='trilinear', align_corners=True)
    return resized.squeeze(0).squeeze(0).numpy().astype(np.float32)

class Dataset(TorchDataset):
    def __init__(self, mode="total", data_dir="data", seed=42, kfold=5, current_fold=1, return_4c=False):
        self.mode = mode
        self.return_4c = return_4c
        
        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
        random.seed(seed)
        random.shuffle(all_files)
        
        if kfold > 1:
            kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
            splits = list(kf.split(all_files))
            trainval_idx, test_idx = splits[current_fold - 1]
            test_files = [all_files[i] for i in test_idx]
            trainval_files = [all_files[i] for i in trainval_idx]
            
            val_size = int(len(all_files) * 0.1)
            valid_files = trainval_files[:val_size]
            train_files = trainval_files[val_size:]
        else:
            total = len(all_files)
            train_end = int(total * 0.7)
            val_end = int(total * 0.8)
            train_files = all_files[:train_end]
            valid_files = all_files[train_end:val_end]
            test_files = all_files[val_end:]

        self.imgs = []
        
        def filter_files(files, allowed_labels):
            filtered = []
            for f in files:
                path = os.path.join(data_dir, f)
                sample = np.load(path)
                label = sample["label"].item()
                if label in allowed_labels:
                    filtered.append((path, label))
            return filtered

        if mode == "total_cn":
            self.imgs = filter_files(train_files, ["CN"])
        elif mode == "total_ad":
            self.imgs = filter_files(train_files, ["AD"])
        elif mode == "total_mci":
            self.imgs = filter_files(train_files, ["sMCI", "pMCI"])
        elif mode == "valid":
            self.imgs = filter_files(valid_files, ["CN", "sMCI", "pMCI", "AD"])
        elif mode == "test":
            self.imgs = filter_files(test_files, ["CN", "sMCI", "pMCI", "AD"])

    def __getitem__(self, index):
        data_path, string_label = self.imgs[index]
        
        sample = np.load(data_path)
        mwp1 = sample["mwp1"]
        mwp1 = np.nan_to_num(mwp1, nan=0.0)
        mwp1 = resize_volume_fast(mwp1, TARGET_SHAPE)
        A = torch.from_numpy(mwp1).type(torch.FloatTensor)
        
        if self.mode in ["valid", "test"]:
            value = EVAL_MAP[string_label]
            if self.return_4c:
                return A.unsqueeze(0), value, value # For valid/test, 3c and 4c are both handled downstream if needed, though typically unused
            return A.unsqueeze(0), value
        else:
            value = TRAIN_MAP[string_label]
            if self.return_4c:
                value_4c = EVAL_MAP[string_label]
                return A.unsqueeze(0), value, value_4c
            return A.unsqueeze(0), value

    def __len__(self):
        return len(self.imgs)
