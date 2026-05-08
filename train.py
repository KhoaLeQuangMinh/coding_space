from src.data import MRIPETDataset
from torch.utils.data import DataLoader
import torch
from torch.utils.data import DataLoader, random_split
import yaml

with open("configs/kaggle.yaml", "r") as f:
    config = yaml.safe_load(f)

print(config)
dataset = MRIPETDataset(root=config["data"]["root"])

# 1. Define your split sizes (e.g., 70% train, 10% val, 20% test)
train_size = int(config["split"]["train_ratio"] * len(dataset))
val_size = int(config["split"]["val_ratio"] * len(dataset))
test_size = len(dataset) - train_size - val_size

# 2. Perform the random split
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

# 3. Create the DataLoaders
train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=config["training"]["num_workers"])
val_loader   = DataLoader(val_ds,   batch_size=config["training"]["batch_size"], shuffle=False, num_workers=config["training"]["num_workers"])
test_loader  = DataLoader(test_ds,  batch_size=config["training"]["batch_size"], shuffle=False, num_workers=config["training"]["num_workers"])

# Quick check
print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

