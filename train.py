from src.data import MRIPETDataset
from torch.utils.data import DataLoader
import torch
from torch.utils.data import DataLoader, random_split
from src.utils import read_config
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.baseline_model import BaselineModel
from src.fusion_modules import ConcatFusion, SumFusion, FiLM, GatedFusion, CrossAttention, ClinicalGuideCrossAttention
from src.engine import train_end_to_end
from src.engine import test_model


config = read_config("configs/kaggle.yaml")
config_experiment = read_config("configs/baseline.yaml")
dataset = MRIPETDataset(root=config["data"]["root"])
generator = torch.Generator().manual_seed(12345)

# 1. Define your split sizes (e.g., 70% train, 10% val, 20% test)
train_size = int(config["split"]["train_ratio"] * len(dataset))
val_size = int(config["split"]["val_ratio"] * len(dataset))
test_size = len(dataset) - train_size - val_size

# 2. Perform the random split
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=generator)

# 3. Create the DataLoaders
train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=config["training"]["num_workers"])
val_loader   = DataLoader(val_ds,   batch_size=config["training"]["batch_size"], shuffle=False, num_workers=config["training"]["num_workers"])
test_loader  = DataLoader(test_ds,  batch_size=config["training"]["batch_size"], shuffle=False, num_workers=config["training"]["num_workers"])

# Quick check
print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

model = train_end_to_end(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    config=config_experiment
)

results = test_model(
    test_loader=test_loader,
    model_path=f"[{config_experiment['experiment']['name']}].pth",
    config=config_experiment
)