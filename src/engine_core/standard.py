import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score
from src.engine_core.base import BaseEngine

class StandardEngine(BaseEngine):
    def __init__(self, criterion, decode_fn):
        self.criterion = criterion
        self.decode_fn = decode_fn

    def train_one_epoch(self, model, loader, optimizer, args, current_epoch=None, total_epochs=None):
        model.train()
        total_loss = 0.0

        bar = tqdm(loader, desc="Train", leave=False)
        for batch in bar:
            mri    = batch["mri"].to(args.device)
            pet    = batch["pet"].to(args.device)

            if args.loss == "mse":
                labels = batch["label"].float().to(args.device)
            else:
                labels = batch["label"].to(args.device)

            optimizer.zero_grad()
            outputs = model(mri, pet)
            if args.loss == "mse":
                outputs = outputs.squeeze(1)   # (B, 1) -> (B,) to match float label shape
            loss    = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(loader)

    def evaluate(self, model, loader, args):
        model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                mri    = batch["mri"].to(args.device)
                pet    = batch["pet"].to(args.device)

                if args.loss == "mse":
                    labels = batch["label"].float().to(args.device)
                else:
                    labels = batch["label"].to(args.device)

                outputs = model(mri, pet)
                if args.loss == "mse":
                    outputs = outputs.squeeze(1)   # (B, 1) -> (B,)
                loss    = self.criterion(outputs, labels)
                total_loss += loss.item()

                preds = self.decode_fn(outputs, args)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(
                    labels.long().cpu().numpy()
                    if args.loss == "mse"
                    else labels.cpu().numpy()
                )

        avg_loss   = total_loss / len(loader)
        acc        = accuracy_score(all_labels, all_preds)
        macro_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        macro_rec  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        return avg_loss, acc, macro_f1, macro_rec
