import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score
from src.engine_core.base import BaseEngine

class HopeEngine(BaseEngine):
    def __init__(self, criterion):
        self.criterion = criterion

    def train_one_epoch(self, model, loader, optimizer, args, current_epoch=1, total_epochs=40):
        model.train()
        total_loss = 0.0
        
        bar = tqdm(loader, desc="Train", leave=False)
        for batch in bar:
            mri    = batch["mri"].to(args.device)
            labels = batch["label"].to(args.device)

            optimizer.zero_grad()
            
            # Hope ResNet returns features, outputs, spmci_prob
            features, outputs, spmci_prob = model(mri)
            
            loss, loss_dict = self.criterion(features, outputs, labels, current_epoch, total_epochs)
            
            loss.backward()
            optimizer.step()
            
            # Online EMA prototype update
            if hasattr(model, 'update'):
                model.update(features, labels)

            total_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}", ce=f'{loss_dict["ce"]:.4f}')

        return total_loss / len(loader)

    def evaluate(self, model, loader, args):
        model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                mri    = batch["mri"].to(args.device)
                labels = batch["label"].to(args.device)

                features, outputs, spmci_prob = model(mri)
                # Ensure we don't scale by epoch during evaluation
                loss, _ = self.criterion(features, outputs, labels, args.epochs, args.epochs)
                total_loss += loss.item()

                _, preds = torch.max(outputs.data, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss   = total_loss / len(loader)
        acc        = accuracy_score(all_labels, all_preds)
        macro_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        macro_rec  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        return avg_loss, acc, macro_f1, macro_rec
