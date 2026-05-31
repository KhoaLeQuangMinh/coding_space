import torch
import torch.nn as nn
from src.ranking import RankLoss
from src.basic_computing import BasicComputing

class HopeLossCriterion(nn.Module):
    def __init__(self, class_num=3, lambda_val=1.0):
        super(HopeLossCriterion, self).__init__()
        self.class_num = class_num
        self.basic_computing = BasicComputing(class_num=class_num)
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_rank = RankLoss(lambda_val)

    def forward(self, features, outputs, labels, current_epoch, total_epochs):
        """
        Calculates the combined HOPE loss.
        """
        device = features.device

        # CE loss
        loss_ce = self.criterion_ce(outputs, labels)

        # Basic computing — returns (compactness, separation, mus[K, D])
        compactness_loss, separation_loss, mus = self.basic_computing(features, labels)

        # Hybrid-granularity ordinal loss
        loss_ins2ins = self.criterion_rank(features, labels.float())  # instance-to-instance
        loss_ins2cls = compactness_loss / features.shape[1]           # instance-to-class
        
        # Determine ordinal targets based on actual present classes
        present_classes = torch.unique(labels).float().to(device)
        
        # class-to-class: requires >= 2 classes present
        if len(present_classes) > 1 and separation_loss > 0:
            loss_cls2cls = features.shape[1] / separation_loss + self.criterion_rank(mus, present_classes)
        else:
            loss_cls2cls = torch.tensor(0.0, device=device)

        loss_hyb = loss_ins2ins + loss_ins2cls + loss_cls2cls

        # Total loss with epoch-dependent weighting
        lambda_hyb = current_epoch * (1.0 / total_epochs)
        total_loss = loss_ce + lambda_hyb * loss_hyb

        return total_loss, {
            "ce": loss_ce.item(),
            "ins2ins": loss_ins2ins.item() if torch.is_tensor(loss_ins2ins) else loss_ins2ins,
            "ins2cls": loss_ins2cls.item() if torch.is_tensor(loss_ins2cls) else float(loss_ins2cls),
            "cls2cls": loss_cls2cls.item() if torch.is_tensor(loss_cls2cls) else float(loss_cls2cls),
            "hyb": loss_hyb.item() if torch.is_tensor(loss_hyb) else float(loss_hyb),
        }
