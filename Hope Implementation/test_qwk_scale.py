import torch
import torch.nn.functional as F
from models.qwk_loss import DifferentiableQWKLoss

criterion = DifferentiableQWKLoss(num_classes=4)
# Random logits (approx 0) -> probs approx 0.25
logits = torch.randn(16, 4) * 0.1
targets = torch.randint(0, 4, (16,))
loss = criterion(logits, targets)
print(f"Random Guess Loss: {loss.item()}")

# Perfect logits
logits = F.one_hot(targets, num_classes=4).float() * 10.0
loss = criterion(logits, targets)
print(f"Perfect Prediction Loss: {loss.item()}")

# Worst logits (inverted)
worst_targets = (targets + 2) % 4
logits = F.one_hot(worst_targets, num_classes=4).float() * 10.0
loss = criterion(logits, targets)
print(f"Worst Prediction Loss: {loss.item()}")

