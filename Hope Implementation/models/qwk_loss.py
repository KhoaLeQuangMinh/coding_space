import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableQWKLoss(nn.Module):
    """
    A Differentiable Quadratic Weighted Kappa (QWK) Loss.
    Standard QWK is a discrete metric. To use it as a loss function, we:
    1. Output soft probabilities via Softmax.
    2. Compute the expected (soft) confusion matrix.
    3. Apply the quadratic penalty weight matrix.
    4. Compute actual errors and expected random errors to form the Kappa score.
    """
    def __init__(self, num_classes=4, epsilon=1e-8):
        super(DifferentiableQWKLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        
        # Precompute the quadratic weight matrix
        # penalty[i, j] = (i - j)^2 / (N - 1)^2
        indices = torch.arange(num_classes).float()
        weight_matrix = (indices.unsqueeze(1) - indices.unsqueeze(0)) ** 2
        weight_matrix = weight_matrix / ((num_classes - 1) ** 2)
        
        # Register as a buffer so it automatically moves to the correct device (e.g., GPU)
        self.register_buffer('weight_matrix', weight_matrix)

    def forward(self, logits, targets):
        """
        logits: [batch_size, num_classes] (raw scores from network)
        targets: [batch_size] (integer class labels)
        """
        # 1. Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # 2. Convert integer targets to one-hot probabilities
        # Ensure targets are on the same device
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # 3. Build soft confusion matrix (O)
        # O[i, j] is the sum of probabilities that true class is i and pred is j
        # Shape: [num_classes, num_classes]
        O = torch.matmul(targets_one_hot.t(), probs)
        
        # 4. Calculate marginal distributions
        # True class distribution
        actual_hist = targets_one_hot.sum(dim=0)
        # Predicted class distribution
        pred_hist = probs.sum(dim=0)
        
        # 5. Build expected random confusion matrix (E)
        # E[i, j] = (actual_i * pred_j) / batch_size
        batch_size = logits.size(0)
        E = torch.outer(actual_hist, pred_hist) / batch_size
        
        # Normalize matrices to sum to 1
        O_norm = O / (O.sum() + self.epsilon)
        E_norm = E / (E.sum() + self.epsilon)
        
        # 6. Apply weights and calculate Kappa
        numerator = torch.sum(self.weight_matrix * O_norm)
        denominator = torch.sum(self.weight_matrix * E_norm)
        
        # Kappa = 1 - (numerator / denominator)
        # Since we want to MINIMIZE loss, and a perfect Kappa is 1.0, 
        # we can simply return the ratio (numerator / denominator), 
        # or -Kappa, or log(numerator / denominator).
        # Returning log(numerator / denominator) is often numerically more stable for neural networks.
        
        # loss = torch.log((numerator + self.epsilon) / (denominator + self.epsilon))
        
        # Returning the raw ratio is the standard formulation.
        # Random guessing -> numerator == denominator -> loss ≈ 1.0
        # Perfect prediction -> numerator == 0 -> loss ≈ 0.0
        loss = numerator / (denominator + self.epsilon)
        
        return loss
