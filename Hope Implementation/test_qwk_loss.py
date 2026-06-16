import torch
from models.qwk_loss import DifferentiableQWKLoss

def test_qwk():
    print("Testing Differentiable QWK Loss...")
    
    batch_size = 8
    num_classes = 4
    
    # Create dummy logits (random values)
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    
    # Create dummy integer targets (0 to 3)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    print(f"Logits shape: {logits.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Targets: {targets}")
    
    # Initialize Loss
    criterion = DifferentiableQWKLoss(num_classes=num_classes)
    
    # Forward pass
    loss = criterion(logits, targets)
    print(f"\nForward Pass Loss value: {loss.item()}")
    
    # Check for NaNs
    assert not torch.isnan(loss), "Loss contains NaNs!"
    
    # Backward pass
    loss.backward()
    
    print(f"Backward Pass Gradients (first row of logits):\n{logits.grad[0]}")
    assert logits.grad is not None, "Gradients were not computed!"
    assert not torch.isnan(logits.grad).any(), "Gradients contain NaNs!"
    
    print("\nTest Passed! Differentiable QWK Loss is mathematically stable and gradients flow backward correctly.")

if __name__ == '__main__':
    test_qwk()
