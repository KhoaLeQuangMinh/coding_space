import torch
checkpoint = torch.load('/kaggle/working/checkpoints/hope_original_fold1/60_net.pth', map_location='cpu')
prototypes = checkpoint['prototypes']
print("Prototypes shape:", prototypes.shape)
print("Norm of prototypes:", torch.norm(prototypes, p=2, dim=1))
print("Dot product CN vs AD:", torch.sum(prototypes[0] * prototypes[2]))
