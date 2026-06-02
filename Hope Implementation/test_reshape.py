import torch
cos_0 = torch.tensor([0.9, 0.8, 0.7])
cos_2 = torch.tensor([0.1, 0.2, 0.3])
concat = torch.cat((cos_0, cos_2), dim=0)
reshaped = concat.reshape(2, 3)
print("Concat:", concat)
print("Reshaped:\n", reshaped)
