import torch
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        # Unregistered tensor
        self.prototypes = torch.zeros(3, 128)
        
    def state_dict(self, *args, **kwargs):
        print("MyNet state_dict called!")
        sd = super().state_dict(*args, **kwargs)
        sd['prototypes'] = self.prototypes
        return sd

net = MyNet()
dp_net = nn.DataParallel(net)
sd = dp_net.state_dict()
print("Keys in DataParallel state_dict:", sd.keys())
