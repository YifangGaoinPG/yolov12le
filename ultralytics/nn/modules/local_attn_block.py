import torch
import torch.nn as nn

class LocalAttnBlock(nn.Module):
    def __init__(self, c=None):
        super().__init__()
        self.conv1 = None
        self.c_in = c  # placeholder

    def forward(self, x):
        if self.conv1 is None:
            c = x.shape[1]
            self.conv1 = nn.Conv2d(c, c, kernel_size=1).to(x.device)
            self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c).to(x.device)
            self.act = nn.SiLU()
        return x + self.act(self.conv1(x) + self.conv2(x))


class ESHABlock(nn.Module):
    def __init__(self, c=None):
        super().__init__()
        self.fc1 = None
        self.c_in = c

    def forward(self, x):
        if self.fc1 is None:
            c = x.shape[1]
            mid = max(c // 4, 1)
            self.fc1 = nn.Conv2d(c, mid, 1).to(x.device)
            self.fc2 = nn.Conv2d(mid, c, 1).to(x.device)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.act = nn.SiLU()
            self.sigmoid = nn.Sigmoid()
        w = self.sigmoid(self.fc2(self.act(self.fc1(self.pool(x)))))
        return x * w

    
class Add(nn.Module):
    def __init__(self, _=None):  # _ 保留以兼容 YAML
        super().__init__()

    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 2, "Add expects two inputs"
        return x[0] + x[1]
        
