# local_attn_block.py
import torch
import torch.nn as nn

class LocalAttnBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, kernel_size=1)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c)  # DW conv
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.conv1(x) + self.conv2(x))

class ESHABlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c // 4, 1)
        self.fc2 = nn.Conv2d(c // 4, c, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.sigmoid(self.fc2(nn.SiLU()(self.fc1(self.pool(x)))))
        return x * w
    
class ParallelFusionBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.local = LocalAttnBlock(c)
        self.esha = ESHABlock(c)
        self.concat = nn.Conv2d(c * 2, c, kernel_size=1)  # optional channel fusion

    def forward(self, x):
        out1 = self.local(x)
        out2 = self.esha(x)
        fused = torch.cat([out1, out2], dim=1)  # concat on channel dim
        return self.concat(fused)
    
class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y

