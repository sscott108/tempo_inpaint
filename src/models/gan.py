import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=1, base_ch=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch*2, 4, stride=2, padding=1), nn.BatchNorm2d(base_ch*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch*2, base_ch*4, 4, stride=2, padding=1), nn.BatchNorm2d(base_ch*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch*4, 1, 4, padding=1)  # Patch map output
        )
    def forward(self, x):
        return self.model(x)