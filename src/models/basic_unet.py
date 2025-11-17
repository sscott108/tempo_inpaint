import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv → ReLU → Conv → ReLU) block"""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class SimpleUNet(nn.Module):
    """
    UNet-style Autoencoder for 1-channel 128×128 images.
    Keeps spatial resolution (128→16→128) and reconstructs input.
    """
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        # ---- Encoder ----
        self.down1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)         # 128 → 64
        self.down2 = DoubleConv(base, base*2)
        self.pool2 = nn.MaxPool2d(2)         # 64 → 32
        self.down3 = DoubleConv(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)         # 32 → 16
        self.bottleneck = DoubleConv(base*4, base*8)

        # ---- Decoder ----
        self.up3 = nn.ConvTranspose2d(base*8, base*4, kernel_size=2, stride=2)  # 16 → 32
        self.dec3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, kernel_size=2, stride=2)  # 32 → 64
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, kernel_size=2, stride=2)    # 64 → 128
        self.dec1 = DoubleConv(base*2, base)

        # ---- Output ----
        self.out = nn.Conv2d(base, in_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.down1(x)
        c2 = self.down2(self.pool1(c1))
        c3 = self.down3(self.pool2(c2))
        b  = self.bottleneck(self.pool3(c3))

        # Decoder
        x = self.up3(b)
        x = self.dec3(torch.cat([x, c3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, c2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, c1], dim=1))

        return self.out(x)
