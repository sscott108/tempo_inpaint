import torch
import torch.nn as nn
import torch.nn.functional as F

class PConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, use_bias=True, activation=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation if activation is not None else nn.Identity()

        # Image kernel
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=kernel_size // 2, bias=use_bias)

        # Fixed mask kernel (all ones)
        self.register_buffer("mask_kernel",
                             torch.ones(1, 1, kernel_size, kernel_size))

    def forward(self, img, mask):
        # Apply mask to image
        img_masked = img * mask

        # Convolution
        img_out = self.conv(img_masked)

        # Convolve mask (counts valid pixels)
        with torch.no_grad():
            mask_out = F.conv2d(mask, self.mask_kernel,
                                stride=self.stride, padding=self.kernel_size // 2)
            mask_out = torch.clamp(mask_out, 0, 1)

        # Normalize by valid ratio
        n = self.kernel_size * self.kernel_size
        mask_ratio = n / (mask_out + 1e-8)
        mask_ratio = mask_ratio * mask_out
        img_out = img_out * mask_ratio

        return self.activation(img_out), mask_out

class PConvUNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        super().__init__()

        # Encoder
        self.enc1 = PConv2D(in_ch, base_ch, activation=nn.LeakyReLU(0.1))
        self.enc2 = PConv2D(base_ch, base_ch*2, stride=2, activation=nn.LeakyReLU(0.1))
        self.enc3 = PConv2D(base_ch*2, base_ch*4, stride=2, activation=nn.LeakyReLU(0.1))
        self.enc4 = PConv2D(base_ch*4, base_ch*8, stride=2, activation=nn.LeakyReLU(0.1))

        # Bottleneck
        self.bot  = PConv2D(base_ch*8, base_ch*8, activation=nn.LeakyReLU(0.1))

        # Decoder
        self.up3  = nn.ConvTranspose2d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.dec3 = PConv2D(base_ch*8, base_ch*4, activation=nn.LeakyReLU(0.1))

        self.up2  = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2 = PConv2D(base_ch*4, base_ch*2, activation=nn.LeakyReLU(0.1))

        self.up1  = nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec1 = PConv2D(base_ch*2, base_ch, activation=nn.LeakyReLU(0.1))

        # Final output
        self.final = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, img, mask):
        masks = []
        # Encoder
        e1, m1 = self.enc1(img, mask)
        e2, m2 = self.enc2(e1, m1)
        e3, m3 = self.enc3(e2, m2)
        e4, m4 = self.enc4(e3, m3)

        # Bottleneck
        b, mb = self.bot(e4, m4)

        # Decoder
        d3_in = torch.cat([self.up3(b), e3], dim=1)
        m3_in = torch.maximum(F.interpolate(mb, scale_factor=2, mode="nearest"), m3)
        d3, md3 = self.dec3(d3_in, m3_in)

        d2_in = torch.cat([self.up2(d3), e2], dim=1)
        m2_in = torch.maximum(F.interpolate(md3, scale_factor=2, mode="nearest"), m2)
        d2, md2 = self.dec2(d2_in, m2_in)

        d1_in = torch.cat([self.up1(d2), e1], dim=1)
        m1_in = torch.maximum(F.interpolate(md2, scale_factor=2, mode="nearest"), m1)
        d1, md1 = self.dec1(d1_in, m1_in)

        return self.final(d1), md1
