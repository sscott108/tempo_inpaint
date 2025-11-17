import torch
import torch.nn as nn
import torch.nn.functional as F



class SelfAttention2D(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SelfAttention2D, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        # Query, Key, Value projections
        self.query = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Output projection
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Learnable parameter to control attention influence
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Normalization
        self.norm = nn.LayerNorm([in_channels])
        
    def forward(self, x, mask=None):
        batch_size, C, H, W = x.size()
        
        # Generate Q, K, V
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # [B, HW, C//r]
        key = self.key(x).view(batch_size, -1, H * W)  # [B, C//r, HW]
        value = self.value(x).view(batch_size, -1, H * W)  # [B, C, HW]
        
        # Compute attention scores
        attention = torch.bmm(query, key)  # [B, HW, HW]
        
        # Apply mask to attention if provided
        if mask is not None:
            mask_flat = mask.view(batch_size, 1, H * W)  # [B, 1, HW]
            mask_matrix = torch.bmm(mask_flat.transpose(1, 2), mask_flat)  # [B, HW, HW]
            attention = attention * mask_matrix
            attention = attention.masked_fill(mask_matrix == 0, float('-inf'))
        
        # Softmax normalization
        attention = F.softmax(attention, dim=-1)  # [B, HW, HW]
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, HW]
        out = out.view(batch_size, C, H, W)  # [B, C, H, W]
        
        # Output projection
        out = self.out_proj(out)
        
        # Residual connection with learnable parameter
        out = self.gamma * out + x
        
        # Layer normalization
        out = out.permute(0, 2, 3, 1)  # [B, H, W, C]
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


# ---- Simple channel attention ----
class SimpleChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ---- Extended UNet (enc5/dec4 added) ----
class OriginalPlusMinimalAttentionDeepOld(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        super().__init__()

        # Encoder (5 levels now)
        self.enc1 = PConv2D(in_ch, base_ch, activation=nn.LeakyReLU(0.1))
        self.enc2 = PConv2D(base_ch, base_ch*2, stride=2, activation=nn.LeakyReLU(0.1))
        self.enc3 = PConv2D(base_ch*2, base_ch*4, stride=2, activation=nn.LeakyReLU(0.1))
        self.enc4 = PConv2D(base_ch*4, base_ch*8, stride=2, activation=nn.LeakyReLU(0.1))
        self.enc5 = PConv2D(base_ch*8, base_ch*16, stride=2, activation=nn.LeakyReLU(0.1))

        # Bottleneck
        self.bot = PConv2D(base_ch*16, base_ch*16, activation=nn.LeakyReLU(0.1))
        self.dilated = DilatedBlock(base_ch*16, base_ch*16, dilations=(1,2,4,8))
        self.bottleneck_attention = SimpleChannelAttention(base_ch*16)

        # Decoder (symmetric)
        self.up4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, kernel_size=2, stride=2)
        self.dec4 = PConv2D(base_ch*16, base_ch*8, activation=nn.LeakyReLU(0.1))

        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.dec3 = PConv2D(base_ch*8, base_ch*4, activation=nn.LeakyReLU(0.1))

        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2 = PConv2D(base_ch*4, base_ch*2, activation=nn.LeakyReLU(0.1))

        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec1 = PConv2D(base_ch*2, base_ch, activation=nn.LeakyReLU(0.1))

        self.final = nn.Conv2d(base_ch, out_ch, kernel_size=1)

        # Gated skips (4 gates now)
        self.gate4 = GatedSkip2d(ch_skip=base_ch*8, ch_up=base_ch*8)
        self.gate3 = GatedSkip2d(ch_skip=base_ch*4, ch_up=base_ch*4)
        self.gate2 = GatedSkip2d(ch_skip=base_ch*2, ch_up=base_ch*2)
        self.gate1 = GatedSkip2d(ch_skip=base_ch,   ch_up=base_ch)

    def forward(self, img, mask):
        # Encoder
        e1, m1 = self.enc1(img, mask)
        e2, m2 = self.enc2(e1, m1)
        e3, m3 = self.enc3(e2, m2)
        e4, m4 = self.enc4(e3, m3)
        e5, m5 = self.enc5(e4, m4)

        # Bottleneck
        b, mb = self.bot(e5, m5)
        b = self.dilated(b)
        b = self.bottleneck_attention(b)

        # Decoder
        up4 = self.up4(b)
        e4_g = self.gate4(e4, up4)
        d4_in = torch.cat([up4, e4_g], dim=1)
        m4_in = torch.maximum(F.interpolate(mb, scale_factor=2, mode="nearest"), m4)
        d4, md4 = self.dec4(d4_in, m4_in)

        up3 = self.up3(d4)
        e3_g = self.gate3(e3, up3)
        d3_in = torch.cat([up3, e3_g], dim=1)
        m3_in = torch.maximum(F.interpolate(md4, scale_factor=2, mode="nearest"), m3)
        d3, md3 = self.dec3(d3_in, m3_in)

        up2 = self.up2(d3)
        e2_g = self.gate2(e2, up2)
        d2_in = torch.cat([up2, e2_g], dim=1)
        m2_in = torch.maximum(F.interpolate(md3, scale_factor=2, mode="nearest"), m2)
        d2, md2 = self.dec2(d2_in, m2_in)

        up1 = self.up1(d2)
        e1_g = self.gate1(e1, up1)
        d1_in = torch.cat([up1, e1_g], dim=1)
        m1_in = torch.maximum(F.interpolate(md2, scale_factor=2, mode="nearest"), m1)
        d1, md1 = self.dec1(d1_in, m1_in)

        return self.final(d1), md1
    
class PConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, use_bias=True, activation=None):
        super().__init__()
        self.ks = kernel_size
        self.stride = stride
        self.activation = activation if activation is not None else nn.Identity()

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=kernel_size // 2, bias=use_bias)
        self.register_buffer("mask_kernel", torch.ones(1, 1, kernel_size, kernel_size))

    def forward(self, img, mask):
        # masked convolution on the image
        img = img.float()
        mask = mask.float()

        img_out = self.conv(img * mask)

        # count of valid pixels per window (no clamping!)
        with torch.no_grad():
            mask_count = F.conv2d(mask, self.mask_kernel,
                                  stride=self.stride, padding=self.ks // 2)

        # normalize by (#elements / count) and gate where count>0
        n = float(self.ks * self.ks)
        eps = 1e-8
        mask_ratio = n / (mask_count + eps)
        mask_any   = (mask_count > 0).float()

        img_out = img_out * mask_ratio * mask_any
        return self.activation(img_out), mask_any



class DilatedBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=(1, 2, 4)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=d, dilation=d)
            for d in dilations
        ])
        self.proj = nn.Conv2d(len(dilations) * out_ch, out_ch, kernel_size=1)

    def forward(self, x):
        feats = [conv(x) for conv in self.convs]
        out = torch.cat(feats, dim=1)
        return F.leaky_relu(self.proj(out), 0.1)


class GatedSkip2d(nn.Module):
    """Learn a per-pixel gate for the encoder skip before concatenation.
       gate = σ(Conv1x1([skip, up])) ;  gated_skip = gate * skip
    """
    def __init__(self, ch_skip, ch_up, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = max(ch_skip // 2, 8)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_skip + ch_up, hidden, kernel_size=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, ch_skip, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, skip, up):
        gate = self.conv(torch.cat([skip, up], dim=1))
        return skip * gate