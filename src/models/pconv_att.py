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


class LightweightSelfAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(LightweightSelfAttention, self).__init__()
        self.in_channels = in_channels
        
        # Much smaller attention dimensions
        hidden_dim = max(in_channels // reduction, 8)
        
        # Use 1x1 convs for efficiency
        self.query = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.key = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.value = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.out = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, mask=None):
        batch_size, C, H, W = x.size()
        
        # Downsample for attention computation (reduces memory)
        if H > 32 or W > 32:
            x_down = F.adaptive_avg_pool2d(x, (min(H, 32), min(W, 32)))
            h, w = x_down.shape[2], x_down.shape[3]
        else:
            x_down = x
            h, w = H, W
        
        # Compute attention on downsampled feature
        q = self.query(x_down).view(batch_size, -1, h * w).permute(0, 2, 1)
        k = self.key(x_down).view(batch_size, -1, h * w)
        v = self.value(x_down).view(batch_size, -1, h * w)
        
        attention = torch.bmm(q, k)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, -1, h, w)
        out = self.out(out)
        
        # Upsample back to original size if needed
        if h != H or w != W:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        
        return self.gamma * out + x

class SimpleChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(SimpleChannelAttention, self).__init__()
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

class EfficientPConvUNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32, attention_layers=[4]):
        """
        attention_layers: list of encoder levels to apply attention (0-4)
        Example: [4] = only bottleneck, [3,4] = enc4 and bottleneck
        """
        super().__init__()
        self.attention_layers = attention_layers

        # Encoder
        self.enc1 = PConv2D(in_ch, base_ch, activation=nn.LeakyReLU(0.1))
        self.enc2 = PConv2D(base_ch, base_ch*2, stride=2, activation=nn.LeakyReLU(0.1))
        self.enc3 = PConv2D(base_ch*2, base_ch*4, stride=2, activation=nn.LeakyReLU(0.1))
        self.enc4 = PConv2D(base_ch*4, base_ch*8, stride=2, activation=nn.LeakyReLU(0.1))

        # Selective attention - only where specified
        self.attentions = nn.ModuleDict()
        if 1 in attention_layers:
            self.attentions['1'] = SimpleChannelAttention(base_ch)
        if 2 in attention_layers:
            self.attentions['2'] = SimpleChannelAttention(base_ch*2)
        if 3 in attention_layers:
            self.attentions['3'] = SimpleChannelAttention(base_ch*4)
        if 4 in attention_layers:
            self.attentions['4'] = LightweightSelfAttention(base_ch*8, reduction=8)

        # Bottleneck
        self.bot = PConv2D(base_ch*8, base_ch*8, activation=nn.LeakyReLU(0.1))
        if 'bot' in [str(x) for x in attention_layers] or 5 in attention_layers:
            self.attentions['bot'] = LightweightSelfAttention(base_ch*8, reduction=8)

        # Decoder - simplified
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.dec3 = PConv2D(base_ch*8, base_ch*4, activation=nn.LeakyReLU(0.1))

        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2 = PConv2D(base_ch*4, base_ch*2, activation=nn.LeakyReLU(0.1))

        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec1 = PConv2D(base_ch*2, base_ch, activation=nn.LeakyReLU(0.1))

        # Simple final layer
        self.final = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, img, mask):
        # Encoder
        e1, m1 = self.enc1(img, mask)
        if '1' in self.attentions:
            e1 = self.attentions['1'](e1)

        e2, m2 = self.enc2(e1, m1)
        if '2' in self.attentions:
            e2 = self.attentions['2'](e2)

        e3, m3 = self.enc3(e2, m2)
        if '3' in self.attentions:
            e3 = self.attentions['3'](e3)

        e4, m4 = self.enc4(e3, m3)
        if '4' in self.attentions:
            e4 = self.attentions['4'](e4, m4)

        # Bottleneck
        b, mb = self.bot(e4, m4)
        if 'bot' in self.attentions:
            b = self.attentions['bot'](b, mb)

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
    
class MinimalAttentionPConvUNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=16):  # Reduced base channels
        super().__init__()

        # Encoder - smaller channels
        self.enc1 = PConv2D(in_ch, base_ch, activation=nn.LeakyReLU(0.1))
        self.enc2 = PConv2D(base_ch, base_ch*2, stride=2, activation=nn.LeakyReLU(0.1))
        self.enc3 = PConv2D(base_ch*2, base_ch*4, stride=2, activation=nn.LeakyReLU(0.1))

        # Only one attention at bottleneck
        self.bottleneck_attention = SimpleChannelAttention(base_ch*4)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2 = PConv2D(base_ch*4, base_ch*2, activation=nn.LeakyReLU(0.1))

        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec1 = PConv2D(base_ch*2, base_ch, activation=nn.LeakyReLU(0.1))

        self.final = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, img, mask):
        # Encoder
        e1, m1 = self.enc1(img, mask)
        e2, m2 = self.enc2(e1, m1)
        e3, m3 = self.enc3(e2, m2)

        # Apply attention only at bottleneck
        e3_att = self.bottleneck_attention(e3)

        # Decoder
        d2_in = torch.cat([self.up2(e3_att), e2], dim=1)
        m2_in = torch.maximum(F.interpolate(m3, scale_factor=2, mode="nearest"), m2)
        d2, md2 = self.dec2(d2_in, m2_in)

        d1_in = torch.cat([self.up1(d2), e1], dim=1)
        m1_in = torch.maximum(F.interpolate(md2, scale_factor=2, mode="nearest"), m1)
        d1, md1 = self.dec1(d1_in, m1_in)

        return self.final(d1), md1
    
class OriginalPlusMinimalAttention(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        super().__init__()

        # Your original encoder/decoder (unchanged)
        self.enc1 = PConv2D(in_ch, base_ch, activation=nn.LeakyReLU(0.1))
        self.enc2 = PConv2D(base_ch, base_ch*2, stride=2, activation=nn.LeakyReLU(0.1))
        self.enc3 = PConv2D(base_ch*2, base_ch*4, stride=2, activation=nn.LeakyReLU(0.1))
        self.enc4 = PConv2D(base_ch*4, base_ch*8, stride=2, activation=nn.LeakyReLU(0.1))
        self.bot = PConv2D(base_ch*8, base_ch*8, activation=nn.LeakyReLU(0.1))

        # Add ONLY bottleneck attention
        self.bottleneck_attention = SimpleChannelAttention(base_ch*8)

        # Your original decoder (unchanged)
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.dec3 = PConv2D(base_ch*8, base_ch*4, activation=nn.LeakyReLU(0.1))
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2 = PConv2D(base_ch*4, base_ch*2, activation=nn.LeakyReLU(0.1))
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec1 = PConv2D(base_ch*2, base_ch, activation=nn.LeakyReLU(0.1))
        self.final = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, img, mask):
        # Your original forward pass
        e1, m1 = self.enc1(img, mask)
        e2, m2 = self.enc2(e1, m1)
        e3, m3 = self.enc3(e2, m2)
        e4, m4 = self.enc4(e3, m3)
        
        b, mb = self.bot(e4, m4)
        
        # Add attention ONLY here
        b = self.bottleneck_attention(b)
        
        # Continue with original decoder
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