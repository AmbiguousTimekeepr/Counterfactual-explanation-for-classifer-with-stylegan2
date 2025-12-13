import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ModulatedConv2d(nn.Module):
    """
    StyleGAN2-style Demodulated Convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, style_dim=512, demodulate=True):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.padding = kernel_size // 2

        # Weights [Out, In, K, K]
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channels * kernel_size ** 2)
        
        # Modulation (Style -> Scale)
        self.modulation = nn.Linear(style_dim, in_channels)
        self.modulation.bias.data.fill_(1.0) # Init with identity-like scale
        
        # Noise injection
        self.noise_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, style, noise=None):
        batch, in_c, h, w = x.shape
        
        # 1. Modulate
        style = self.modulation(style).view(batch, 1, in_c, 1, 1) # [B, 1, In, 1, 1]
        weight = self.weight * self.scale # [Out, In, K, K]
        weights = weight.unsqueeze(0) * style # [B, Out, In, K, K]

        # 2. Demodulate
        if self.demodulate:
            dcoefs = (weights.pow(2).sum(dim=[2, 3, 4]) + 1e-8).rsqrt() # [B, Out]
            weights = weights * dcoefs.view(batch, -1, 1, 1, 1)

        # 3. Conv
        # Reshape to use group convolution trick for per-sample weights
        x = x.view(1, -1, h, w) # [1, B*In, H, W]
        weights = weights.view(-1, in_c, self.kernel_size, self.kernel_size) # [B*Out, In, K, K]
        x = F.conv2d(x, weights, padding=self.padding, groups=batch)
        x = x.view(batch, self.out_channels, h, w)

        # 4. Noise
        if noise is None:
            noise = torch.randn(batch, 1, h, w, device=x.device)
        x = x + noise * self.noise_scale
        
        return F.leaky_relu(x, 0.2)

class ToRGB(nn.Module):
    """Project feature map to RGB"""
    def __init__(self, in_channels, style_dim=512):
        super().__init__()
        self.conv = ModulatedConv2d(in_channels, 3, kernel_size=1, style_dim=style_dim, demodulate=False)
        
    def forward(self, x, style):
        return self.conv(x, style)

class HierarchicalSynthesisNet(nn.Module):
    """
    StyleGAN3-style Decoder with Spatial Latent Injection.
    Resolution: 4x4 -> 128x128
    """
    def __init__(self, style_dim=512, latent_dim=64):
        super().__init__()
        self.style_dim = style_dim

        layers = [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        in_features = latent_dim
        for _ in range(6):
            layers.append(nn.Linear(in_features, style_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_features = style_dim
        self.mapping = nn.Sequential(*layers)

        self.input_projector = nn.Sequential(
            nn.Conv2d(latent_dim, 512, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(512)
        )

        self.conv_8 = ModulatedConv2d(512, 512, style_dim=style_dim)
        self.to_rgb_8 = ToRGB(512, style_dim)

        self.conv_16 = ModulatedConv2d(512, 512, style_dim=style_dim)
        self.inj_mid = nn.Conv2d(latent_dim, 512, 1)
        self.to_rgb_16 = ToRGB(512, style_dim)

        self.conv_32 = ModulatedConv2d(512, 256, style_dim=style_dim)
        self.inj_bot = nn.Conv2d(latent_dim, 512, 1)
        self.to_rgb_32 = ToRGB(256, style_dim)

        self.conv_64 = ModulatedConv2d(256, 128, style_dim=style_dim)
        self.to_rgb_64 = ToRGB(128, style_dim)

        self.conv_128 = ModulatedConv2d(128, 64, style_dim=style_dim)
        self.to_rgb_128 = ToRGB(64, style_dim)

    def forward(self, z_list, masks=None):
        z_t, z_m, z_b = z_list
        w = self.mapping(z_t)

        x = self.input_projector(z_t)

        x = self.conv_8(x, w)
        rgb = self.to_rgb_8(x, w)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False)
        x = x + self.inj_mid(z_m)
        x = self.conv_16(x, w)
        rgb = rgb + self.to_rgb_16(x, w)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False)
        x = x + self.inj_bot(z_b)
        x = self.conv_32(x, w)
        rgb = rgb + self.to_rgb_32(x, w)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv_64(x, w)
        rgb = rgb + self.to_rgb_64(x, w)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv_128(x, w)
        rgb = rgb + self.to_rgb_128(x, w)

        return torch.tanh(rgb) # [-1, 1]