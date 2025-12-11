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
        
        # A. Mapping Network (Top Latent -> Style w)
        # We assume Top Latent (8x8) contains global identity info
        self.mapping = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(latent_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2)
        )
        
        # B. Constant Input (4x4)
        self.const_input = nn.Parameter(torch.randn(1, 512, 4, 4))
        
        # C. Synthesis Blocks
        # 4x4 Block (Const + Style)
        self.conv1 = ModulatedConv2d(512, 512, style_dim=style_dim)
        self.to_rgb1 = ToRGB(512, style_dim)
        
        # 8x8 Block (Upsample + Inject Mid + Conv)
        self.conv2 = ModulatedConv2d(512, 512, style_dim=style_dim)
        self.injector_mid = nn.Conv2d(latent_dim, 512, 1) # Adapts 16x16/8x8 latent to features
        self.to_rgb2 = ToRGB(512, style_dim)
        
        # 16x16 Block
        self.conv3 = ModulatedConv2d(512, 512, style_dim=style_dim)
        # Usually Mid (16x16) fits best here naturally
        self.to_rgb3 = ToRGB(512, style_dim)
        
        # 32x32 Block (Inject Bottom)
        self.conv4 = ModulatedConv2d(512, 256, style_dim=style_dim)
        self.injector_bot = nn.Conv2d(latent_dim, 512, 1)  # Match channel count before addition
        self.to_rgb4 = ToRGB(256, style_dim)
        
        # 64x64 Block
        self.conv5 = ModulatedConv2d(256, 128, style_dim=style_dim)
        self.to_rgb5 = ToRGB(128, style_dim)
        
        # 128x128 Block
        self.conv6 = ModulatedConv2d(128, 64, style_dim=style_dim)
        self.to_rgb6 = ToRGB(64, style_dim)

    def forward(self, z_list):
        """
        z_list: [z_top(8x8), z_mid(16x16), z_bot(32x32)]
        """
        z_t, z_m, z_b = z_list
        batch = z_t.size(0)
        
        # 1. Generate Style 'w' from Top Level (Identity)
        w = self.mapping(z_t) # [B, 512]
        
        # 2. 4x4 (Start)
        x = self.const_input.repeat(batch, 1, 1, 1)
        x = self.conv1(x, w)
        rgb = self.to_rgb1(x, w)
        
        # 3. 8x8 (Upsample -> Inject Mid? -> Conv)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) # 8x8
        rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Injection: User requested "Mid-level VQ (16x16)" at 8x8 block
        # We downsample z_mid to 8x8 to add it
        feat_mid = self.injector_mid(z_m) # [B, 512, 16, 16]
        feat_mid_8 = F.interpolate(feat_mid, size=(8,8), mode='area') # Downsample to match
        x = x + feat_mid_8 # Injection
        
        x = self.conv2(x, w)
        rgb = rgb + self.to_rgb2(x, w)
        
        # 4. 16x16
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) # 16x16
        rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False)
        # (Could inject z_mid here again naturally, but sticking to user plan)
        x = self.conv3(x, w)
        rgb = rgb + self.to_rgb3(x, w)
        
        # 5. 32x32 (Inject Bottom)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) # 32x32
        rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False)
        
        feat_bot = self.injector_bot(z_b) # [B, 512, 32, 32]
        x = x + feat_bot
        x = self.conv4(x, w)
        rgb = rgb + self.to_rgb4(x, w)
        
        # 6. 64x64
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv5(x, w)
        rgb = rgb + self.to_rgb5(x, w)
        
        # 7. 128x128
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv6(x, w)
        rgb = rgb + self.to_rgb6(x, w)
        
        return torch.tanh(rgb) # [-1, 1]