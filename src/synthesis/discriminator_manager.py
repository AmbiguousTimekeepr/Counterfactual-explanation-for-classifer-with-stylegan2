import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------
# 1. Default discriminator
# --------------------------------------------------------------
class ConfGANDefaultDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def spectral_norm_conv(in_ch, out_ch, kernel, stride, padding):
            return nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel, stride, padding))
        self.net = nn.Sequential(
            spectral_norm_conv(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            spectral_norm_conv(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
            spectral_norm_conv(128, 256, 4, 2, 1), nn.LeakyReLU(0.2),
            spectral_norm_conv(256, 512, 4, 2, 1), nn.LeakyReLU(0.2),
            spectral_norm_conv(512, 1, 4, 1, 0)
        )
    def forward(self, x):
        return self.net(x).view(x.size(0), -1)
    

# --------------------------------------------------------------
# 2. PatchGAN-style discriminator
# --------------------------------------------------------------
class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for local feature discrimination
    Outputs a patch-wise discrimination map instead of single value
    """
    def __init__(self, input_channels=3, base_channels=64, n_layers=3):
        super().__init__()
        
        def spectral_norm_conv(in_ch, out_ch, kernel, stride, padding, bias=True):
            return nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=bias))
        
        # First layer (no normalization)
        layers = [
            spectral_norm_conv(input_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Intermediate layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)  # Cap at 8x base_channels
            layers += [
                spectral_norm_conv(base_channels * nf_mult_prev, base_channels * nf_mult, 4, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        # Penultimate layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            spectral_norm_conv(base_channels * nf_mult_prev, base_channels * nf_mult, 4, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Final layer
        layers += [spectral_norm_conv(base_channels * nf_mult, 1, 4, 1, 1)]
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through PatchGAN discriminator
        Args:
            x: Input images [B, 3, 64, 64]
        Returns:
            Patch-wise discrimination [B, 1, H', W'] -> flattened to [B, -1]
        """
        output = self.model(x)
        # For compatibility, flatten to single value per sample
        return output.view(x.size(0), -1).mean(dim=1, keepdim=True)


# --------------------------------------------------------------
# 3. StyleGAN3-style discriminator
# --------------------------------------------------------------
class StyleGAN3Discriminator(nn.Module):
    """
    StyleGAN3-style Discriminator with modern techniques
    Uses filtered downsampling and anti-aliasing
    """
    def __init__(self, input_channels=3, base_channels=64, max_channels=512):
        super().__init__()
        
        self.base_channels = base_channels
        self.max_channels = max_channels
        
        # Anti-aliasing downsampling filter
        self.register_buffer('blur_kernel', self._get_blur_kernel())
        
        # Progressive blocks: 64->32->16->8->4
        self.blocks = nn.ModuleList()
        in_ch = input_channels
        out_ch = base_channels
        
        # Initial block (64x64 -> 32x32)
        self.blocks.append(self._make_stylegan_block(in_ch, out_ch, first_block=True))
        in_ch = out_ch
        
        # Progressive blocks
        for i in range(3):  # 32->16->8->4
            out_ch = min(in_ch * 2, max_channels)
            self.blocks.append(self._make_stylegan_block(in_ch, out_ch))
            in_ch = out_ch
        
        # Minibatch standard deviation (StyleGAN technique)
        self.minibatch_std = True
        
        # Final classification - account for minibatch std extra channel
        final_channels = in_ch + (1 if self.minibatch_std else 0)
        self.final_conv = nn.Conv2d(final_channels, 1, 4, 1, 0)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _get_blur_kernel(self):
        """Get anti-aliasing blur kernel"""
        kernel = torch.tensor([1, 3, 3, 1], dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel / kernel.sum()
        return kernel[None, None, :, :]
    
    def _make_stylegan_block(self, in_channels, out_channels, first_block=False):
        """Create a StyleGAN3-style discriminator block"""
        layers = []
        
        if not first_block:
            # Residual connection for non-first blocks
            layers.append(ResidualBlock(in_channels, out_channels))
        else:
            # Regular conv for first block
            layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Anti-aliased downsampling
        layers.append(BlurDownsample(out_channels))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        """Initialize weights using He initialization"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x, return_features: bool = False):
        """
        Forward pass through StyleGAN3 discriminator
        Args:
            x: Input images [B, 3, 64, 64]
            return_features: If True, also return intermediate features for
                feature-matching losses.
        Returns:
            logits [B, 1] if return_features is False
            (logits, feature_list) if return_features is True
        """
        features = []

        # Progressive downsampling through blocks
        for block in self.blocks:
            x = block(x)
            if return_features:
                features.append(x)
        
        # Minibatch standard deviation
        if self.minibatch_std and self.training:
            x = self._minibatch_stddev(x)
            if return_features:
                features.append(x)
        
        # Final classification
        x = self.final_conv(x)  # [B, 1, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 1]
        
        if return_features:
            return x, features
        return x
    
    def _minibatch_stddev(self, x):
        """Add minibatch standard deviation feature"""
        b, c, h, w = x.shape
        std = x.std(dim=0, keepdim=True)  # [1, c, h, w]
        std = std.mean([1, 2, 3], keepdim=True)  # [1, 1, 1, 1]
        std = std.repeat(b, 1, h, w)  # [b, 1, h, w]
        return torch.cat([x, std], dim=1)


class ResidualBlock(nn.Module):
    """Residual block for StyleGAN3 discriminator"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        skip = self.skip(x)
        
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        
        return self.activation(out + skip)


class BlurDownsample(nn.Module):
    """Anti-aliased downsampling with blur"""
    def __init__(self, channels):
        super().__init__()
        # Simple blur kernel
        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer('kernel', kernel)
        self.channels = channels
        
    def forward(self, x):
        # Apply blur
        x = F.conv2d(x, self.kernel, padding=1, groups=self.channels)
        # Downsample
        x = F.avg_pool2d(x, 2, stride=2)
        return x


def create_discriminator(model_name):
    """
    Create a discriminator based on the model name
    
    Args:
        model_name: One of 'R3GAN', 'R3GAN_Enhanced', 'PatchGAN', 'StyleGAN3'
    
    Returns:
        Discriminator model
    """
    if model_name == 'PatchGAN':
        print("Using PatchGANDiscriminator as the discriminator model.")
        return PatchGANDiscriminator()
    elif model_name == 'StyleGAN3':
        print("Using StyleGAN3Discriminator as the discriminator model.")
        return StyleGAN3Discriminator()
    else:
        print(f"Discriminator model '{model_name}' not recognized. Available: 'R3GAN', 'R3GAN_Enhanced', 'PatchGAN', 'StyleGAN3'. Using default ConfGANDefaultDiscriminator.")
        return ConfGANDefaultDiscriminator()