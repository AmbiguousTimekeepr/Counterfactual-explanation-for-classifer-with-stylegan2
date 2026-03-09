# models/synthesis.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===============================================================
# 1. Core StyleGAN2/3-style modulated conv (correct demodulation)
# ===============================================================
class ModulatedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, style_dim=512, upsample=False, demodulate=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, out_ch, in_ch, kernel_size, kernel_size))
        self.modulation = nn.Linear(style_dim, in_ch)
        self.demodulate = demodulate
        self.upsample = upsample
        self.padding = kernel_size // 2

    def forward(self, x, style):
        B, _, H, W = x.shape
        style = self.modulation(style).view(B, 1, -1, 1, 1)
        weight = self.weight * style

        if self.demodulate:
            decoefs = (weight.pow(2).sum((2, 3, 4)) + 1e-8).rsqrt()
            weight = weight * decoefs.view(B, -1, 1, 1, 1)

        # Reshape for grouped convolution: [B*out_ch, in_ch, k, k]
        weight = weight.view(B * weight.shape[1], weight.shape[2], weight.shape[3], weight.shape[4])
        x = x.view(1, B * x.shape[1], H, W)

        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        out = F.conv2d(x, weight, padding=self.padding, groups=B)
        out = out.view(B, -1, out.shape[2], out.shape[3])
        out = out + self.bias.view(1, -1, 1, 1) if hasattr(self, 'bias') else out
        return out

class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        return x + self.scale * noise

# ===============================================================
# 2. Synthesis Block
# ===============================================================
class SynthesisBlock(nn.Module):
    def __init__(self, in_ch, out_ch, style_dim, upsample=True):
        super().__init__()
        self.conv1 = ModulatedConv2d(in_ch, out_ch, 3, style_dim, upsample=upsample)
        self.noise1 = NoiseInjection(out_ch)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = ModulatedConv2d(out_ch, out_ch, 3, style_dim, upsample=False)
        self.noise2 = NoiseInjection(out_ch)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        self.to_rgb = ModulatedConv2d(out_ch, 3, 1, style_dim, demodulate=False)

    def forward(self, x, w):
        x = self.conv1(x, w)
        x = self.noise1(x)
        x = self.act1(x)
        x = self.conv2(x, w)
        x = self.noise2(x)
        x = self.act2(x)
        rgb = self.to_rgb(x, w)
        return x, rgb

# ===============================================================
# 3. Full Synthesis Network 
# ===============================================================
class StyleGAN3CounterfactualSynthesis(nn.Module):
    def __init__(self, resolution=128, style_dim=512):
        super().__init__()
        self.resolution = resolution
        self.style_dim = style_dim
        log_size = int(torch.log2(torch.tensor(resolution)))
        self.resolutions = [4 * 2**i for i in range(log_size - 1)]  # [4, 8, 16, 32, 64, 128]

        channel_base = 32768
        self.channels = {
            res: min(channel_base // res, 512) for res in self.resolutions
        }

        # Mapping network: z → w
        self.mapping = nn.Sequential(
            nn.Linear(512, style_dim), nn.LeakyReLU(0.2),
            *[nn.Sequential(nn.Linear(style_dim, style_dim), nn.LeakyReLU(0.2)) for _ in range(7)]
        )
        self.num_ws = len(self.resolutions) * 2  # two convs per block

        # Projectors: VQ latent levels → feature maps at correct resolution
        self.projectors = nn.ModuleList([
            nn.Conv2d(64, self.channels[4], 1),   # coarse  → 4x4
            nn.Conv2d(64, self.channels[8], 1),   # medium → 8x8
            nn.Conv2d(64, self.channels[16], 1),  # fine   → 16x16+
        ])

        # Synthesis blocks
        self.blocks = nn.ModuleList()
        in_ch = self.channels[4]
        for i, res in enumerate(self.resolutions):
            out_ch = self.channels[res]
            upsample = (i > 0)
            self.blocks.append(SynthesisBlock(in_ch, out_ch, style_dim, upsample))
            in_ch = out_ch

        # Fallback learned const
        self.register_parameter('const', nn.Parameter(torch.randn(1, self.channels[4], 4, 4)))
        # Optional: running w_avg for truncation trick
        self.w_avg = nn.Parameter(torch.zeros(style_dim), requires_grad=False)

    def forward(self, z_noise, z_list_edited=None, alpha_vec=None, truncation=0.7):
        """
        z_noise:        [B, 512] random latent for style
        z_list_edited:  list of 3 edited VQ-VAE latents [coarse, medium, fine]
                        each: [B, 64, H, W]
        alpha_vec:      [B, 3] curriculum strength per level (optional)
        """
        B = z_noise.shape[0]
        device = z_noise.device

        # ------------------------------------------------------------------
        # 1. Mapping network: z → w (global style control)
        # ------------------------------------------------------------------
        w = self.mapping(z_noise)                                      # [B, 512]
        if truncation < 1.0:
            w_avg = self.w_avg.lerp(w, truncation) if hasattr(self, 'w_avg') else w
            w = w_avg
        w = w.unsqueeze(1).repeat(1, self.num_ws, 1)                   # [B, num_ws, 512]

        # ------------------------------------------------------------------
        # 2. Start from edited coarse latent → FULLY replaces the learned const
        # ------------------------------------------------------------------
        if z_list_edited is not None and z_list_edited[0] is not None:
            x = self.projectors[0](z_list_edited[0])                   # [B, C, H, W] → [B, 512, ?, ?]
            x = F.interpolate(x, size=(4, 4), mode='bilinear', align_corners=False)
        else:
            x = self.const.repeat(B, 1, 1, 1)                           # fallback learned const

        # ------------------------------------------------------------------
        # 3. Prepare alpha (default to 1.0 if not provided)
        # ------------------------------------------------------------------
        if alpha_vec is None:
            alpha_vec = torch.ones(B, 3, device=device)
        alpha_coarse, alpha_medium, alpha_fine = alpha_vec[:, 0], alpha_vec[:, 1], alpha_vec[:, 2]

        # ------------------------------------------------------------------
        # 4. Synthesis loop with proper multi-level injection
        # ------------------------------------------------------------------
        rgb = None
        w_idx = 0

        for block_idx, block in enumerate(self.blocks):
            current_res = 4 * (2 ** block_idx)                         # 4→8→16→32→64→128

            # Inject medium at 8×8 resolution
            if block_idx == 1 and z_list_edited is not None and z_list_edited[1] is not None:
                delta = self.projectors[1](z_list_edited[1])
                delta = F.interpolate(delta, size=(current_res, current_res),
                                      mode='bilinear', align_corners=False)
                x = x + delta * alpha_medium.view(B, 1, 1, 1) * 0.8

            # Inject fine from 16×16 onward
            if block_idx >= 2 and z_list_edited is not None and z_list_edited[2] is not None:
                delta = self.projectors[2](z_list_edited[2])
                delta = F.interpolate(delta, size=(current_res, current_res),
                                      mode='bilinear', align_corners=False)
                x = x + delta * alpha_fine.view(B, 1, 1, 1) * 0.8

            # Forward through StyleGAN block (two modulated convs)
            x = block.conv1(x, w[:, w_idx])
            x = block.noise1(x)
            x = block.act1(x)
            x = block.conv2(x, w[:, w_idx + 1])
            x = block.noise2(x)
            x = block.act2(x)
            w_idx += 2

            # ToRGB
            rgb_block = block.to_rgb(x, w[:, w_idx - 1])               # reuse last w of the block

            if rgb is None:
                rgb = rgb_block
            else:
                rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False)
                rgb = rgb + rgb_block

        return rgb.clamp(-1, 1)