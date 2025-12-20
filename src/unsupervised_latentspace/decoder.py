import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int, num_res_blocks: int, stride: int):
        super().__init__()
        self.res_stack = nn.Sequential(
            *[ResBlock(in_channels, in_channels) for _ in range(num_res_blocks)],
            nn.ReLU(),
        )
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res_stack(x)
        return self.up(x)


class HVQDecoder(nn.Module):
    """Hierarchical decoder shared by HVQVAE levels."""

    def __init__(self, cfg):
        super().__init__()
        hd = cfg.hidden_dim
        ed = cfg.embed_dim

        self.dec_t = DecoderBlock(ed, hd, hd, cfg.num_res_blocks, stride=2)
        self.dec_m = DecoderBlock(hd + ed, hd, hd, cfg.num_res_blocks, stride=2)
        self.dec_b = nn.Sequential(
            nn.Conv2d(hd + ed, hd, 3, 1, 1),
            *[ResBlock(hd, hd) for _ in range(cfg.num_res_blocks)],
            nn.ConvTranspose2d(hd, hd // 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hd // 2, cfg.in_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, q_t: torch.Tensor, q_m: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
        dec_t_out = self.dec_t(q_t)

        if dec_t_out.shape[2:] != q_m.shape[2:]:
            dec_t_out = F.interpolate(dec_t_out, size=q_m.shape[2:], mode="nearest")

        dec_m_in = torch.cat([dec_t_out, q_m], dim=1)
        dec_m_out = self.dec_m(dec_m_in)

        if dec_m_out.shape[2:] != q_b.shape[2:]:
            dec_m_out = F.interpolate(dec_m_out, size=q_b.shape[2:], mode="nearest")

        dec_b_in = torch.cat([dec_m_out, q_b], dim=1)
        return self.dec_b(dec_b_in)
