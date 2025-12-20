# src/latent_mutator/module.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentMutator(nn.Module):
    """
    Hierarchical latent editor:
    - GradCAM++ as spatial mask ("where to edit")
    - Integrated Gradients as strength modulator ("how much to edit")
    - Large steps, continuous mutation during training
    """

    def __init__(
        self,
        num_attributes: int = 12,
        embed_dim: int = 64,
        num_levels: int = 3,
        step_max_multiplier: float = 15.0,
        step_bias: float = 5.0,
    ) -> None:
        super().__init__()
        self.num_attributes = num_attributes
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.step_max_multiplier = step_max_multiplier
        self.step_bias = step_bias

        # === CORRECT FEATURE COUNT ===
        # 3 pooled latents: embed_dim * 3
        # current_probs + target_vec: num_attributes * 2
        # IG stats: num_levels (one mean per level)
        input_dim = embed_dim * num_levels + num_attributes * 2 + num_levels

        self.context_backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        # === END FIX ===

        self.directions = nn.Parameter(torch.randn(num_attributes, embed_dim) * 0.2)
        self.step_heads = nn.ModuleList([
            nn.Linear(128, num_levels) for _ in range(num_attributes)
        ])
        nn.init.orthogonal_(self.directions)

    def forward(
        self,
        z_list: list[torch.Tensor],        # [z_top, z_mid, z_bottom] [B, C, H, W]
        cam_maps: list[torch.Tensor],       # [B, 1, H, W] per level - localization masks
        ig_maps: list[torch.Tensor],        # [B, 1, H, W] per level - importance strength
        target_vec: torch.Tensor,           # [B, num_attr]
        current_probs: torch.Tensor,        # [B, num_attr]
        active_attrs: torch.Tensor | None = None,  # tensor of active attr indices
    ) -> tuple[list[torch.Tensor], list[float]]:
        B, device = z_list[0].shape[0], z_list[0].device

        if active_attrs is None:
            diff = (target_vec - current_probs).abs()
            active_attrs = torch.nonzero(diff > 0.1)[:, 1].unique()

        # Global context
        pooled = [F.adaptive_avg_pool2d(z, 1).flatten(1) for z in z_list]
        ig_stats = torch.cat([m.mean(dim=[2, 3]).flatten(1) for m in ig_maps], dim=1)  # [B, num_levels]
        context = torch.cat(pooled + [current_probs, target_vec] + [ig_stats], dim=1)
        context = self.context_backbone(context)  # [B, 128]

        mutated_z = []
        avg_steps = []

        for level_idx in range(self.num_levels):
            z = z_list[level_idx]
            cam_mask = cam_maps[level_idx]  # [B, 1, H, W]
            ig_strength = ig_maps[level_idx]  # [B, 1, H, W]

            delta = torch.zeros_like(z)

            # accumulate base_step across attributes (fallback to zeros if none applied)
            base_step_accum = torch.zeros((B, 1, 1, 1), device=device)
            base_step_count = 0

            for attr_idx in active_attrs:
                attr_idx = int(attr_idx.item())
                attr_mask = (target_vec[:, attr_idx] != current_probs[:, attr_idx].round()).float()
                if attr_mask.sum() == 0:
                    continue

                direction = self.directions[attr_idx].view(1, -1, 1, 1)

                # Base step per sample
                step_logits = self.step_heads[attr_idx](context)[:, level_idx]
                base_step_attr = torch.sigmoid(step_logits).view(-1, 1, 1, 1)
                base_step_attr = base_step_attr * self.step_max_multiplier + self.step_bias  # 5–30

                # IG strength boost
                strength = 1.0 + 4.0 * ig_strength  # 1x → 5x in important areas

                # Sign
                sign = torch.sign(target_vec[:, attr_idx] - current_probs[:, attr_idx])
                sign = sign.view(-1, 1, 1, 1)

                # Apply
                attr_delta = sign * (base_step_attr * strength) * direction * cam_mask
                delta += attr_delta * attr_mask.view(-1, 1, 1, 1)

                base_step_accum = base_step_accum + base_step_attr
                base_step_count += 1

            mutated = z + delta
            mutated_z.append(mutated)
            if base_step_count > 0:
                base_step = base_step_accum / float(base_step_count)
            else:
                base_step = torch.zeros((B, 1, 1, 1), device=device)
            avg_steps.append(base_step.mean().item())

        return mutated_z, avg_steps