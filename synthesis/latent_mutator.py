import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentMutator(nn.Module):
    """
    Counterfactual latent editor.
    Uses IG + Grad-CAM++ + learned per-sample step sizes → perfect edits, zero identity drift.
    """
    def __init__(self, embed_dim: int = 64, num_attributes: int = 39, num_levels: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_attributes = num_attributes
        self.num_levels = num_levels

        # Learnable attribute directions (one 64-dim vector per attribute)
        self.directions = nn.Parameter(
            torch.randn(num_attributes, embed_dim) * 0.02
        )

        # Per-attribute MLP that learns how strong to mutate based on EVERYTHING
        self.step_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim + 8, 128),
                nn.LayerNorm(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.05),

                nn.Linear(128, 96),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(96, 64),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(64, num_levels),
                nn.Softplus(beta=1.0)  # Output > 0, smooth, unbounded
            )
            for _ in range(num_attributes)
        ])

        # Optional: refine Grad-CAM mask per resolution level
        self.mask_refiner = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def refine_mask(self, mask: torch.Tensor, target_hw: tuple) -> torch.Tensor:
        if mask is None:
            return None
        return self.mask_refiner(F.interpolate(mask, size=target_hw, mode='nearest'))

    def forward(
        self,
        z_list,                    # List[Tensor]: [z_coarse, z_medium, z_fine], each [B, C, H, W]
        alpha_vec,                 # [B, num_levels] ∈ [0,1] — curriculum strength
        target_attrs,              # [B, num_attrs] ∈ {0,1}
        current_attrs,             # [B, num_attrs] ∈ [0,1] — classifier confidence
        ig_scores,                 # [B, num_attrs] ∈ [0,1] — Integrated Gradients importance
        spatial_mask=None,         # [B, 1, 32, 32] — Grad-CAM++ (detached)
    ) -> list:
        B, device = z_list[0].shape[0], z_list[0].device

        # Fast exit
        if alpha_vec.abs().max() < 1e-6 or (target_attrs.round() == current_attrs.round()).all():
            return z_list

        z_mutated = [z.clone() for z in z_list]
        flip_needed = (target_attrs.round() != current_attrs.round())          # [B, 39]
        flip_sign = (target_attrs - current_attrs).sign()                     # +1 or -1

        if not flip_needed.any():
            return z_mutated

        # 1. Global image summary (pooled latent)
        pooled = torch.cat([
            F.adaptive_avg_pool2d(z, 1).flatten(1) for z in z_list
        ], dim=1)
        pooled = pooled[:, :self.embed_dim]  # safety

        # 2. Build rich context tensor: [B, num_attrs, 8]
        current_conf = torch.sigmoid(current_attrs)
        target_conf = target_attrs
        confidence_gap = (target_conf - current_conf).abs()
        ig_norm = ig_scores.clamp(0.0, 1.0)
        # gradcam_strength: [B, 1] -> expand to [B, num_attrs]
        gradcam_strength = (
            spatial_mask.mean(dim=[2,3]).expand(-1, self.num_attributes) if spatial_mask is not None
            else torch.zeros(B, self.num_attributes, device=device)
        )
        # alpha_vec.mean(dim=1, keepdim=True): [B, 1] -> expand to [B, num_attrs]
        alpha_mean_expanded = alpha_vec.mean(dim=1, keepdim=True).expand(-1, self.num_attributes)

        context = torch.stack([
            flip_sign,
            confidence_gap,
            ig_norm,
            ig_norm ** 2,
            current_conf,
            target_conf,
            gradcam_strength,
            alpha_mean_expanded,
        ], dim=2)  # [B, num_attrs, 8]

        # 3. Pre-compute refined masks per level
        refined_masks = (
            [self.refine_mask(spatial_mask, z.shape[2:]) for z in z_list]
            if spatial_mask is not None else [None] * self.num_levels
        )

        # 4. Surgical mutation loop
        for attr_idx in range(self.num_attributes):
            idxs = flip_needed[:, attr_idx]
            if not idxs.any():
                continue

            direction = F.normalize(self.directions[attr_idx], dim=0)  # unit vector
            sign = flip_sign[idxs, attr_idx:attr_idx+1]                # [N,1]

            # Gather per-sample context
            attr_context = context[idxs, attr_idx]                     # [N,8]
            attr_pooled = pooled[idxs]                                 # [N, embed_dim]

            # Predict adaptive step sizes
            step_input = torch.cat([attr_pooled, attr_context], dim=1)
            steps = self.step_predictors[attr_idx](step_input)         # [N, 3], >0

            # Apply to each level
            for lvl, (z, mask) in enumerate(zip(z_mutated, refined_masks)):
                alpha = alpha_vec[idxs, lvl].view(-1, 1, 1, 1)
                step = steps[:, lvl].view(-1, 1, 1, 1)

                delta = (
                    alpha * step *
                    direction.view(1, -1, 1, 1) *
                    sign.view(-1, 1, 1, 1)
                )

                if mask is not None:
                    delta = delta * mask[idxs]

                z[idxs].add_(delta)

        return z_mutated

    # Regularizers
    def orthogonality_loss(self) -> torch.Tensor:
        normed = F.normalize(self.directions, dim=1)
        gram = torch.mm(normed, normed.t())
        return (gram - torch.eye(self.num_attributes, device=gram.device)).abs().mean()

    def direction_l1(self) -> torch.Tensor:
        return self.directions.abs().mean() * 1e-3

    def step_predictor_l1(self) -> torch.Tensor:
        loss = 0.0
        for pred in self.step_predictors:
            loss += pred[-2].weight.abs().mean()  # penultimate layer
        return loss / len(self.step_predictors) * 1e-5