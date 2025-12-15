import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentMutator(nn.Module):
    """
    Single-Vector Latent Mutator with Statistical IG Guidance.
    Optimized for memory efficiency and theoretical robustness.
    """
    def __init__(self, embed_dim=64, num_attributes=40, num_levels=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_attributes = num_attributes
        self.num_levels = num_levels
        self.directions = nn.Parameter(torch.randn(num_attributes, embed_dim) * 0.10)

        self.step_predictors = nn.ModuleList()
        for _ in range(num_attributes):
            predictor = nn.Sequential(
                nn.Linear(embed_dim + 8, 128),
                nn.LayerNorm(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(128, 96),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(96, 64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(64, num_levels)
            )
            predictor[-1].bias.data.fill_(5.0)
            self.step_predictors.append(predictor)

    def get_ig_features(self, ig_map):
        """
        Extracts 8 statistical features from the IG spatial map [B, H, W]
        """
        # Flatten spatial dims: [B, H*W]
        flat = ig_map.view(ig_map.size(0), -1).abs()
        
        # 1. Mean Intensity
        f_mean = flat.mean(dim=1, keepdim=True)
        # 2. Max Intensity
        f_max = flat.max(dim=1, keepdim=True).values
        # 3. Std Dev
        f_std = flat.std(dim=1, keepdim=True)
        # 4. Positive Ratio (Assuming input is abs, checking > 0 isn't useful, 
        # so we check 'significant' attribution > 1e-4)
        f_pos = (flat > 1e-4).float().mean(dim=1, keepdim=True)
        # 5. 90th Quantile
        f_q90 = torch.quantile(flat, 0.9, dim=1, keepdim=True)
        # 6. 10th Quantile
        f_q10 = torch.quantile(flat, 0.1, dim=1, keepdim=True)
        # 7. Hotspot Ratio (pixels > 95% of max)
        # Note: using max() per sample for threshold
        threshold = flat.max(dim=1, keepdim=True).values * 0.95
        f_hotspot = (flat > threshold).float().mean(dim=1, keepdim=True)
        # 8. Entropy (Spatial Concentration)
        # Normalize to prob dist
        probs = flat / (flat.sum(dim=1, keepdim=True) + 1e-8)
        f_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)

        # Concat: [B, 8]
        return torch.cat([f_mean, f_max, f_std, f_pos, f_q90, f_q10, f_hotspot, f_entropy], dim=1)

    def forward(self, z_list, ig_map, cam_masks, target_vector, current_probs, active_attr_idx):
        """
        Args:
            z_list: List of latents [z_top, z_mid, z_bot]
            ig_map: Integrated Gradients map [B, 128, 128] (aggregated spatial)
            cam_masks: List of masks resized to [8x8, 16x16, 32x32]
            target_vector: Target binary labels [B, num_attributes]
            current_probs: Current predictions [B, num_attributes]
            active_attr_idx: The single attribute index we are currently editing (int)
        """
        batch_size = z_list[0].size(0)
        z_mutated = [z.clone() for z in z_list]
        step_values = []
        
        # 1. Calculate Direction Sign
        # If Target=1, Prob=0.1 -> Diff=0.9 (Pos) -> Add direction
        # If Target=0, Prob=0.9 -> Diff=-0.9 (Neg) -> Subtract direction
        # We process only the 'active_attr_idx' to save compute
        target = target_vector[:, active_attr_idx]
        prob = current_probs[:, active_attr_idx]
        sign = torch.sign(target - prob).view(batch_size, 1, 1, 1) # [B, 1, 1, 1]
        
        # 2. Get Direction Vector
        direction_vec = F.normalize(self.directions[active_attr_idx], dim=0)
        direction = direction_vec.view(1, -1, 1, 1)  # [1, C, 1, 1]

        # 3. Calculate Step Size (Intensity)
        # Get IG stats [B, 8]
        ig_stats = self.get_ig_features(ig_map)

        context = z_list[0].mean(dim=[2, 3])
        step_input = torch.cat([context, ig_stats], dim=1)

        step_logits = self.step_predictors[active_attr_idx](step_input)

        steps = []
        for level_idx in range(step_logits.size(1)):
            step = torch.sigmoid(step_logits[:, level_idx]) * 1.3 + 0.2
            steps.append(step)

        for level_idx, (z, step) in enumerate(zip(z_list, steps)):
            step_reshaped = step.view(batch_size, 1, 1, 1)
            step_values.append(step_reshaped)

            mask = cam_masks[level_idx].unsqueeze(1)

            delta = sign * step_reshaped * direction * mask
            z_mutated[level_idx] = z + delta

        return z_mutated, step_values