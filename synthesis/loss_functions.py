# synthesis/loss_functions.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import lpips


class CounterfactualLossManager(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

        # Core losses
        self.lpips = lpips.LPIPS(net='vgg').to(device).eval()

        # ✅ FIXED: Properly slice VGG features using nn.Sequential
        vgg_full = models.vgg16(pretrained=True).features
        # Extract layers 0-15 (first 16 layers) and wrap in Sequential
        vgg_layers = nn.Sequential(*list(vgg_full.children())[:16])
        self.vgg = vgg_layers.to(device).eval()
        
        for p in self.vgg.parameters():
            p.requires_grad = False
        
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        self.base_weights = {
            'cf': 10.0,
            'retention': 5.0,
            'latent_prox': 5.0,
            'ortho': 0.3,
            'sparse': 1e-3
        }

    # ===============================================================
    # 1. Counterfactual Classification Loss (Margin-based)
    # ===============================================================
    def counterfactual_loss(self, logits_new, target_labels, current_probs, flip_mask):
        """
        Margin-based loss for counterfactual generation.
        
        Strategy:
        - For attributes that NEED to flip: strong penalty if wrong direction
        - For attributes that DON'T need to flip: weak penalty if they drift
        - Uses margin targets instead of hard 0/1 targets
        
        Args:
            logits_new: Raw model outputs [B, num_attrs]
            target_labels: Target attribute values [B, num_attrs] (0 or 1)
            current_probs: Current predictions before editing [B, num_attrs]
            flip_mask: Which attributes MUST flip [B, num_attrs] (0 or 1)
        
        Returns:
            scalar loss
        """
        with torch.no_grad():
            current_pred = (current_probs > 0.5).float()
            target_pred = target_labels
            
            # Which attributes need to flip (change from current state)
            needs_flip = (current_pred != target_pred).float() * flip_mask  # [B, num_attrs]
            
            # Margin targets: push predictions closer to margin bounds
            # If target=1: aim for 0.9 (confident positive)
            # If target=0: aim for 0.1 (confident negative)
            margin_targets = target_labels * 0.9 + (1 - target_labels) * 0.1  # [B, num_attrs]
        
        # ✅ NEW: Margin loss instead of BCE
        # For sigmoid: margin_loss penalizes distance from margin_targets
        probs_new = torch.sigmoid(logits_new)
        
        # Margin-based penalty: L2 distance from target margin
        margin_loss_per_attr = (probs_new - margin_targets).pow(2)  # [B, num_attrs]
        
        # Weight by flip requirement:
        # - Attributes that NEED to flip: weight=5.0 (strong push)
        # - Attributes that DON'T flip: weight=0.1 (allow natural change)
        weights = needs_flip * 5.0 + (1 - needs_flip) * 0.1  # [B, num_attrs]
        
        weighted_margin_loss = margin_loss_per_attr * weights
        
        return weighted_margin_loss.mean()

    # ===============================================================
    # 2. Masked Identity Retention (L1 + LPIPS + SSIM) — THE SCALPEL
    # ===============================================================
    def masked_retention_loss(self, x_orig, x_new, cam_masks):
        """
        cam_masks: list of 3 tensors [B, H_l, W_l] for each level (8x8, 16x16, 32x32)
        We preserve ONLY where ALL masks are low → true background
        """
        with torch.no_grad():
            # Combine all Grad-CAM masks (logical AND of edit regions)
            # First add channel dimension if needed: [B, H, W] -> [B, 1, H, W]
            masks_4d = [m.unsqueeze(1) if m.ndim == 3 else m for m in cam_masks]
            
            # Interpolate all to 128x128
            combined_mask = torch.clamp_max(torch.stack([
                F.interpolate(m, size=(128, 128), mode='bilinear', align_corners=False) 
                for m in masks_4d
            ]).sum(0), 1.0)  # [B, 1, 128, 128]

            # Invert: preserve where mask is low (background)
            preserve_mask = 1.0 - combined_mask.clamp(0, 1)  # [B, 1, 128, 128]

        # L1 loss only in preserved regions
        l1 = (F.l1_loss(x_new, x_orig, reduction='none') * preserve_mask).mean()
        
        # LPIPS loss
        lpips_loss = self.lpips(x_new, x_orig)
        lpips_loss = (lpips_loss * preserve_mask.mean(dim=[1, 2, 3])).mean()

        return l1 + lpips_loss * 0.8

    # ===============================================================
    # 3. Latent Proximity (Minimal Code Change)
    # ===============================================================
    def latent_proximity_loss(self, z_orig_list, z_edited_list, cam_masks):
        loss = 0.0
        for z_orig, z_edited, mask in zip(z_orig_list, z_edited_list, cam_masks):
            # ✅ DETACH z_orig since encoder is frozen
            z_orig = z_orig.detach()
            # ✅ DETACH mask (no gradient needed)
            mask = mask.detach()
            
            # Ensure mask is 4D [B, 1, H, W]
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
            
            # Now interpolate safely
            mask_resized = F.interpolate(mask, size=z_orig.shape[2:], mode='nearest')
            edit_region = mask_resized > 0.3
            preserve_region = ~edit_region

            # Only penalize changes in preserved regions
            diff = (z_orig - z_edited).abs()
            loss += (diff * preserve_region.float()).mean() * 10.0
        return loss / len(z_orig_list)

    # ===============================================================
    # 4. Orthogonality + Direction Sparsity
    # ===============================================================
    def orthogonality_loss(self, directions):
        d_norm = F.normalize(directions, dim=1)
        gram = torch.mm(d_norm, d_norm.t())
        off_diagonal = gram - torch.diag(torch.diag(gram))
        return off_diagonal.abs().mean()

    def direction_sparsity(self, directions):
        return directions.abs().mean()

    # ===============================================================
    # 5. Full Generator Loss
    # ===============================================================
    def generator_loss(self,
                       x_orig, x_new,
                       logits_new, target_labels, current_probs,
                       flip_mask,
                       z_orig_list, z_edited_list,
                       cam_masks,
                       directions,
                       weights=None):
        if weights is None:
            weights = self.base_weights

        losses = {}

        losses['cf'] = self.counterfactual_loss(logits_new, target_labels, current_probs, flip_mask)
        losses['retention'] = self.masked_retention_loss(x_orig, x_new, cam_masks)
        losses['latent_prox'] = self.latent_proximity_loss(z_orig_list, z_edited_list, cam_masks)
        losses['ortho'] = self.orthogonality_loss(directions)
        losses['sparse'] = self.direction_sparsity(directions)

        total = (weights['cf'] * losses['cf'] +
                 weights['retention'] * losses['retention'] +
                 weights['latent_prox'] * losses['latent_prox'] +
                 weights['ortho'] * losses['ortho'] +
                 weights['sparse'] * losses['sparse'])

        losses['total'] = total
        return losses

    def __call__(self, x_orig, x_new, logits_target, target_labels, directions, mask=None):
        """
        Legacy forward method for backward compatibility.
        Returns dict with individual loss components.
        """
        batch_size = logits_target.size(0)
        
        # Counterfactual loss (simplified)
        target_labels_binary = target_labels.float()
        cf_loss = F.binary_cross_entropy_with_logits(
            logits_target.unsqueeze(1), target_labels_binary.unsqueeze(1)
        )
        
        # Retention loss (L1)
        if mask is not None:
            ret_l1 = (F.l1_loss(x_new, x_orig, reduction='none') * mask).mean()
        else:
            ret_l1 = F.l1_loss(x_new, x_orig)
        
        # Perception loss (LPIPS)
        perc = self.lpips(x_new, x_orig).mean()
        
        # Orthogonality loss
        ortho = self.orthogonality_loss(directions)
        
        return {
            'cf': cf_loss,
            'ret_l1': ret_l1,
            'perc': perc,
            'ortho': ortho,
            'total': cf_loss + ret_l1 + perc + ortho
        }