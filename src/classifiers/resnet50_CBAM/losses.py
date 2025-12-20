"""Loss functions for CelebA attribute classification."""
from __future__ import annotations

import torch
import torch.nn as nn


class AsymmetricLossOptimized(nn.Module):
    """Implementation of Asymmetric Loss for multi-label classification.

    This variant follows the formulation from "Asymmetric Loss For Multi-Label
    Classification" (Ridnik et al., 2021) and supports optional clipping of the
    negative probabilities for improved numerical stability.
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the asymmetric loss.

        Args:
            logits: Raw model outputs of shape (B, C).
            targets: Multi-label targets in {0, 1} with shape (B, C).
        """
        targets = targets.float()
        probs = torch.sigmoid(logits)
        prob_pos = probs
        prob_neg = 1.0 - probs

        if self.clip is not None and self.clip > 0:
            prob_neg = torch.clamp(prob_neg + self.clip, max=1.0)

        log_pos = torch.log(prob_pos.clamp(min=self.eps))
        log_neg = torch.log(prob_neg.clamp(min=self.eps))

        focal_pos = torch.pow(1.0 - prob_pos, self.gamma_pos)
        focal_neg = torch.pow(prob_pos, self.gamma_neg)

        loss_pos = targets * focal_pos * log_pos
        loss_neg = (1.0 - targets) * focal_neg * log_neg
        loss = loss_pos + loss_neg

        return -loss.mean()


__all__ = ["AsymmetricLossOptimized"]
