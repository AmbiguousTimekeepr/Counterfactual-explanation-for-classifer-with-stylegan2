"""Compatibility layer for classifier loss functions."""
from __future__ import annotations

from .resnet50_CBAM.losses import AsymmetricLossOptimized

__all__ = ["AsymmetricLossOptimized"]
