"""Compatibility layer for ResNet-based classifier models."""
from __future__ import annotations

from .resnet50_CBAM.model import ResNet50_CBAM, SquarePadResize

__all__ = ["ResNet50_CBAM", "SquarePadResize"]
