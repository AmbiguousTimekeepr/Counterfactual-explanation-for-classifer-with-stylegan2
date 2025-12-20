"""Compatibility layer for inference helpers."""
from __future__ import annotations

from .resnet50_CBAM.inference import inference_single_image

__all__ = ["inference_single_image"]
