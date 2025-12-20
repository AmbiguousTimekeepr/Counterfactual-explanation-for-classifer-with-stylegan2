"""Compatibility layer for shared dataset utilities."""
from __future__ import annotations

from .resnet50_CBAM.dataset import (
    CelebADataset,
    load_attribute_dataframe,
    map_attribute_names,
)

__all__ = [
    "CelebADataset",
    "load_attribute_dataframe",
    "map_attribute_names",
]
