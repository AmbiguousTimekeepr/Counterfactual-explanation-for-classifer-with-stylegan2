"""CelebA dataset helpers for multi-label classification."""
from __future__ import annotations

import os
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CelebADataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str,
        transform=None,
        attribute_map: Optional[Mapping[str, str]] = None,
        return_image_name: bool = False,
    ) -> None:
        self.df = df.copy()
        self.root_dir = root_dir
        self.transform = transform
        self.return_image_name = return_image_name
        self.attribute_names = list(self.df.columns)
        self.attribute_display_names = (
            [attribute_map.get(attr, attr) for attr in self.attribute_names]
            if attribute_map
            else self.attribute_names
        )
        self.img_names = self.df.index.astype(str).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        labels = self.df.iloc[idx].to_numpy(dtype=np.float32, copy=True)
        labels[labels == -1] = 0
        label_tensor = torch.from_numpy(labels)

        if self.transform:
            image = self.transform(image)

        if self.return_image_name:
            return image, label_tensor, img_name
        return image, label_tensor


def load_attribute_dataframe(
    csv_path: str,
    attributes: Sequence[str],
    image_id_column: Optional[str] = None,
    drop_missing: bool = True,
) -> pd.DataFrame:
    """Load CelebA attributes CSV and ensure expected structure."""

    df = pd.read_csv(csv_path)

    if image_id_column and image_id_column in df.columns:
        df.set_index(image_id_column, inplace=True)
    elif df.index.name is None or df.index.name not in (image_id_column or "image_id",):
        candidate = image_id_column or "image_id"
        if candidate in df.columns:
            df.set_index(candidate, inplace=True)
        else:
            df.set_index(df.columns[0], inplace=True)

    missing = [attr for attr in attributes if attr not in df.columns]
    if missing:
        if drop_missing:
            raise ValueError(f"Missing required CelebA attributes: {missing}")
        for attr in missing:
            df[attr] = 0

    return df.loc[:, list(attributes)].copy()


def map_attribute_names(
    attributes: Iterable[str], attribute_map: Optional[Mapping[str, str]] = None
) -> Sequence[str]:
    """Map attribute identifiers to human-readable aliases."""

    if attribute_map is None:
        return list(attributes)
    return [attribute_map.get(attr, attr) for attr in attributes]


__all__ = [
    "CelebADataset",
    "load_attribute_dataframe",
    "map_attribute_names",
]
