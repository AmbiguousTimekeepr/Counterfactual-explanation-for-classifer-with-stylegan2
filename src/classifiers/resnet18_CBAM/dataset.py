"""Dataset helpers tailored for the ResNet18+CBAM workflow."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import pandas as pd
from torch.utils.data import Dataset

from ..dataset import CelebADataset as _BaseCelebADataset  # type: ignore[attr-defined]
from ..dataset import load_attribute_dataframe as _load_attribute_dataframe  # type: ignore[attr-defined]
from ..dataset import map_attribute_names as _map_attribute_names  # type: ignore[attr-defined]


class CelebADataset(_BaseCelebADataset):
    """Typed alias that exposes the shared CelebA dataset implementation."""

    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str,
        transform=None,
        attribute_map: Optional[Mapping[str, str]] = None,
        return_image_name: bool = False,
    ) -> None:
        super().__init__(
            df=df,
            root_dir=root_dir,
            transform=transform,
            attribute_map=attribute_map,
            return_image_name=return_image_name,
        )


def load_attribute_dataframe(
    csv_path: str,
    attributes: Sequence[str],
    image_id_column: Optional[str] = None,
    drop_missing: bool = True,
) -> pd.DataFrame:
    return _load_attribute_dataframe(
        csv_path=csv_path,
        attributes=attributes,
        image_id_column=image_id_column,
        drop_missing=drop_missing,
    )


def map_attribute_names(
    attributes: Sequence[str], attribute_map: Optional[Mapping[str, str]] = None
) -> Sequence[str]:
    return _map_attribute_names(attributes, attribute_map)


__all__ = [
    "CelebADataset",
    "load_attribute_dataframe",
    "map_attribute_names",
]
