import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image


SELECTED_ATTRIBUTES = [
    'Bald',
    'Bangs',
    'Black_Hair',
    'Blond_Hair',
    'Brown_Hair',
    'Bushy_Eyebrows',
    'Eyeglasses',
    'Male',
    'Mouth_Slightly_Open',
    'Mustache',
    'Pale_Skin',
    'Young',
]


class CelebADataset(Dataset):
    """
    CelebA dataset loader with attribute annotations
    
    Expected directory structure:
    data/CelebA/
    ├── img_align_celeba/          (128x128 or 256x256 aligned face images)
    ├── list_attr_celeba.csv       (CSV with header: index, attr1, attr2, ...)
    ├── identity_CelebA.csv        (identity mapping, optional)
    ├── list_bbox_celeba.csv       (bounding boxes, optional)
    └── list_eval_partition.csv    (train/val/test split, optional)
    """
    
    def __init__(self, 
                 root_dir,
                 split='train',
                 image_size=128,
                 num_attributes=40,
                 transform=None,
                 return_filename=False,
                 use_eval_partition=False):
        """
        Args:
            root_dir: Path to CelebA root directory
            split: 'train', 'val', or 'test'
            image_size: Target image size (e.g., 128)
            num_attributes: Original argument preserved for compatibility (ignored; SELECTED_ATTRIBUTES used)
            transform: Image transforms
            return_filename: Whether to return filename in __getitem__
            use_eval_partition: If True, use list_eval_partition.csv for splits
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        if num_attributes not in (None, len(SELECTED_ATTRIBUTES)):
            print(
                f"Warning: overriding requested num_attributes={num_attributes} "
                f"with len(SELECTED_ATTRIBUTES)={len(SELECTED_ATTRIBUTES)}"
            )
        self.num_attributes = len(SELECTED_ATTRIBUTES)
        self.return_filename = return_filename
        self.use_eval_partition = use_eval_partition
        
        # Image directory
        self.img_dir = self.root_dir / "img_align_celeba"
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        
        # Load attribute file (CSV format)
        attr_file = self.root_dir / "list_attr_celeba.csv"
        if not attr_file.exists():
            raise FileNotFoundError(f"Attribute file not found: {attr_file}")
        
        # Parse attributes file using pandas
        df = pd.read_csv(attr_file)
        
        # First column is typically the image filename
        self.image_names = df.iloc[:, 0].values.tolist()
        
        # Attribute names restricted to SELECTED_ATTRIBUTES
        available_attrs = set(df.columns[1:])
        missing_attrs = [attr for attr in SELECTED_ATTRIBUTES if attr not in available_attrs]
        if missing_attrs:
            raise ValueError(f"Missing required attributes in CSV: {missing_attrs}")

        self.attr_names = SELECTED_ATTRIBUTES

        # Get attribute values [convert -1 to 0, keep 1 as 1]
        attrs_matrix = df[self.attr_names].replace(-1, 0).astype(np.int32).values
        self.attributes = [row for row in attrs_matrix]
        
        # Split dataset
        if use_eval_partition:
            # Use official list_eval_partition.csv if available
            partition_file = self.root_dir / "list_eval_partition.csv"
            if partition_file.exists():
                self._load_eval_partition(partition_file)
            else:
                self._use_default_split()
        else:
            # Use default 70/10/20 split
            self._use_default_split()
        
        # Default transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])
        else:
            self.transform = transform
        
        print(f"Loaded {split} split: {len(self.image_names)} images")
    
    def _use_default_split(self):
        """Use 70/10/20 train/val/test split"""
        total = len(self.image_names)
        train_size = int(0.7 * total)
        val_size = int(0.1 * total)
        
        if self.split == 'train':
            indices = range(0, train_size)
        elif self.split == 'val':
            indices = range(train_size, train_size + val_size)
        elif self.split == 'test':
            indices = range(train_size + val_size, total)
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        self.image_names = [self.image_names[i] for i in indices]
        self.attributes = [self.attributes[i] for i in indices]
    
    def _load_eval_partition(self, partition_file):
        """
        Load official CelebA eval partition file from CSV.
        Expected format: filename,partition (where partition: 0=train, 1=val, 2=test)
        """
        partition_map = {'train': 0, 'val': 1, 'test': 2}
        target_partition = partition_map[self.split]
        
        # Read partition file
        partition_df = pd.read_csv(partition_file)
        partitions = {}
        
        # Handle different column name formats
        img_col = partition_df.columns[0]
        part_col = partition_df.columns[1]
        
        for idx, row in partition_df.iterrows():
            img_name = str(row[img_col]).strip()
            partition = int(row[part_col])
            partitions[img_name] = partition
        
        # Filter images by partition
        filtered_names = []
        filtered_attrs = []
        
        for img_name, attrs in zip(self.image_names, self.attributes):
            if partitions.get(img_name) == target_partition:
                filtered_names.append(img_name)
                filtered_attrs.append(attrs)
        
        self.image_names = filtered_names
        self.attributes = filtered_attrs
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor [3, H, W] normalized to [-1, 1]
            attributes: Tensor [num_attributes] with values in {0, 1}
            filename: (optional) Image filename
        """
        img_name = self.image_names[idx]
        img_path = self.img_dir / img_name
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy image if loading fails
            image = Image.new('RGB', (self.image_size, self.image_size))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get attributes
        attrs = torch.tensor(self.attributes[idx], dtype=torch.float32)
        
        if self.return_filename:
            return image, attrs, img_name
        else:
            return image, attrs


def get_loader(cfg, split='train', batch_size=None, num_workers=None, shuffle=None, return_filename=False):
    """
    Create a DataLoader for CelebA
    
    Args:
        cfg: Config object
        split: 'train', 'val', or 'test'
        batch_size: Batch size (uses cfg.batch_size if None)
        num_workers: Number of workers (uses cfg.num_workers if None)
        shuffle: Whether to shuffle (True for train, False otherwise)
    
    Returns:
        DataLoader
    """
    if batch_size is None:
        batch_size = cfg.batch_size
    if num_workers is None:
        num_workers = cfg.num_workers
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataset = CelebADataset(
        root_dir=cfg.data_root,
        split=split,
        image_size=cfg.image_size,
        num_attributes=cfg.num_attributes,
        return_filename=return_filename,
        use_eval_partition=getattr(cfg, 'use_eval_partition', False)
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else 0
    )
    
    return loader


def get_attribute_names():
    """Get the restricted CelebA attribute names used by this project"""
    return SELECTED_ATTRIBUTES
