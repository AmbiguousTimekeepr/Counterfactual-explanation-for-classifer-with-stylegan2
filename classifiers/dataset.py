"""
CelebA Dataset for Multi-label Classification
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class CelebADataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        """
        Args:
            df: DataFrame with image_id as index and attributes as columns
            root_dir: Directory with all the images
            transform: Optional transform to be applied on a sample
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.img_names = self.df.index.tolist()  # Image_id is index
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        
        # Lấy labels, chuyển -1 thành 0
        # Label trong CSV CelebA: 1 (có), -1 (không)
        labels = torch.tensor(self.df.iloc[idx].values, dtype=torch.float32)
        labels[labels == -1] = 0 
        
        if self.transform:
            image = self.transform(image)
            
        return image, labels
