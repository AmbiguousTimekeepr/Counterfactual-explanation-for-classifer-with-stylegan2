import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CelebADataset(Dataset):
    """
    Unsupervised CelebA dataset for VQ-VAE training.
    Returns only images, no labels.
    """
    def __init__(self, cfg):
        self.img_dir = os.path.join(cfg.data_path, "img_align_celeba")
        
        # List all images
        self.img_names = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))])
        
        self.transform = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            img = torch.zeros(3, 128, 128)
        
        return img

def get_loader(cfg):
    dataset = CelebADataset(cfg)
    return DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers, 
        pin_memory=cfg.pin_memory
    )