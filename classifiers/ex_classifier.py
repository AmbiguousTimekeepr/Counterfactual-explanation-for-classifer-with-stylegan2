import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np

CELEBA_39_ATTRIBUTES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Brown_Hair",
    "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
    "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
    "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
    "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young"
]

class CelebAClassifierDataset(Dataset):
    def __init__(self, images_dir, attr_file, transform=None, attributes=CELEBA_39_ATTRIBUTES):
        self.images_dir = images_dir
        self.transform = transform
        self.selected_attributes = attributes

        # Robust attribute file loading
        if attr_file.endswith('.csv'):
            df = pd.read_csv(attr_file)
        else:
            df = pd.read_csv(attr_file, delim_whitespace=True, header=None)
            if len(df.columns) == len(self.selected_attributes) + 1:
                df.columns = ['image_id'] + self.selected_attributes
            else:
                df = pd.read_csv(attr_file, delim_whitespace=True, header=1)
                if 'index' in df.columns:
                    df = df.reset_index()
                    df.rename(columns={'index': 'image_id'}, inplace=True)
                elif df.columns[0] not in ['image_id', 'image']:
                    df.rename(columns={df.columns[0]: 'image_id'}, inplace=True)

        available_imgs = set(os.listdir(images_dir))
        if 'image_id' not in df.columns:
            df['image_id'] = df[df.columns[0]]
        self.df = df[df['image_id'].isin(available_imgs)].copy()
        
        # **SAFEGUARD: Reorder columns by selected_attributes to ensure alignment**
        missing_attrs = [attr for attr in self.selected_attributes if attr not in self.df.columns]
        if missing_attrs:
            raise ValueError(f"Missing attributes in CSV: {missing_attrs}")
        self.df = self.df[['image_id'] + self.selected_attributes]
        
        for attr in self.selected_attributes:
            self.df[attr] = (self.df[attr] + 1) // 2
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row['image_id'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # **SAFEGUARD: Labels in selected_attributes order**
        labels = torch.tensor(row[self.selected_attributes].values.astype(np.float32))
        return image, labels

def create_classifier(model_name: str, num_classes: int, pretrained=True):
    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(1280, num_classes)
        target_layer = model.features[-1]
        target_layer_name = 'features_last'
    elif model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(pretrained=pretrained)
        model.classifier[3] = nn.Linear(1024, num_classes)
        target_layer = model.features[-1]
        target_layer_name = 'features_last'
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(512, num_classes)
        target_layer = model.layer4
        target_layer_name = 'layer4'
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
        model.fc = nn.Linear(512, num_classes)
        target_layer = model.layer4
        target_layer_name = 'layer4'
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model, target_layer, target_layer_name

class ExplainableClassifier(nn.Module):
    def __init__(self, model_name='mobilenet_v2', num_classes=39, attribute_names=CELEBA_39_ATTRIBUTES):
        super().__init__()
        self.model, self.target_layer, self.target_layer_name = create_classifier(model_name, num_classes)
        self.attribute_names = attribute_names
        self.gradients = {}
        self.activations = {}
        def save_grad(name):
            def hook(module, gin, gout):
                self.gradients[name] = gout[0].detach()
            return hook
        def save_act(name):
            def hook(m, i, o): self.activations[name] = o.detach()
            return hook
        self.target_layer.register_forward_hook(save_act(self.target_layer_name))
        self.target_layer.register_full_backward_hook(save_grad(self.target_layer_name))
    def forward(self, x):
        return self.model(x)

def load_attributes_by_name(attr_file, image_names, attribute_names, img_id_col=None):
    """
    Load attributes from CSV/file and reorder by attribute_names to ensure alignment.
    
    Args:
        attr_file: Path to attribute CSV
        image_names: List of image filenames to extract
        attribute_names: List of attribute names in desired order
        img_id_col: Column name for image ID (auto-detect if None)
    
    Returns:
        dict: {image_name: torch.Tensor of shape (len(attribute_names),)}
    """
    if attr_file.endswith('.csv'):
        df = pd.read_csv(attr_file)
    else:
        df = pd.read_csv(attr_file, delim_whitespace=True, header=None)
        if len(df.columns) == len(attribute_names) + 1:
            df.columns = ['image_id'] + attribute_names
        else:
            df = pd.read_csv(attr_file, delim_whitespace=True, header=1)
            if 'index' in df.columns:
                df = df.reset_index()
                df.rename(columns={'index': 'image_id'}, inplace=True)
            elif df.columns[0] not in ['image_id', 'image']:
                df.rename(columns={df.columns[0]: 'image_id'}, inplace=True)
    
    # Auto-detect image ID column
    if img_id_col is None:
        if 'image_id' in df.columns:
            img_id_col = 'image_id'
        elif 'image' in df.columns:
            img_id_col = 'image'
        else:
            img_id_col = df.columns[0]
    
    # **SAFEGUARD: Reorder columns by attribute_names**
    missing_attrs = [attr for attr in attribute_names if attr not in df.columns]
    if missing_attrs:
        raise ValueError(f"Missing attributes in {attr_file}: {missing_attrs}")
    
    df = df[[img_id_col] + attribute_names]
    
    # Normalize attribute values: convert -1/1 to 0/1
    for attr in attribute_names:
        df[attr] = (df[attr] + 1) // 2
    
    result = {}
    for img_name in image_names:
        rows = df[df[img_id_col] == img_name]
        if len(rows) > 0:
            row = rows.iloc[0]
            attrs = torch.tensor(row[attribute_names].values.astype(np.float32))
            result[img_name] = attrs
    
    return result

def load_trained_model(model_path, model_name, num_classes=39, attribute_names=CELEBA_39_ATTRIBUTES, device=None):
    """
    Load a trained ExplainableClassifier from a checkpoint.
    Args:
        model_path: Path to .pth file
        model_name: Model architecture name
        num_classes: Number of output classes
        attribute_names: List of attribute names
        device: torch.device or None
    Returns:
        ExplainableClassifier instance with loaded weights
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExplainableClassifier(model_name, num_classes, attribute_names).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model