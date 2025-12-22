"""
Example script demonstrating how to use the classifier module
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Import from classifiers module
from classifiers import (
    ResNet18_CBAM, 
    SquarePadResize, 
    CelebADataset,
    visualize_gradcam,
    visualize_integrated_gradients,
    inference_single_image
)


# --- CONFIGURATION ---
BATCH_SIZE = 64
IMAGE_SIZE = 224
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
CSV_PATH = r"/mnt/c/KLTN/LDM/data/list_attr_celeba.csv"
IMAGE_PATH = r"/mnt/c/KLTN/LDM/data/img_align_celeba/img_align_celeba"
PATH_CHECKPOINT = r"/mnt/e/KLTN/GAN/classifier/checkpoints/cnn_classfier"
NUM_CLASSES = 40
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
WEIGHT_DECAY = 1e-4

os.makedirs(PATH_CHECKPOINT, exist_ok=True)


def main():
    # --- LOAD DATA ---
    print("Loading CSV metadata...")
    df_attr = pd.read_csv(CSV_PATH, index_col=0)
    attribute_names = df_attr.columns.tolist()

    # Split Train/Val (90/10)
    train_df, val_df = train_test_split(df_attr, test_size=0.1, random_state=42)

    # Calculate pos_weight for loss function
    print("Calculating positive weights for loss function...")
    train_labels = train_df.replace(-1, 0).values
    num_pos = np.sum(train_labels, axis=0)
    num_neg = len(train_df) - num_pos
    pos_weights_tensor = torch.tensor(num_neg / (num_pos + 1e-5), dtype=torch.float32).to(DEVICE)

    # --- TRANSFORMS ---
    train_transform = transforms.Compose([
        SquarePadResize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
    ])

    val_transform = transforms.Compose([
        SquarePadResize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    viz_transform = transforms.Compose([
        SquarePadResize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    # --- DATASETS ---
    train_dataset = CelebADataset(train_df, IMAGE_PATH, transform=train_transform)
    val_dataset = CelebADataset(val_df, IMAGE_PATH, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=8, pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=8, pin_memory=True, prefetch_factor=4)

    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

    # --- MODEL SETUP ---
    print(f"Initializing ResNet18 + CBAM on {DEVICE}...")
    model = ResNet18_CBAM(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # --- LOAD TRAINED MODEL ---
    model.load_state_dict(torch.load(os.path.join(PATH_CHECKPOINT, "best_model.pth")))
    model.eval()

    # --- INFERENCE EXAMPLE ---
    print("\n" + "="*80)
    print("Running Inference Example")
    print("="*80)
    
    sample_img_path = os.path.join(IMAGE_PATH, val_df.index[132])
    
    inference_single_image(
        model=model,
        image_path=sample_img_path,
        transform=val_transform,
        attribute_names=attribute_names,
        device=DEVICE,
        df_attr=df_attr,
        threshold=0.5
    )

    # --- VISUALIZATION EXAMPLES ---
    attr_idx = attribute_names.index("Brown_Hair")
    
    print("\n" + "="*80)
    print(f"Visualizing explanations for attribute: {attribute_names[attr_idx]}")
    print("="*80)

    # Grad-CAM for POSITIVE class
    print("\n=== Grad-CAM for POSITIVE class ===")
    visualize_gradcam(
        model=model,
        image_path=sample_img_path,
        attribute_idx=attr_idx,
        attribute_name=attribute_names[attr_idx],
        transform=val_transform,
        viz_transform=viz_transform,
        device=DEVICE,
        target_layer=model.layer4,
        target_class=1,
        method='gradcam'
    )

    # Grad-CAM for NEGATIVE class
    print("\n=== Grad-CAM for NEGATIVE class ===")
    visualize_gradcam(
        model=model,
        image_path=sample_img_path,
        attribute_idx=attr_idx,
        attribute_name=attribute_names[attr_idx],
        transform=val_transform,
        viz_transform=viz_transform,
        device=DEVICE,
        target_layer=model.layer4,
        target_class=0,
        method='gradcam'
    )

    # Grad-CAM++ for POSITIVE class
    print("\n=== Grad-CAM++ for POSITIVE class ===")
    visualize_gradcam(
        model=model,
        image_path=sample_img_path,
        attribute_idx=attr_idx,
        attribute_name=attribute_names[attr_idx],
        transform=val_transform,
        viz_transform=viz_transform,
        device=DEVICE,
        target_layer=model.cbam4,
        target_class=1,
        method='gradcam++'
    )

    # Integrated Gradients for POSITIVE class
    print("\n=== Integrated Gradients for POSITIVE class ===")
    visualize_integrated_gradients(
        model=model,
        image_path=sample_img_path,
        attribute_idx=attr_idx,
        attribute_name=attribute_names[attr_idx],
        transform=val_transform,
        viz_transform=viz_transform,
        device=DEVICE,
        target_class=1,
        steps=50
    )

    # Integrated Gradients for NEGATIVE class
    print("\n=== Integrated Gradients for NEGATIVE class ===")
    visualize_integrated_gradients(
        model=model,
        image_path=sample_img_path,
        attribute_idx=attr_idx,
        attribute_name=attribute_names[attr_idx],
        transform=val_transform,
        viz_transform=viz_transform,
        device=DEVICE,
        target_class=0,
        steps=50
    )


if __name__ == "__main__":
    main()
