import argparse
import json
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ..attributes import SELECTED_ATTRIBUTES
from ..dataset import CelebADataset
from ..model import ResNet50_CBAM, SquarePadResize


def build_dataloaders(
    csv_path: str,
    image_dir: str,
    batch_size: int,
    image_size: int,
    num_workers: int,
    val_split: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader, torch.Tensor, Sequence[str], pd.DataFrame, pd.DataFrame]:
    """Create train/validation dataloaders and class weights."""

    df_attr = pd.read_csv(csv_path, index_col=0)
    missing = [attr for attr in SELECTED_ATTRIBUTES if attr not in df_attr.columns]
    if missing:
        raise ValueError(f"Missing required CelebA attributes: {missing}")

    df_attr = df_attr.loc[:, SELECTED_ATTRIBUTES]
    train_df, val_df = train_test_split(df_attr, test_size=val_split, random_state=seed)

    train_labels = train_df.replace(-1, 0).values
    num_pos = np.sum(train_labels, axis=0)
    num_neg = len(train_df) - num_pos
    pos_weights = torch.tensor(num_neg / (num_pos + 1e-5), dtype=torch.float32)

    train_transform = transforms.Compose([
        SquarePadResize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
    ])

    val_transform = transforms.Compose([
        SquarePadResize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CelebADataset(train_df, image_dir, transform=train_transform)
    val_dataset = CelebADataset(val_df, image_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    return train_loader, val_loader, pos_weights, SELECTED_ATTRIBUTES, train_df, val_df


def create_model(num_classes: int, device: torch.device) -> nn.Module:
    """Initialise the ResNet50 + CBAM model."""

    model = ResNet50_CBAM(num_classes=num_classes)
    return model.to(device)


def train_resnet50_cbam(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[_LRScheduler],
    device: torch.device,
    num_epochs: int,
    attribute_names: Sequence[str],
    checkpoint_dir: str,
    checkpoint_prefix: str = "resnet50_cbam",
    save_every: int = 5,
    threshold: float = 0.5,
) -> Dict[str, Sequence[float]]:
    """Train ResNet50+CBAM with validation tracking and checkpointing."""

    os.makedirs(checkpoint_dir, exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "val_attr_acc": []}
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")

        for images, labels in train_iter:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            train_iter.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / len(train_loader.dataset)
        history["train_loss"].append(epoch_train_loss)

        current_lr = None
        if scheduler is not None:
            scheduler.step()
            if hasattr(scheduler, "get_last_lr"):
                current_lr = scheduler.get_last_lr()[0]

        model.eval()
        val_running_loss = 0.0
        all_preds = []
        all_targets = []
        val_iter = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]")

        with torch.no_grad():
            for images, labels in val_iter:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        history["val_loss"].append(epoch_val_loss)

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        binary_preds = (all_preds > threshold).astype(int)
        correct_counts = np.sum(binary_preds == all_targets, axis=0)
        acc_per_attr = correct_counts / len(val_loader.dataset)
        history["val_attr_acc"].append(acc_per_attr.tolist())
        mean_acc = float(np.mean(acc_per_attr))

        header = "-" * 60
        print(f"\n--- Epoch {epoch} Report ---")
        if current_lr is not None:
            print(
                f"LR: {current_lr:.6f} | Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} | Mean Val Acc: {mean_acc:.4f}"
            )
        else:
            print(
                f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} "
                f"| Mean Val Acc: {mean_acc:.4f}"
            )
        print(header)
        print(f"{'Attribute':<25} | {'Accuracy':<10} | {'Sample Prob (Mean)':<15}")
        print(header)
        mean_probs = np.mean(all_preds, axis=0)
        for idx, attr_name in enumerate(attribute_names):
            print(f"{attr_name:<25} | {acc_per_attr[idx]:.4f}     | {mean_probs[idx]:.4f}")
        print(header)

        if save_every and epoch % save_every == 0:
            ckpt_name = f"{checkpoint_prefix}_epoch_{epoch}.pth"
            save_path = os.path.join(checkpoint_dir, ckpt_name)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_val_loss,
                },
                save_path,
            )
            print(f"Saved checkpoint: {save_path}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))

    print("Training Complete.")
    return history


def save_history(history: Dict[str, Sequence[float]], output_dir: str) -> None:
    """Persist training history to disk."""

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet50 + CBAM on CelebA attributes")
    parser.add_argument("csv", help="Path to CelebA attribute CSV containing selected attributes")
    parser.add_argument("images", help="Directory with CelebA images")
    parser.add_argument("--output", default="outputs/cnn_classifier", help="Directory to store checkpoints")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, val_loader, pos_weights, attribute_names, train_df, val_df = build_dataloaders(
        csv_path=args.csv,
        image_dir=args.images,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
    )

    model = create_model(num_classes=len(attribute_names), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    history = train_resnet50_cbam(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        attribute_names=attribute_names,
        checkpoint_dir=args.output,
        checkpoint_prefix="resnet50_cbam",
        save_every=5,
        threshold=0.5,
    )

    save_history(history, args.output)
    train_df.to_csv(os.path.join(args.output, "train_split.csv"))
    val_df.to_csv(os.path.join(args.output, "val_split.csv"))


if __name__ == "__main__":
    main()

