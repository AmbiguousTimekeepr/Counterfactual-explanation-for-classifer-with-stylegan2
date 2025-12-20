"""Training utilities for the ResNet18+CBAM classifier."""
from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ...losses import AsymmetricLossOptimized  # type: ignore[import]
from ..attributes import ATTR_MAP, ATTRIBUTE_DISPLAY_NAMES, SELECTED_ATTRIBUTES  # type: ignore[import]
from ..dataset import CelebADataset, load_attribute_dataframe  # type: ignore[import]
from ..model import ResNet18_CBAM  # type: ignore[import]
from ..visualizations import (  # type: ignore[import]
    visualize_predictions,
    visualize_specific_attribute,
    visualize_specific_attribute_negative,
)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def build_datasets(
    csv_path: str,
    image_dir: str,
    attributes: Sequence[str],
    attribute_map: dict[str, str],
    image_size: int,
    batch_size: int,
    val_split: float,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, CelebADataset, CelebADataset, torch.Tensor, pd.DataFrame, pd.DataFrame]:
    df_attr = load_attribute_dataframe(csv_path, attributes)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df_attr))
    rng.shuffle(indices)

    val_size = int(len(indices) * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_df = df_attr.iloc[train_indices].copy()
    val_df = df_attr.iloc[val_indices].copy()

    train_transform, val_transform = build_transforms(image_size)

    train_dataset = CelebADataset(
        train_df,
        image_dir,
        transform=train_transform,
        attribute_map=attribute_map,
        return_image_name=True,
    )
    val_dataset = CelebADataset(
        val_df,
        image_dir,
        transform=val_transform,
        attribute_map=attribute_map,
        return_image_name=True,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    train_labels = train_df.replace(-1, 0).to_numpy(dtype=np.float32)
    num_pos = train_labels.sum(axis=0)
    num_neg = len(train_df) - num_pos
    pos_weights = torch.tensor(num_neg / (num_pos + 1e-5), dtype=torch.float32)

    return train_loader, val_loader, train_dataset, val_dataset, pos_weights, train_df, val_df


def _detach_batch(batch):
    if len(batch) == 3:
        return batch[0], batch[1], batch[2]
    if len(batch) == 2:
        return batch[0], batch[1], None
    raise ValueError("Unexpected batch format from DataLoader")


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: OneCycleLR,
    device: torch.device,
) -> float:
    model.train()
    epoch_loss = 0.0
    progress = tqdm(loader, desc="Train", leave=False)

    for batch in progress:
        images, labels, _ = _detach_batch(batch)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    return epoch_loss / len(loader)


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, torch.Tensor, torch.Tensor, float]:
    model.eval()
    epoch_loss = 0.0
    correct_counts = torch.zeros(len(SELECTED_ATTRIBUTES))
    total_counts = torch.zeros(len(SELECTED_ATTRIBUTES))
    prob_sums = torch.zeros(len(SELECTED_ATTRIBUTES))

    progress = tqdm(loader, desc="Val", leave=False)

    with torch.no_grad():
        for batch in progress:
            images, labels, _ = _detach_batch(batch)
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            labels_cpu = labels.cpu()
            preds_cpu = preds.cpu()
            probs_cpu = probs.cpu()

            correct_counts += (preds_cpu == labels_cpu).sum(dim=0)
            total_counts += labels_cpu.size(0)
            prob_sums += probs_cpu.sum(dim=0)

    avg_loss = epoch_loss / len(loader)
    attr_acc = (correct_counts / total_counts.clamp(min=1)).mul(100)
    avg_probs = prob_sums / total_counts.clamp(min=1)
    mean_acc = attr_acc.mean().item()
    return avg_loss, attr_acc, avg_probs, mean_acc


def train_resnet18_cbam(args: argparse.Namespace) -> dict[str, list]:
    attribute_map = ATTR_MAP.copy()
    if args.attr_map_json:
        with open(args.attr_map_json, "r", encoding="utf-8") as f:
            attribute_map.update(json.load(f))

    device = torch.device(args.device)

    (
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
        pos_weights,
        train_df,
        val_df,
    ) = build_datasets(
        csv_path=args.csv,
        image_dir=args.images,
        attributes=SELECTED_ATTRIBUTES,
        attribute_map=attribute_map,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = ResNet18_CBAM(num_classes=len(SELECTED_ATTRIBUTES)).to(device)
    if args.loss == "asl":
        criterion = AsymmetricLossOptimized(
            gamma_neg=args.gamma_neg,
            gamma_pos=args.gamma_pos,
            clip=args.asl_clip,
        )
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
    )

    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "per_attribute_accuracy": [],
        "per_attribute_prob": [],
        "mean_val_acc": [],
    }

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, attr_acc, avg_probs, mean_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["per_attribute_accuracy"].append(attr_acc.tolist())
        history["per_attribute_prob"].append(avg_probs.tolist())
        history["mean_val_acc"].append(mean_acc)

        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Mean Val Acc: {mean_acc:.2f}%"
        )
        print(f"{'Attribute':<20} | {'Acc (%)':<10} | {'Avg Prob':<10}")
        print("-" * 45)
        for name, acc, prob in zip(ATTRIBUTE_DISPLAY_NAMES, attr_acc, avg_probs):
            print(f"{name:<20} | {acc:.2f}%     | {prob:.4f}")

        if args.save_every and epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"resnet18_cbam_epoch_{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")

        if args.visualize_every and epoch % args.visualize_every == 0:
            viz_path = visualize_predictions(
                model,
                val_dataset,
                device,
                epoch,
                attribute_display=ATTRIBUTE_DISPLAY_NAMES,
                save_dir=args.checkpoint_dir,
                image_size=args.image_size,
            )
            print(f"Saved visualization to {viz_path}")

    history["train_split"] = train_df.index.tolist()
    history["val_split"] = val_df.index.tolist()
    history["attribute_names"] = list(SELECTED_ATTRIBUTES)
    history["attribute_display_names"] = list(ATTRIBUTE_DISPLAY_NAMES)

    if args.export_history:
        history_path = os.path.join(args.checkpoint_dir, "history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"Saved history to {history_path}")

    if args.debug_attribute:
        attr_save_dir = os.path.join(args.checkpoint_dir, "positive_samples")
        files = list(
            visualize_specific_attribute(
                model,
                val_dataset,
                device,
                args.debug_attribute,
                SELECTED_ATTRIBUTES,
                attr_save_dir,
                args.image_size,
                num_samples=args.debug_samples,
            )
        )
        print(f"Saved positive attribute visualizations: {files}")

    if args.debug_attribute_negative:
        neg_save_dir = os.path.join(args.checkpoint_dir, "negative_samples")
        files = list(
            visualize_specific_attribute_negative(
                model,
                val_dataset,
                device,
                args.debug_attribute_negative,
                SELECTED_ATTRIBUTES,
                neg_save_dir,
                args.image_size,
                num_samples=args.debug_samples,
            )
        )
        print(f"Saved negative attribute visualizations: {files}")

    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet18 + CBAM on a CelebA subset")
    parser.add_argument("csv", help="Path to the CelebA attribute CSV")
    parser.add_argument("images", help="Root directory containing CelebA images")
    parser.add_argument("--checkpoint-dir", default="outputs/resnet18_cbam", help="Directory to store artifacts")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", choices=["bce", "asl"], default="bce")
    parser.add_argument("--gamma-neg", type=float, default=4.0)
    parser.add_argument("--gamma-pos", type=float, default=1.0)
    parser.add_argument("--asl-clip", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=5, help="Checkpoint interval (0 to disable)")
    parser.add_argument("--visualize-every", type=int, default=5, help="Visualization interval (0 to disable)")
    parser.add_argument("--export-history", action="store_true", help="Persist history.json")
    parser.add_argument("--attr-map-json", type=str, default=None, help="Optional override for attribute display map")
    parser.add_argument("--debug-attribute", type=str, default=None, help="Attribute key for positive samples visualization")
    parser.add_argument("--debug-attribute-negative", type=str, default=None, help="Attribute key for negative samples visualization")
    parser.add_argument("--debug-samples", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = train_resnet18_cbam(args)
    if args.export_history:
        print("Training completed and history exported.")
    else:
        print("Training completed.")


if __name__ == "__main__":
    main()
