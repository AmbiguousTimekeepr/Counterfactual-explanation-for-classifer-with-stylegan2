import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

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
from ..dataset import CelebADataset, map_attribute_names
from ..model import ResNet50_CBAM, SquarePadResize
from ..losses import AsymmetricLossOptimized


def build_dataloaders(
    csv_path: str,
    image_dir: str,
    batch_size: int,
    image_size: int,
    num_workers: int,
    val_split: float,
    seed: int,
    attribute_name_map: Optional[Dict[str, str]] = None,
    train_return_image_name: bool = False,
    val_return_image_name: bool = False,
) -> Tuple[
    DataLoader,
    DataLoader,
    torch.Tensor,
    Sequence[str],
    Sequence[str],
    pd.DataFrame,
    pd.DataFrame,
]:
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

    train_dataset = CelebADataset(
        train_df,
        image_dir,
        transform=train_transform,
        attribute_map=attribute_name_map,
        return_image_name=train_return_image_name,
    )
    val_dataset = CelebADataset(
        val_df,
        image_dir,
        transform=val_transform,
        attribute_map=attribute_name_map,
        return_image_name=val_return_image_name,
    )

    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)

    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)

    attribute_display_names = map_attribute_names(SELECTED_ATTRIBUTES, attribute_name_map)

    return (
        train_loader,
        val_loader,
        pos_weights,
        SELECTED_ATTRIBUTES,
        attribute_display_names,
        train_df,
        val_df,
    )


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
    attribute_display_names: Optional[Sequence[str]] = None,
    sample_record_limit: int = 0,
) -> Dict[str, object]:
    """Train ResNet50+CBAM with validation tracking and checkpointing."""

    os.makedirs(checkpoint_dir, exist_ok=True)
    history: Dict[str, List] = {
        "train_loss": [],
        "val_loss": [],
        "val_attr_acc": [],
        "mean_val_acc": [],
        "learning_rate": [],
    }
    if sample_record_limit > 0:
        history["val_sample_predictions"] = []
    best_val_loss = float("inf")
    display_names = attribute_display_names or attribute_names

    def _split_batch(batch):
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                return batch[0], batch[1], batch[2]
            if len(batch) == 2:
                return batch[0], batch[1], None
        raise ValueError("Unexpected batch structure from DataLoader")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")

        for batch in train_iter:
            images, labels, _ = _split_batch(batch)
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

        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        history["learning_rate"].append(current_lr)
        model.eval()
        val_running_loss = 0.0
        all_preds = []
        all_targets = []
        recorded_samples: List[Dict[str, object]] = []
        val_iter = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]")

        with torch.no_grad():
            for batch in val_iter:
                images, labels, names = _split_batch(batch)
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                probs = torch.sigmoid(outputs)
                probs_np = probs.cpu().numpy()
                targets_np = labels.cpu().numpy()
                all_preds.append(probs_np)
                all_targets.append(targets_np)

                if sample_record_limit > 0 and names is not None:
                    for name, target_row, prob_row in zip(names, targets_np, probs_np):
                        if len(recorded_samples) >= sample_record_limit:
                            break
                        pred_row = (prob_row > threshold).astype(int)
                        recorded_samples.append(
                            {
                                "image": str(name),
                                "targets": target_row.astype(int).tolist(),
                                "probabilities": prob_row.tolist(),
                                "predictions": pred_row.tolist(),
                            }
                        )

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        history["val_loss"].append(epoch_val_loss)

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        binary_preds = (all_preds > threshold).astype(int)
        correct_counts = np.sum(binary_preds == all_targets, axis=0)
        acc_per_attr = correct_counts / len(val_loader.dataset)
        history["val_attr_acc"].append(
            {
                display_names[idx]: float(acc_per_attr[idx])
                for idx in range(len(display_names))
            }
        )
        mean_acc = float(np.mean(acc_per_attr))
        history["mean_val_acc"].append(mean_acc)
        if sample_record_limit > 0:
            history["val_sample_predictions"].append(recorded_samples)

        header = "-" * 60
        print(f"\n--- Epoch {epoch} Report ---")
        print(
            f"LR: {current_lr:.6f} | Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | Mean Val Acc: {mean_acc:.4f}"
        )
        print(header)
        print(f"{'Attribute':<25} | {'Accuracy':<10} | {'Sample Prob (Mean)':<15}")
        print(header)
        mean_probs = np.mean(all_preds, axis=0)
        for idx, attr_name in enumerate(display_names):
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

    history["attribute_names"] = list(attribute_names)
    history["attribute_display_names"] = list(display_names)
    history["threshold"] = threshold
    print("Training Complete.")
    return history


def save_history(history: Dict[str, object], output_dir: str) -> None:
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
    parser.add_argument("--loss", default="bce", choices=["bce", "asl"], help="Loss function to use")
    parser.add_argument("--gamma-pos", type=float, default=0.0, help="Positive focusing parameter for ASL")
    parser.add_argument("--gamma-neg", type=float, default=4.0, help="Negative focusing parameter for ASL")
    parser.add_argument("--asl-clip", type=float, default=0.05, help="Probability clipping for ASL")
    parser.add_argument(
        "--attr-map",
        type=str,
        default=None,
        help="Optional path to JSON mapping attribute names to display labels",
    )
    parser.add_argument(
        "--return-names",
        action="store_true",
        help="Return image names from dataloaders for inspection",
    )
    parser.add_argument(
        "--sample-record-limit",
        type=int,
        default=0,
        help="Number of validation samples to record per epoch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    attribute_name_map = None
    if args.attr_map:
        with open(args.attr_map, "r", encoding="utf-8") as f:
            attribute_name_map = json.load(f)

    train_loader, val_loader, pos_weights, attribute_names, attribute_display_names, train_df, val_df = build_dataloaders(
        csv_path=args.csv,
        image_dir=args.images,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
        attribute_name_map=attribute_name_map,
        train_return_image_name=args.return_names,
        val_return_image_name=args.return_names,
    )

    model = create_model(num_classes=len(attribute_names), device=device)
    if args.loss == "asl":
        criterion = AsymmetricLossOptimized(
            gamma_pos=args.gamma_pos,
            gamma_neg=args.gamma_neg,
            clip=args.asl_clip,
        )
    else:
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
        attribute_display_names=attribute_display_names,
        checkpoint_dir=args.output,
        checkpoint_prefix="resnet50_cbam",
        save_every=5,
        threshold=0.5,
        sample_record_limit=args.sample_record_limit,
    )

    save_history(history, args.output)
    train_df.to_csv(os.path.join(args.output, "train_split.csv"))
    val_df.to_csv(os.path.join(args.output, "val_split.csv"))


if __name__ == "__main__":
    main()

