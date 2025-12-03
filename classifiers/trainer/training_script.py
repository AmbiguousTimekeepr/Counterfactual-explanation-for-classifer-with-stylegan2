import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from src.classifiers.ex_classifier import CelebAClassifierDataset, ExplainableClassifier
from src.classifiers.cam_utils import get_gradcam_mask
from src.classifiers.ig_utils import get_ig_safe

def auto_detect_celeba_paths(dataset_dir):
    # Detect train/val/test image dirs and their respective attribute files
    train_img = val_img = test_img = None
    train_attr = val_attr = test_attr = None

    for root, dirs, files in os.walk(dataset_dir):
        for d in dirs:
            if d == "img_align_celeba":
                if "train" in root:
                    train_img = os.path.join(root, d)
                elif "val" in root:
                    val_img = os.path.join(root, d)
                elif "test" in root:
                    test_img = os.path.join(root, d)
        for f in files:
            if f.startswith("list_attr_celeba") and (f.endswith(".txt") or f.endswith(".csv")):
                if "train" in root:
                    train_attr = os.path.join(root, f)
                elif "val" in root:
                    val_attr = os.path.join(root, f)
                elif "test" in root:
                    test_attr = os.path.join(root, f)
    # Fallback: attribute files may be in dataset_dir root
    if train_attr is None or val_attr is None or test_attr is None:
        for f in os.listdir(dataset_dir):
            if f.startswith("list_attr_celeba") and (f.endswith(".txt") or f.endswith(".csv")):
                if "train" in f and train_attr is None:
                    train_attr = os.path.join(dataset_dir, f)
                elif "val" in f and val_attr is None:
                    val_attr = os.path.join(dataset_dir, f)
                elif "test" in f and test_attr is None:
                    test_attr = os.path.join(dataset_dir, f)
    return train_img, val_img, test_img, train_attr, val_attr, test_attr

def calculate_metrics(y_true, y_pred, attribute_names):
    # Multi-label metrics (standalone, no sklearn)
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    exact_match = (y_true == y_pred).all(axis=1).mean()
    hamming = (y_true != y_pred).mean()
    per_attr_acc = (y_true == y_pred).mean(axis=0)
    metrics = {
        'exact_match_ratio': float(exact_match),
        'hamming_loss': float(hamming),
        'per_attribute_accuracy': {attr: float(per_attr_acc[i]) for i, attr in enumerate(attribute_names)}
    }
    return metrics

def plot_training_curves(history, save_dir):
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    val_acc = [h['val_acc'] for h in history]

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, val_acc, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.close()

def plot_per_attribute_heatmap(per_attr_acc, attribute_names, save_dir):
    plt.figure(figsize=(12,6))
    sns.heatmap([list(per_attr_acc.values())], annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=attribute_names)
    plt.title("Per-Attribute Accuracy (Final Epoch)")
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "per_attribute_accuracy_heatmap.png"))
    plt.close()

def generate_markdown_report(history, metrics, save_dir, model_name):
    report = f"""# Training Report: {model_name}

**Best Validation Accuracy:** {max([h['val_acc'] for h in history]):.4f}

## Training Curves

![Training Curves](training_curves.png)

## Per-Attribute Accuracy (Final Epoch)

![Per-Attribute Accuracy](per_attribute_accuracy_heatmap.png)

## Final Metrics

- Exact Match Ratio: {metrics['exact_match_ratio']:.4f}
- Hamming Loss: {metrics['hamming_loss']:.4f}

### Per-Attribute Accuracy

| Attribute | Accuracy |
|-----------|----------|
"""
    for attr, acc in metrics['per_attribute_accuracy'].items():
        report += f"| {attr} | {acc:.4f} |\n"

    with open(os.path.join(save_dir, "README.md"), "w") as f:
        f.write(report)

def train_and_report_all_models(args, attribute_names):
    # If dataset_dir is provided, auto-detect paths
    if hasattr(args, "dataset_dir") and args.dataset_dir:
        train_img, val_img, test_img, train_attr, val_attr, test_attr = auto_detect_celeba_paths(args.dataset_dir)
        args.train_dir = train_img
        args.val_dir = val_img
        args.train_attr_file = train_attr
        args.val_attr_file = val_attr
        print(f"Auto-detected paths:\n  train_dir: {args.train_dir}\n  val_dir: {args.val_dir}\n  train_attr_file: {args.train_attr_file}\n  val_attr_file: {args.val_attr_file}")

    model_variants = ['mobilenet_v2', 'mobilenet_v3_small', 'resnet18', 'resnet34']
    all_histories = {}
    all_metrics = {}

    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_set = CelebAClassifierDataset(args.train_dir, args.train_attr_file, transform, attribute_names)
    val_set = CelebAClassifierDataset(args.val_dir, args.val_attr_file, transform, attribute_names)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    for model_name in model_variants:
        print(f"\n=== Training {model_name} ===")
        model_save_dir = os.path.join(args.save_dir, model_name)
        os.makedirs(model_save_dir, exist_ok=True)
        model = ExplainableClassifier(model_name, len(attribute_names), attribute_names).to(args.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        best_acc = 0.0
        history = []

        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            for img, label in train_loader:
                img, label = img.to(args.device), label.to(args.device)
                optimizer.zero_grad()
                logits = model(img)
                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            val_preds, val_trues = [], []
            val_loss = 0.0
            with torch.no_grad():
                for img, label in val_loader:
                    img, label = img.to(args.device), label.to(args.device)
                    logits = model(img)
                    loss = criterion(logits, label)
                    val_loss += loss.item()
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    val_preds.append(preds.cpu())
                    val_trues.append(label.cpu())
            val_preds = torch.cat(val_preds)
            val_trues = torch.cat(val_trues)
            acc = (val_preds == val_trues).float().mean().item()
            val_loss /= len(val_loader)

            metrics = calculate_metrics(val_trues, val_preds, attribute_names)
            history.append({
                'epoch': epoch+1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': acc,
                'exact_match_ratio': metrics['exact_match_ratio'],
                'hamming_loss': metrics['hamming_loss'],
                'per_attribute_accuracy': metrics['per_attribute_accuracy']
            })

            # Only print concise summary per epoch (no duplication)
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(model_save_dir, "best_model.pth"))

            scheduler.step(val_loss)

        # Save history and metrics
        pd.DataFrame(history).to_csv(os.path.join(model_save_dir, "training_history.csv"), index=False)
        with open(os.path.join(model_save_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        all_histories[model_name] = history
        all_metrics[model_name] = history[-1]

        # Visualizations for each model
        plot_training_curves(history, model_save_dir)
        plot_per_attribute_heatmap(history[-1]['per_attribute_accuracy'], attribute_names, model_save_dir)
        generate_markdown_report(history, metrics, model_save_dir, model_name)

    # --- Comparison Plots ---
    plt.figure(figsize=(8,5))
    for model_name in model_variants:
        val_acc = [h['val_acc'] for h in all_histories[model_name]]
        plt.plot(range(1, len(val_acc)+1), val_acc, label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "val_accuracy_comparison.png"))
    plt.close()

    acc_matrix = []
    for model_name in model_variants:
        accs = list(all_metrics[model_name]['per_attribute_accuracy'].values())
        acc_matrix.append(accs)
    cell_width = 6
    cell_height = 6
    fig_width = cell_width * len(attribute_names)
    fig_height = cell_height * len(model_variants)
    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(acc_matrix, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=attribute_names, yticklabels=model_variants)
    plt.title("Per-Attribute Accuracy (Final Epoch) Across Models")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "per_attribute_accuracy_comparison.png"))
    plt.close()

    # --- Markdown Report ---
    report = "# Model Comparison Report\n\n"
    report += "## Validation Accuracy Comparison\n\n"
    report += "![Validation Accuracy Comparison](val_accuracy_comparison.png)\n\n"
    report += "## Per-Attribute Accuracy Comparison\n\n"
    report += "![Per-Attribute Accuracy Comparison](per_attribute_accuracy_comparison.png)\n\n"
    report += "## Final Metrics Table\n\n"
    report += "| Model | Exact Match | Hamming Loss | Best Val Acc |\n|-------|-------------|--------------|--------------|\n"
    for model_name in model_variants:
        m = all_metrics[model_name]
        report += f"| {model_name} | {m['exact_match_ratio']:.4f} | {m['hamming_loss']:.4f} | {m['val_acc']:.4f} |\n"
    with open(os.path.join(args.save_dir, "README.md"), "w") as f:
        f.write(report)
    print(f"\nAll model results, plots, and markdown report saved to {args.save_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default=None, help='Root CelebA dataset directory')
    parser.add_argument('--train_dir', type=str, default=None)
    parser.add_argument('--val_dir', type=str, default=None)
    parser.add_argument('--attr_file', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='trained_classifiers')
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attribute_names = [
        "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
        "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Brown_Hair",
        "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
        "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
        "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
        "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
        "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
        "Wearing_Necklace", "Wearing_Necktie", "Young"
    ]
    train_and_report_all_models(args, attribute_names)

if __name__ == "__main__":
    main()

