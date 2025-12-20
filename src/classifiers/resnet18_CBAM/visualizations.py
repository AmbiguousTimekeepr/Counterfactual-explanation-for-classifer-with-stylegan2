"""Visualization helpers for model explainability."""
from __future__ import annotations

import importlib
import os
from functools import lru_cache
from typing import Callable, Iterable, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

try:
    resize = getattr(importlib.import_module("skimage.transform"), "resize")
    _SKIMAGE_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - optional dependency guard
    resize = None
    _SKIMAGE_IMPORT_ERROR = exc

_INV_NORMALIZE = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)

_PKG = __package__ or "classifiers.resnet18_CBAM"


@lru_cache()
def _get_gradcam_class():
    module = importlib.import_module(f"{_PKG}.gradcam")
    return getattr(module, "GradCAMPlusPlus")


@lru_cache()
def _get_integrated_gradients():
    module = importlib.import_module(f"{_PKG}.integrated_gradients")
    return getattr(module, "integrated_gradients")


def _denormalize(image: torch.Tensor) -> np.ndarray:
    restored = _INV_NORMALIZE(image).permute(1, 2, 0).cpu().numpy()
    return np.clip(restored, 0, 1)


def visualize_predictions(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    epoch: int,
    attribute_display: Sequence[str],
    save_dir: str,
    image_size: int,
    num_attributes_to_plot: int = 3,
) -> str:
    idx = np.random.randint(len(dataset))
    img_tensor, label_tensor, _ = dataset[idx]
    input_tensor = img_tensor.unsqueeze(0).to(device)

    display_image = _denormalize(img_tensor)

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    GradCAMPlusPlus = _get_gradcam_class()
    target_layer = getattr(model, "layer4")
    grad_cam = GradCAMPlusPlus(model, target_layer)
    grad_cam.register_hooks()

    positive_indices = np.where(label_tensor.numpy() == 1)[0].tolist()
    negative_indices = np.where(label_tensor.numpy() == 0)[0].tolist()
    plot_indices = []
    rng = np.random.default_rng()
    if positive_indices:
        plot_indices.append(rng.choice(positive_indices))
    if negative_indices:
        plot_indices.append(rng.choice(negative_indices))
    while len(plot_indices) < num_attributes_to_plot:
        candidate = rng.integers(0, len(attribute_display))
        if candidate not in plot_indices:
            plot_indices.append(candidate)

    if resize is None:
        raise ImportError("scikit-image is required for visualization") from _SKIMAGE_IMPORT_ERROR
    resize_fn = cast(Callable[..., np.ndarray], resize)

    def _to_image_map(data: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            array = data.detach().cpu().numpy()
        else:
            array = data
        if array.ndim > 2:
            array = np.squeeze(array)
        return array

    fig, axes = plt.subplots(len(plot_indices), 5, figsize=(20, 5 * len(plot_indices)))
    fig.suptitle(f"XAI Visualization - Epoch {epoch}", fontsize=16)

    heatmap = None
    for row, attr_idx in enumerate(plot_indices):
        attr_name = attribute_display[attr_idx]
        gt_label = int(label_tensor[attr_idx].item())
        pred_prob = probs[attr_idx]
        pred_label = 1 if pred_prob > 0.5 else 0
        title = f"{attr_name}\nGT: {gt_label} | Pred: {pred_label} ({pred_prob:.2f})"

        integrated_gradients = _get_integrated_gradients()

        cam_pos = grad_cam.generate_cam(input_tensor, attr_idx, target_class=1)
        cam_neg = grad_cam.generate_cam(input_tensor, attr_idx, target_class=0)
        ig_pos = integrated_gradients(model, input_tensor, attr_idx, target_class=1, steps=30, device=device)
        ig_neg = integrated_gradients(model, input_tensor, attr_idx, target_class=0, steps=30, device=device)

        cam_pos_resized = resize_fn(_to_image_map(cam_pos), (image_size, image_size), mode="reflect", anti_aliasing=True)
        cam_neg_resized = resize_fn(_to_image_map(cam_neg), (image_size, image_size), mode="reflect", anti_aliasing=True)
        ig_pos_resized = resize_fn(_to_image_map(ig_pos), (image_size, image_size), mode="reflect", anti_aliasing=True)
        ig_neg_resized = resize_fn(_to_image_map(ig_neg), (image_size, image_size), mode="reflect", anti_aliasing=True)

        ax_row = axes[row] if len(plot_indices) > 1 else axes
        ax_row[0].imshow(display_image)
        ax_row[0].set_title(title, fontsize=10)
        ax_row[0].axis("off")

        ax_row[1].imshow(display_image)
        heatmap = ax_row[1].imshow(cam_pos_resized, cmap="jet", alpha=0.6, vmin=0, vmax=1)
        ax_row[1].set_title("GradCAM++ Positive", fontsize=10)
        ax_row[1].axis("off")

        ax_row[2].imshow(display_image)
        ax_row[2].imshow(cam_neg_resized, cmap="jet", alpha=0.6, vmin=0, vmax=1)
        ax_row[2].set_title("GradCAM++ Negative", fontsize=10)
        ax_row[2].axis("off")

        ax_row[3].imshow(display_image)
        ax_row[3].imshow(ig_pos_resized, cmap="jet", alpha=0.6, vmin=0, vmax=1)
        ax_row[3].set_title("Integrated Gradients Positive", fontsize=10)
        ax_row[3].axis("off")

        ax_row[4].imshow(display_image)
        ax_row[4].imshow(ig_neg_resized, cmap="jet", alpha=0.6, vmin=0, vmax=1)
        ax_row[4].set_title("Integrated Gradients Negative", fontsize=10)
        ax_row[4].axis("off")

    if heatmap is not None:
        fig.colorbar(heatmap, ax=axes.ravel().tolist(), label="Attribution Intensity", shrink=0.6)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"viz_epoch_{epoch}.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    grad_cam.remove_hooks()
    return output_path


def visualize_specific_attribute(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    attr_name: str,
    attribute_names: Sequence[str],
    save_dir: str,
    image_size: int,
    num_samples: int = 5,
) -> Iterable[str]:
    attr_idx = attribute_names.index(attr_name)
    outputs = []
    os.makedirs(save_dir, exist_ok=True)

    if resize is None:
        raise ImportError("scikit-image is required for visualization") from _SKIMAGE_IMPORT_ERROR
    resize_fn = cast(Callable[..., np.ndarray], resize)

    positive_indices = []
    for i in range(len(dataset)):
        _, label, _ = dataset[i]
        if label[attr_idx] == 1:
            positive_indices.append(i)
        if len(positive_indices) >= num_samples:
            break

    GradCAMPlusPlus = _get_gradcam_class()
    target_layer = getattr(model, "cbam4")
    for idx in positive_indices:
        img_tensor, label_tensor, img_name = dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)

        grad_cam = GradCAMPlusPlus(model, target_layer)
        grad_cam.register_hooks()
        cam = grad_cam.generate_cam(input_tensor, attr_idx, target_class=1)
        grad_cam.remove_hooks()

        display_image = _denormalize(img_tensor)
        cam_resized = resize_fn(cam, (image_size, image_size), mode="reflect", anti_aliasing=True)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(display_image)
        axes[0].set_title(f"{img_name}\nGT: {int(label_tensor[attr_idx])}")
        axes[0].axis("off")
        axes[1].imshow(display_image)
        axes[1].imshow(cam_resized, cmap="jet", alpha=0.6, vmin=0, vmax=1)
        axes[1].set_title(f"GradCAM++ for {attr_name}")
        axes[1].axis("off")

        plt.tight_layout()
        output_path = os.path.join(save_dir, f"debug_{attr_name}_{idx}.png")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output_path)

    return outputs


def visualize_specific_attribute_negative(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    attr_name: str,
    attribute_names: Sequence[str],
    save_dir: str,
    image_size: int,
    num_samples: int = 5,
) -> Iterable[str]:
    attr_idx = attribute_names.index(attr_name)
    outputs = []
    os.makedirs(save_dir, exist_ok=True)

    if resize is None:
        raise ImportError("scikit-image is required for visualization") from _SKIMAGE_IMPORT_ERROR
    resize_fn = cast(Callable[..., np.ndarray], resize)

    negative_indices = []
    for i in range(len(dataset)):
        _, label, _ = dataset[i]
        if label[attr_idx] == 0:
            negative_indices.append(i)
        if len(negative_indices) >= num_samples:
            break

    GradCAMPlusPlus = _get_gradcam_class()
    target_layer = getattr(model, "cbam4")
    for idx in negative_indices:
        img_tensor, label_tensor, img_name = dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)

        grad_cam = GradCAMPlusPlus(model, target_layer)
        grad_cam.register_hooks()
        cam = grad_cam.generate_cam(input_tensor, attr_idx, target_class=0)
        grad_cam.remove_hooks()

        model.eval()
        with torch.no_grad():
            logits = model(input_tensor)
            prob = torch.sigmoid(logits[0, attr_idx]).item()

        display_image = _denormalize(img_tensor)
        cam_resized = resize_fn(cam, (image_size, image_size), mode="reflect", anti_aliasing=True)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(display_image)
        axes[0].set_title(f"{img_name}\nGT: {int(label_tensor[attr_idx])} | Pred: {prob:.3f}")
        axes[0].axis("off")
        axes[1].imshow(display_image)
        axes[1].imshow(cam_resized, cmap="jet", alpha=0.6, vmin=0, vmax=1)
        axes[1].set_title(f"GradCAM++ for NOT {attr_name}")
        axes[1].axis("off")

        plt.tight_layout()
        output_path = os.path.join(save_dir, f"debug_{attr_name}_negative_{idx}.png")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output_path)

    return outputs


__all__ = [
    "visualize_predictions",
    "visualize_specific_attribute",
    "visualize_specific_attribute_negative",
]
