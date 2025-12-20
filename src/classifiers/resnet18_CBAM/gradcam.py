"""Grad-CAM++ utilities specialized for multi-label classifiers."""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch.utils.hooks import RemovableHandle


class GradCAMPlusPlus:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self.handles: List[RemovableHandle] = []

    def _save_activation(self, _: torch.nn.Module, __, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _: torch.nn.Module, __, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def register_hooks(self) -> None:
        self.target_layer._backward_hooks.clear()
        self.target_layer._forward_hooks.clear()
        self.target_layer._is_full_backward_hook = None
        self.handles.append(self.target_layer.register_forward_hook(self._save_activation))
        self.handles.append(self.target_layer.register_full_backward_hook(self._save_gradient))

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.target_layer._is_full_backward_hook = None

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        attribute_idx: int,
        target_class: int = 1,
    ) -> np.ndarray:
        tensor = input_tensor.clone().detach().requires_grad_(True)
        self.model.eval()
        output = self.model(tensor)

        if target_class == 1:
            score = torch.sigmoid(output[0, attribute_idx])
        else:
            score = 1 - torch.sigmoid(output[0, attribute_idx])
        score = torch.log(score + 1e-8)

        self.model.zero_grad()
        score.backward(retain_graph=True)

        grads = self.gradients
        activations = self.activations
        if grads is None or activations is None:
            raise RuntimeError("Hooks did not capture gradients/activations")

        grad_2 = grads.pow(2)
        grad_3 = grad_2 * grads
        eps = 1e-8
        spatial_sum = (activations * grad_3).sum(dim=(2, 3), keepdim=True)
        denom = torch.clamp(2 * grad_2 + spatial_sum, min=eps)
        alphas = grad_2 / denom

        weights = (alphas * torch.relu(grads)).sum(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + eps)
        return cam.squeeze().cpu().numpy()


__all__ = ["GradCAMPlusPlus"]
