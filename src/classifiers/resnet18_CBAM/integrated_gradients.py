"""Integrated Gradients helper for multi-label attribution."""
from __future__ import annotations

import torch


def integrated_gradients(
    model: torch.nn.Module,
    input_image: torch.Tensor,
    attribute_idx: int,
    target_class: int = 1,
    baseline: torch.Tensor | None = None,
    steps: int = 50,
    device: torch.device | None = None,
) -> torch.Tensor:
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    image = input_image.to(device)
    if baseline is None:
        base = torch.zeros_like(image).to(device)
    else:
        base = baseline.to(device)

    diff = image - base
    batch_size = max(1, min(steps + 1, 10))
    grads = []

    for start in range(0, steps + 1, batch_size):
        end = min(start + batch_size, steps + 1)
        interpolated = []
        for k in range(start, end):
            alpha = k / steps
            interpolated.append(base + alpha * diff)
        interpolated_input = torch.cat(interpolated, dim=0)
        interpolated_input.requires_grad_(True)
        interpolated_input.retain_grad()
        outputs = model(interpolated_input)
        score = outputs[:, attribute_idx].sum()
        if target_class == 0:
            score = -score
        model.zero_grad()
        score.backward()
        grad = interpolated_input.grad
        if grad is None:
            raise RuntimeError("Integrated gradients backpropagation produced no gradients")
        grads.append(grad.detach().clone())
        interpolated_input.grad = None

    all_grads = torch.cat(grads, dim=0)
    avg_grad = torch.mean(all_grads, dim=0, keepdim=True)
    ig_attribution = diff * avg_grad
    attribution = torch.sum(torch.abs(ig_attribution), dim=1, keepdim=True)
    attribution = attribution - attribution.min()
    attribution = attribution / (attribution.max() + 1e-8)
    return attribution.detach()


__all__ = ["integrated_gradients"]
