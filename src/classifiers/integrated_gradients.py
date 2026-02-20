"""
Integrated Gradients Implementation
"""
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def integrated_gradients_batch(
    model,
    input_batch: torch.Tensor,
    attribute_idx: int,
    target_classes: torch.Tensor,
    steps: int = 50,
    device: torch.device = None,
    baseline: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute Integrated Gradients for a batch of images.
    
    Args:
        model: Classifier model
        input_batch: Input images [B, C, H, W]
        attribute_idx: Index of the attribute to explain
        target_classes: Target classes per sample [B]
        steps: Number of interpolation steps
        device: Computation device
        baseline: Baseline tensor [B, C, H, W] or None (defaults to zeros)
    
    Returns:
        Attributions tensor [B, C, H, W]
    """
    if device is None:
        device = input_batch.device
    
    model.eval()
    batch_size, C, H, W = input_batch.shape
    
    if baseline is None:
        baseline = torch.zeros_like(input_batch)
    
    # Create interpolation alphas [steps]
    alphas = torch.linspace(0, 1, steps, device=device)
    
    # Expand dimensions for broadcasting
    input_expanded = input_batch.unsqueeze(1)
    baseline_expanded = baseline.unsqueeze(1)
    alphas_expanded = alphas.view(1, steps, 1, 1, 1)
    
    # Interpolated inputs: [B, steps, C, H, W]
    interpolated = baseline_expanded + alphas_expanded * (input_expanded - baseline_expanded)
    
    # Reshape for batch processing: [B * steps, C, H, W]
    interpolated_flat = interpolated.view(batch_size * steps, C, H, W)
    # ✅ Set requires_grad=True BEFORE forward pass
    interpolated_flat = interpolated_flat.detach().requires_grad_(True)
    
    # Forward pass
    outputs = model(interpolated_flat)  # [B * steps, num_attrs]
    
    # Extract logits for target attribute: [B * steps]
    logits = outputs[:, attribute_idx]

    # Make attribution direction-aware when target_classes is provided.
    # If target_class == 1: increase the logit; if target_class == 0: decrease the logit.
    if target_classes is not None:
        # target_classes is [B]; expand to [B*steps]
        target_flat = target_classes.to(device).view(-1, 1).repeat(1, steps).view(-1)
        sign = torch.where(target_flat > 0, torch.ones_like(logits), -torch.ones_like(logits))
        logits = logits * sign

    # Compute gradients
    grad_outputs = torch.ones_like(logits)
    grads = torch.autograd.grad(
        outputs=logits,
        inputs=interpolated_flat,
        grad_outputs=grad_outputs,
        create_graph=False,
        retain_graph=False
    )[0]  # [B * steps, C, H, W]
    
    # Reshape gradients: [B, steps, C, H, W]
    grads_reshaped = grads.view(batch_size, steps, C, H, W)
    
    # Riemann sum approximation (trapezoidal rule)
    # Average gradients across steps: [B, C, H, W]
    avg_grads = grads_reshaped.mean(dim=1)
    
    # Scale by input difference
    attributions = (input_batch - baseline) * avg_grads
    
    return attributions

def visualize_integrated_gradients(model, image_path, attribute_idx, attribute_name,
                                   transform, viz_transform, device, 
                                   target_class=1, steps=50, alpha=0.4, image_size=224):
    """
    Visualization function for Integrated Gradients
    
    Args:
        model: The trained model
        image_path: Path to the image
        attribute_idx: Index of the attribute to visualize
        attribute_name: Name of the attribute
        transform: Transform for model input (with normalization)
        viz_transform: Transform for visualization (without normalization)
        device: Device to run on
        target_class: 0 for negative, 1 for positive
        steps: Number of integration steps
        alpha: Overlay alpha
        image_size: Size of the image
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get original image with padding
    original_padded = viz_transform(image)
    original_padded = original_padded.permute(1, 2, 0).numpy()
    original_padded = (original_padded * 255).astype(np.uint8)
    
    # Compute Integrated Gradients using batch function
    ig_attr = integrated_gradients_batch(
        model=model,
        input_batch=input_tensor,
        attribute_idx=attribute_idx,
        target_classes=torch.tensor([int(target_class)], device=device),
        steps=steps,
        device=device
    )
    
    # Convert attribution to visualization
    # Take absolute value and sum across channels
    ig_attr_abs = torch.abs(ig_attr).sum(dim=1).squeeze().cpu().numpy()
    
    # Normalize to [0, 1]
    ig_attr_norm = ig_attr_abs - ig_attr_abs.min()
    ig_attr_norm = ig_attr_norm / (ig_attr_norm.max() + 1e-8)
    
    # Resize to match image size
    ig_resized = cv2.resize(ig_attr_norm, (image_size, image_size))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * ig_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap
    overlayed = heatmap * alpha + original_padded * (1 - alpha)
    overlayed = overlayed.astype(np.uint8)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output)[0, attribute_idx].item()
        pred = 1 if prob > 0.5 else 0
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_padded)
    axes[0].set_title("Original Image (with padding)")
    axes[0].axis('off')
    
    axes[1].imshow(ig_resized, cmap='jet')
    axes[1].set_title(f"Integrated Gradients\n(Target: {'Negative (0)' if target_class == 0 else 'Positive (1)'})")
    axes[1].axis('off')
    
    axes[2].imshow(overlayed)
    class_name = "Negative (0)" if target_class == 0 else "Positive (1)"
    axes[2].set_title(f"{attribute_name}\nPred: {pred} (prob={prob:.3f})\nTarget Class: {class_name}")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()