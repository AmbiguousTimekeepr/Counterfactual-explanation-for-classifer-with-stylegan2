"""
Integrated Gradients Implementation
"""
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def integrated_gradients(model, input_image, attribute_idx, target_class=1, 
                        baseline=None, steps=50, device=None):
    """
    Tính Integrated Gradients cho một hình ảnh đầu vào.
    
    Args:
        model (torch.nn.Module): Mô hình CNN đã huấn luyện.
        input_image (torch.Tensor): Ảnh đầu vào (1, C, H, W).
        attribute_idx (int): Index của attribute cần giải thích.
        target_class (int): 0 cho negative prediction, 1 cho positive prediction.
        baseline (torch.Tensor): Ảnh nền (x'), mặc định là ảnh đen (zeros).
        steps (int): Số bước tích phân (m), thường từ 20-300.
        device: Device để tính toán.
    
    Returns:
        ig_attributions: Attribution map (1, C, H, W)
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    input_image = input_image.to(device)
    
    # 1. Thiết lập Baseline (x') là ảnh đen nếu không cung cấp
    if baseline is None:
        baseline = torch.zeros_like(input_image).to(device)
    else:
        baseline = baseline.to(device)

    # Tính hiệu số (x - x')
    diff = input_image - baseline

    # 2. Tính Gradient cho từng ảnh nội suy (batched để tăng tốc)
    batch_size = 10  # Chia nhỏ để tránh OOM
    all_grads = []
    
    for start in range(0, steps + 1, batch_size):
        end = min(start + batch_size, steps + 1)
        
        # Tạo các ảnh nội suy cho batch này
        interpolated_images = []
        for k in range(start, end):
            alpha = k / steps
            interpolated_image = baseline + alpha * diff
            interpolated_images.append(interpolated_image)
        
        # Gộp thành batch
        interpolated_inputs = torch.cat(interpolated_images, dim=0)
        interpolated_inputs = interpolated_inputs.to(device)
        interpolated_inputs.requires_grad_(True)
        
        # Forward pass
        outputs = model(interpolated_inputs)
        
        # Get score for specific attribute
        score = outputs[:, attribute_idx].sum()
        
        # For negative class (class 0), invert the score
        if target_class == 0:
            score = -score
        
        # Backward
        model.zero_grad()
        grads = torch.autograd.grad(
            outputs=score,
            inputs=interpolated_inputs,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]

        all_grads.append(grads.detach())
    
    # 3. Gộp tất cả gradients
    all_grads = torch.cat(all_grads, dim=0)  # (steps+1, C, H, W)
    
    # 4. Tính tích phân xấp xỉ (trung bình các gradient)
    avg_grads = torch.mean(all_grads, dim=0, keepdim=True)
    
    # 5. Nhân với hiệu số đầu vào (x - x')
    ig_attributions = diff * avg_grads
    
    return ig_attributions.detach()


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
    
    # Compute Integrated Gradients
    ig_attr = integrated_gradients(
        model=model,
        input_image=input_tensor,
        attribute_idx=attribute_idx,
        target_class=target_class,
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
