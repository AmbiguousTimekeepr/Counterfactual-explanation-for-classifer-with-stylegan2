"""
Grad-CAM and Grad-CAM++ Implementation
"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


class GradCAM:
    def __init__(self, model, target_layer):
        """
        Args:
            model: The CNN model
            target_layer: The target layer to extract gradients from
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def register_hooks(self):
        """Register forward and backward hooks"""
        # Clear ALL existing hooks and reset the flag
        self.target_layer._backward_hooks.clear()
        self.target_layer._forward_hooks.clear()
        self.target_layer._is_full_backward_hook = None
        
        self.handles.append(
            self.target_layer.register_forward_hook(self.save_activation)
        )
        self.handles.append(
            self.target_layer.register_full_backward_hook(self.save_gradient)
        )
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.target_layer._is_full_backward_hook = None
    
    def generate_cam(self, input_tensor, attribute_idx, target_class=1):
        """
        Generate CAM for specific attribute
        
        Args:
            input_tensor: Input image tensor
            attribute_idx: Index of attribute to visualize
            target_class: 0 for negative prediction, 1 for positive prediction
        """
        # Clone and enable gradient computation
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Get score for specific attribute
        score = output[0, attribute_idx]
        
        # For negative class (class 0), we want to maximize the negative score
        if target_class == 0:
            score = -score
        
        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # Calculate weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # Apply ReLU (only positive influences)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().detach().numpy()


class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        """
        Args:
            model: The CNN model
            target_layer: The target layer to extract gradients from
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def register_hooks(self):
        """Register forward and backward hooks"""
        # Clear ALL existing hooks and reset the flag
        self.target_layer._backward_hooks.clear()
        self.target_layer._forward_hooks.clear()
        self.target_layer._is_full_backward_hook = None
        
        self.handles.append(
            self.target_layer.register_forward_hook(self.save_activation)
        )
        self.handles.append(
            self.target_layer.register_full_backward_hook(self.save_gradient)
        )
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.target_layer._is_full_backward_hook = None
    
    def generate_cam(self, input_tensor, attribute_idx, target_class=1):
        """
        Generate CAM++ for specific attribute
        
        Args:
            input_tensor: Input image tensor
            attribute_idx: Index of attribute to visualize
            target_class: 0 for negative prediction, 1 for positive prediction
        """
        # Clone and enable gradient computation
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Get score for specific attribute
        score = output[0, attribute_idx]
        
        # For negative class (class 0), invert the score
        if target_class == 0:
            score = -score
        
        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        grads = self.gradients  # Shape: (B, C, H, W)
        activations = self.activations  # Shape: (B, C, H, W)
        
        # GradCAM++ formula
        grad_2 = grads.pow(2)
        grad_3 = grad_2 * grads
        
        # Calculate alpha with numerical stability
        eps = 1e-8
        spatial_sum = (activations * grad_3).sum(dim=(2, 3), keepdim=True)
        denom = 2 * grad_2 + spatial_sum
        denom = torch.clamp(denom, min=eps)
        
        alphas = grad_2 / denom
        
        # Calculate weights
        weights = (alphas * torch.relu(grads)).sum(dim=(2, 3), keepdim=True)
        
        # Calculate CAM
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + eps)
        
        return cam.squeeze().cpu().detach().numpy()


def visualize_gradcam(model, image_path, attribute_idx, attribute_name, 
                      transform, viz_transform, device, target_layer,
                      target_class=1, alpha=0.4, image_size=224, method='gradcam'):
    """
    Visualization function for GradCAM and GradCAM++
    
    Args:
        model: The trained model
        image_path: Path to the image
        attribute_idx: Index of the attribute to visualize
        attribute_name: Name of the attribute
        transform: Transform for model input (with normalization)
        viz_transform: Transform for visualization (without normalization)
        device: Device to run on
        target_layer: Layer to extract CAM from
        target_class: 0 for negative, 1 for positive
        alpha: Overlay alpha
        image_size: Size of the image
        method: 'gradcam' or 'gradcam++'
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get original image with padding
    original_padded = viz_transform(image)
    original_padded = original_padded.permute(1, 2, 0).numpy()
    original_padded = (original_padded * 255).astype(np.uint8)
    
    # Initialize CAM
    if method.lower() == 'gradcam++':
        cam_extractor = GradCAMPlusPlus(model, target_layer)
    else:
        cam_extractor = GradCAM(model, target_layer)
    
    cam_extractor.register_hooks()
    
    try:
        # Generate CAM
        cam = cam_extractor.generate_cam(input_tensor, attribute_idx, target_class)
        
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (image_size, image_size))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
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
        
        method_name = "Grad-CAM++" if method.lower() == 'gradcam++' else "Grad-CAM"
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title(f"{method_name}\n(Target: {'Negative (0)' if target_class == 0 else 'Positive (1)'})")
        axes[1].axis('off')
        
        axes[2].imshow(overlayed)
        class_name = "Negative (0)" if target_class == 0 else "Positive (1)"
        axes[2].set_title(f"{attribute_name}\nPred: {pred} (prob={prob:.3f})\nTarget Class: {class_name}")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    finally:
        cam_extractor.remove_hooks()
