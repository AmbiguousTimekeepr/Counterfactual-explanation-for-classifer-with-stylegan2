import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGPerceptualLoss(nn.Module):
    """
    Computes Perceptual Loss using a frozen VGG16 network.
    Extracts features from multiple layers to capture both 
    low-level texture and high-level semantics.
    """
    def __init__(self, layer='relu5_1', device='cuda'):
        super().__init__()
        self.layer = layer
        self.device = device
        
        # Lazy import to avoid circular dependency
        from torchvision import models
        
        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True).to(device)
        
        # Define feature layers
        self.features = nn.Sequential(*list(vgg.features.children())[:30]).to(device)
        self.features.eval()
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Normalization constants (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))
    
    def forward(self, x, target):
        """
        Args:
            x: predicted image [B, 3, H, W]
            target: target image [B, 3, H, W]
        
        Returns:
            perceptual loss (scalar)
        """
        # Normalize inputs
        x_norm = (x - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        
        # Extract features
        x_feat = self.features(x_norm)
        target_feat = self.features(target_norm)
        
        # Compute loss
        loss = F.mse_loss(x_feat, target_feat.detach())
        return loss


def calculate_loss(recon, target, vq_loss, weights, perceptual_fn=None):
    """
    L = L_recon (MSE) + L_perceptual (VGG) + L_vq
    """
    # 1. MSE Reconstruction (Structure)
    recon_loss = F.mse_loss(recon, target)
    
    # 2. Perceptual Loss (Texture)
    p_loss = torch.tensor(0.0, device=recon.device)
    if perceptual_fn is not None and weights['perceptual'] > 0:
        p_loss = perceptual_fn(recon, target)
    
    # Total
    total_loss = (weights['recon'] * recon_loss) + \
                 (weights['vq'] * vq_loss) + \
                 (weights['perceptual'] * p_loss)
    
    return total_loss, recon_loss, vq_loss, p_loss