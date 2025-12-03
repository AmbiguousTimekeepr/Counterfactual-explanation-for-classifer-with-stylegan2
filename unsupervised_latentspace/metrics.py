# hierarchical_vqvae/utils/metrics.py
import torch
import torch.nn.functional as F
import numpy as np
from scipy import linalg
import os
from datetime import datetime
import pandas as pd

class MetricsCalculator:
    """Compute reconstruction and quality metrics for HVQ-GAN"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.history = {
            'step': [],
            'recon_mse': [],
            'recon_l1': [],
            'psnr': [],
            'ssim': [],
        }
    
    def psnr(self, img1, img2):
        """Peak Signal-to-Noise Ratio (higher is better)"""
        # Assuming images in [-1, 1]
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return torch.tensor(100.0)
        return 20 * torch.log10(torch.tensor(2.0) / torch.sqrt(mse))
    
    def ssim(self, img1, img2, window_size=11):
        """Structural Similarity Index (higher is better)"""
        # Convert to [0, 1]
        img1_norm = (img1 + 1) / 2
        img2_norm = (img2 + 1) / 2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = F.avg_pool2d(img1_norm, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(img2_norm, window_size, stride=1, padding=window_size//2)
        
        sigma1_sq = F.avg_pool2d(img1_norm ** 2, window_size, stride=1, padding=window_size//2) - mu1 ** 2
        sigma2_sq = F.avg_pool2d(img2_norm ** 2, window_size, stride=1, padding=window_size//2) - mu2 ** 2
        sigma12 = F.avg_pool2d(img1_norm * img2_norm, window_size, stride=1, padding=window_size//2) - mu1 * mu2
        
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()
    
    def compute_metrics(self, original, reconstructed, step):
        """Compute all metrics and store in history"""
        with torch.no_grad():
            mse = F.mse_loss(original, reconstructed).item()
            l1 = F.l1_loss(original, reconstructed).item()
            psnr_val = self.psnr(original, reconstructed).item()
            ssim_val = self.ssim(original, reconstructed).item()
        
        self.history['step'].append(step)
        self.history['recon_mse'].append(mse)
        self.history['recon_l1'].append(l1)
        self.history['psnr'].append(psnr_val)
        self.history['ssim'].append(ssim_val)
        
        return {
            'mse': mse,
            'l1': l1,
            'psnr': psnr_val,
            'ssim': ssim_val
        }
    
    def save_metrics(self, save_dir):
        """Save metrics history to CSV"""
        
        df = pd.DataFrame(self.history)
        metrics_path = os.path.join(save_dir, 'metrics_history.csv')
        df.to_csv(metrics_path, index=False)
        
        return metrics_path


class CodebookAnalyzer:
    """Analyze VQ codebook usage and diversity"""
    
    def __init__(self, num_embeddings, num_levels=3):
        self.num_embeddings = num_embeddings
        self.num_levels = num_levels
        self.usage_counts = [torch.zeros(num_embeddings) for _ in range(num_levels)]
        self.total_codes = [0] * num_levels
    
    def update(self, indices_list):
        """Update codebook usage statistics
        
        Args:
            indices_list: tuple of (ids_t, ids_m, ids_b) from model forward
        """
        for level, indices in enumerate(indices_list):
            if indices is not None:
                # ✅ FIX: Move indices to CPU before processing
                indices_cpu = indices.cpu()
                # Flatten indices and count usage
                flat_indices = indices_cpu.view(-1)
                for idx in flat_indices:
                    self.usage_counts[level][int(idx)] += 1
                self.total_codes[level] += flat_indices.shape[0]
    
    def get_diversity_metrics(self):
        """Calculate codebook diversity metrics"""
        metrics = {}
        
        for level in range(self.num_levels):
            if self.total_codes[level] == 0:
                continue
            
            # Usage ratio
            used_codes = (self.usage_counts[level] > 0).sum().item()
            usage_ratio = used_codes / self.num_embeddings
            
            # Entropy (measure of diversity)
            probs = self.usage_counts[level] / self.total_codes[level]
            probs = probs[probs > 0]  # Only non-zero probabilities
            entropy = -torch.sum(probs * torch.log(probs)).item()
            max_entropy = np.log(self.num_embeddings)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            level_name = ['Top', 'Mid', 'Bottom'][level]
            metrics[f'{level_name}_used_codes'] = used_codes
            metrics[f'{level_name}_usage_ratio'] = usage_ratio
            metrics[f'{level_name}_entropy'] = entropy
            metrics[f'{level_name}_normalized_entropy'] = normalized_entropy
        
        return metrics
    
    def save_analysis(self, save_dir, step):
        """Save codebook analysis to file"""
        analysis_path = os.path.join(save_dir, f'codebook_analysis_step_{step}.txt')
        
        with open(analysis_path, 'w') as f:
            f.write(f"Codebook Analysis at Step {step}\n")
            f.write("=" * 70 + "\n\n")
            
            metrics = self.get_diversity_metrics()
            for key, value in metrics.items():
                f.write(f"{key:30s}: {value:.4f}\n")
        
        return analysis_path