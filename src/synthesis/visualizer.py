import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import cv2


class CounterfactualVisualizer:
    def __init__(self, device='cuda'):
        self.device = device
        self.cmap = cm.get_cmap('jet')
    
    def denormalize(self, img_tensor):
        """Convert from [-1, 1] or [0, 1] to [0, 1] range"""
        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.cpu().numpy()
        
        if img_tensor.min() < 0:  # [-1, 1] range
            img_tensor = (img_tensor + 1) / 2
        
        img_tensor = np.clip(img_tensor, 0, 1)
        return img_tensor
    
    def tensor_to_image(self, tensor):
        """Convert [B, C, H, W] or [C, H, W] tensor to [H, W, C] numpy"""
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        tensor = tensor.cpu().detach()
        if tensor.shape[0] == 3:  # [C, H, W]
            tensor = tensor.permute(1, 2, 0)
        
        img = self.denormalize(tensor.numpy())
        return img
    
    def apply_cam_overlay(self, img, cam_map, alpha=0.4):
        """
        Overlay CAM on image
        
        Args:
            img: [H, W, 3] or [C, H, W] numpy or tensor
            cam_map: [H, W] or [1, H, W] heatmap
            alpha: blending factor
        
        Returns:
            [H, W, 3] overlaid image
        """
        img = self.tensor_to_image(img) if isinstance(img, torch.Tensor) else img
        
        if isinstance(cam_map, torch.Tensor):
            cam_map = cam_map.detach().cpu().numpy()
        
        if cam_map.ndim == 3:
            cam_map = cam_map[0]
        
        # Normalize CAM to [0, 1]
        cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-6)
        
        # Resize CAM to match image size
        if cam_map.shape != img.shape[:2]:
            cam_map = cv2.resize(cam_map, (img.shape[1], img.shape[0]))
        
        # Create heatmap
        heatmap = self.cmap(cam_map)[:, :, :3]  # [H, W, 3]
        
        # Blend
        overlaid = (1 - alpha) * img + alpha * heatmap
        overlaid = np.clip(overlaid, 0, 1)
        
        return overlaid
    
    def create_comparison_grid(self, img_orig, cf_results, labels, attr_names, active_indices=None, active_names=None, image_name=None):
        """
        Create comprehensive comparison grid:
        Row 1: Original | CAM Overlay | (empty space)
        Row 2: CF1 + attr | CF2 + attr | CF3 + attr
        
        Args:
            img_orig: [1, 3, H, W] original image
            cf_results: dict with 'original', 'cfs' (list of dicts with 'image', 'attr_name', 'ig_map', 'cam_map')
            labels: [1, num_attributes] binary labels
            attr_names: list of attribute names
            image_name: optional filename to annotate in the figure
        
        Returns:
            matplotlib figure
        """
        fig = plt.figure(figsize=(17, 8))
        gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 1.3], hspace=0.35, wspace=0.25)
        
        img_orig_np = self.tensor_to_image(img_orig)
        
        # Row 1: Original and CAM
        # ========================
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(img_orig_np)
        ax0.set_title('Original Image', fontsize=12, fontweight='bold')
        ax0.axis('off')
        
        # CAM overlay
        ax1 = fig.add_subplot(gs[0, 1])
        if cf_results['cfs']:
            ig_first = cf_results['cfs'][0]['ig_map']
            overlaid_ig = self.apply_cam_overlay(img_orig, ig_first, alpha=0.5)
            ax1.imshow(overlaid_ig)
        ax1.set_title('IG Overlay (Integrated Gradients)', fontsize=12, fontweight='bold')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 2])
        if cf_results['cfs']:
            cam_first = cf_results['cfs'][0].get('cam_map')
            if cam_first is not None:
                overlaid_cam = self.apply_cam_overlay(img_orig, cam_first, alpha=0.5)
                ax2.imshow(overlaid_cam)
        ax2.set_title('CAM Mask Overlay', fontsize=12, fontweight='bold')
        ax2.axis('off')

        # Attribute info
        ax3 = fig.add_subplot(gs[0, 3])
        ax3.axis('off')
        
        # Determine active attribute subset for display
        if active_indices is None:
            active_indices = list(range(len(attr_names)))
        if active_names is None:
            active_names = [attr_names[idx] for idx in active_indices]

        label_np = labels[0].cpu().numpy().astype(int)

        header_lines = []
        if image_name:
            header_lines.append(f"Image: {image_name}")
        header_lines.append("Active Attributes:")
        attr_text = "\n".join(header_lines) + "\n" + "-" * 30 + "\n"
        display_count = min(10, len(active_indices))
        for idx, name in zip(active_indices[:display_count], active_names[:display_count]):
            val = label_np[idx] if idx < label_np.shape[0] else 0
            attr_text += f"{name}: {val}\n"
        if len(active_indices) > display_count:
            attr_text += "...\n(showing first 10)"
        
        ax3.text(0.1, 0.9, attr_text, transform=ax3.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Row 2: Counterfactuals
        # ========================
        for cf_idx, cf_data in enumerate(cf_results['cfs'][:3]):  # Show first 3
            if cf_idx >= 3:
                break
            ax = fig.add_subplot(gs[1, cf_idx])

            ig_overlay = self.apply_cam_overlay(img_orig, cf_data['ig_map'], alpha=0.5)
            cam_overlay = self.apply_cam_overlay(img_orig, cf_data.get('cam_map', cf_data['ig_map']), alpha=0.5)
            cf_img = self.tensor_to_image(cf_data['image'])
            panel = np.concatenate([ig_overlay, cam_overlay, cf_img], axis=1)
            ax.imshow(panel)

            attr_name = cf_data['attr_name']
            attr_idx = cf_data['attr_idx']
            current_val = label_np[attr_idx]
            new_val = 1 - current_val
            
            title = f"CF{cf_idx+1}: {attr_name}\n"
            title += f"({current_val} → {new_val}) | IG → CAM → Edit"
            
            ax.set_title(title, fontsize=11, fontweight='bold', color='darkblue')
            ax.axis('off')
        
        if len(cf_results['cfs']) < 3:
            for empty_idx in range(len(cf_results['cfs']), 3):
                ax_empty = fig.add_subplot(gs[1, empty_idx])
                ax_empty.axis('off')

        ax_unused = fig.add_subplot(gs[1, 3])
        ax_unused.axis('off')

        # Add overall title
        title_suffix = f" — {image_name}" if image_name else ""
        fig.suptitle(f"Counterfactual Generation Results{title_suffix}", fontsize=14, fontweight='bold', y=0.98)
        
        return fig
    
    def create_detailed_comparison(self, img_orig, cf_results, attr_names, save_path=None):
        """
        Create detailed side-by-side comparison with metrics
        
        Args:
            img_orig: Original image tensor
            cf_results: Counterfactual results
            attr_names: Attribute names
            save_path: Path to save figure
        
        Returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        img_orig_np = self.tensor_to_image(img_orig)
        axes[0, 0].imshow(img_orig_np)
        axes[0, 0].set_title('Original', fontweight='bold')
        axes[0, 0].axis('off')
        
        for idx, (ax, cf_data) in enumerate(zip(axes.flatten()[1:], cf_results['cfs'][:3])):
            cf_img = self.tensor_to_image(cf_data['image'])
            ax.imshow(cf_img)
            ax.set_title(f"CF: {cf_data['attr_name']}", fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_training_curves(self, csv_log_path, save_path=None):
        """
        Plot training curves from CSV log
        
        Args:
            csv_log_path: Path to training_log.csv
            save_path: Path to save plot
        
        Returns:
            matplotlib figure
        """
        import pandas as pd
        
        df = pd.read_csv(csv_log_path)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        loss_fields = ['total_loss', 'cf_loss', 'retention_loss', 'latent_prox_loss', 'ortho_loss', 'sparse_loss']
        
        for ax, field in zip(axes.flatten(), loss_fields):
            if field in df.columns:
                ax.plot(df['step'], df[field], linewidth=1.5)
                ax.set_xlabel('Step')
                ax.set_ylabel(field)
                ax.set_title(field.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
