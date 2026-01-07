"""
Visualization utilities for VQVAE_HF model
Simplified for residual quantizer architecture
"""
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from datetime import datetime

def viz_traversals(model, dataloader, device, save_dir="traversals", num_samples=4):
    """Visualize latent code traversals"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            imgs = batch[0].to(device)
            z = model.encode(imgs)
            # Traverse all stages
            for stage_idx in range(model.quantizer.num_stages):
                recon = model.decode_from_stage(z, stage_idx)
                # Save grid for each stage
                grid = recon[:8].cpu()
                grid = (grid + 1) / 2  # [-1,1] to [0,1]
                grid = torch.clamp(grid, 0, 1)
                grid_np = grid.permute(0, 2, 3, 1).numpy()
                fig, axs = plt.subplots(1, 8, figsize=(16, 2))
                for i in range(8):
                    axs[i].imshow(grid_np[i])
                    axs[i].axis('off')
                fig.suptitle(f"Stage {stage_idx} Traversal (Batch {batch_idx})")
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"traversal_batch{batch_idx}_stage{stage_idx}.png"))
                plt.close(fig)
            if batch_idx >= 2:
                break  # Only visualize a few batches for speed
    
    print(f"  Generated traversal visualizations at {save_dir}")

def viz_usage(model, dataloader, device, save_dir="usage", num_batches=10):
    """Analyze codebook usage across stages"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    codebook_usages = [ [] for _ in range(model.quantizer.num_stages) ]
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch[0].to(device)
            z = model.encode(imgs)
            _, _, indices_list = model.quantizer(z)
            for stage_idx, indices in enumerate(indices_list):
                codebook_usages[stage_idx].extend(indices.cpu().numpy().tolist())
    
    # Plot usage for all stages
    num_stages = model.quantizer.num_stages
    fig, axs = plt.subplots(1, num_stages, figsize=(5*num_stages, 4))
    if num_stages == 1:
        axs = [axs]
    for stage_idx in range(num_stages):
        axs[stage_idx].hist(codebook_usages[stage_idx], bins=50, color='skyblue')
        axs[stage_idx].set_title(f"Stage {stage_idx} Codebook Usage")
        axs[stage_idx].set_xlabel("Code Index")
        axs[stage_idx].set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "codebook_usage_all_stages.png"))
    plt.close(fig)
    print(f"  Generated codebook usage plots at {save_dir}")

def viz_tsne(model, dataloader, device, save_dir="tsne", num_samples=100):
    """Visualize encoded latents with t-SNE"""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  Warning: scikit-learn not installed, skipping t-SNE visualization")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    latents = []
    labels = []
    collected = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if collected >= num_samples:
                break
            images = batch[0].to(device)
            z = model.encode(images)
            
            # Flatten spatial dimensions
            B, C, H, W = z.shape
            z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, C)
            latents.append(z_flat.cpu().numpy())
            collected += z_flat.shape[0]
    
    if len(latents) > 0:
        data = np.vstack(latents)[:num_samples]
        
        # Apply t-SNE
        perplexity = max(5, min(30, len(data) // 3))
        embedding = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(data)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(embedding[:, 0], embedding[:, 1], s=20, alpha=0.6)
        plt.title("Latent Space t-SNE Visualization")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "latent_tsne.png"), dpi=150)
        plt.close()
        print(f"  Generated t-SNE visualization at {save_dir}")

def viz_neighbors(model, dataloader, device, save_dir="neighbors"):
    """Visualize reconstruction by stage progression"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    batch = next(iter(dataloader))
    images = batch[0][:4].to(device)
    
    with torch.no_grad():
        z = model.encode(images)
        
        # Get reconstructions at different stages
        reconstructions = [images]
        
        for stage_idx in range(0, model.quantizer.num_stages, 2):
            try:
                recon = model.decode_from_stage(z, stage_idx)
                reconstructions.append(recon)
            except Exception as e:
                print(f"    Warning: error at stage {stage_idx}: {e}")
    
    # Create grid
    if len(reconstructions) > 1:
        from torchvision.utils import make_grid
        grid = make_grid(torch.cat(reconstructions, dim=0), nrow=len(images), normalize=True, value_range=(-1, 1))
        save_image(grid, os.path.join(save_dir, "stage_progression.png"))
        print(f"  Generated stage progression visualization at {save_dir}")

def viz_error_maps(model, dataloader, device, save_dir="errors", num_batches=5):
    """Visualize reconstruction error across images"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    errors = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device)
            
            # Handle new 4-return signature
            recon, vq_loss, info, indices_list = model(images)
            
            # Compute per-pixel error
            error = (images - recon).abs().mean(dim=1)  # (B, H, W)
            errors.append(error)
    
    if len(errors) > 0:
        all_errors = torch.cat(errors, dim=0)
        mean_error = all_errors.mean(dim=0).cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Error heatmap
        im = axes[0].imshow(mean_error, cmap='hot')
        axes[0].set_title('Mean Reconstruction Error')
        plt.colorbar(im, ax=axes[0])
        
        # Error distribution
        axes[1].hist(all_errors.cpu().numpy().flatten(), bins=50, alpha=0.7)
        axes[1].set_title('Error Distribution')
        axes[1].set_xlabel('Absolute Error')
        axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "error_analysis.png"), dpi=150)
        plt.close()
        print(f"  Generated error maps at {save_dir}")

class TrainingVisualizer:
    """Visualize training metrics and model outputs"""
    
    @staticmethod
    def plot_loss_curves(loss_tracker, save_dir, step):
        """Plot training loss curves"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Training Losses at Step {step:,}', fontsize=16, fontweight='bold')
        
        # Get history
        if not hasattr(loss_tracker, 'losses') or len(loss_tracker.losses['total']) == 0:
            print("Warning: no loss history available for plotting")
            return None
        
        steps = range(len(loss_tracker.losses['total']))
        
        # Total Loss
        axes[0, 0].plot(steps, loss_tracker.losses['total'], linewidth=2)
        axes[0, 0].set_title('Total Loss', fontweight='bold')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reconstruction Loss
        axes[0, 1].plot(steps, loss_tracker.losses['recon'], color='orange', linewidth=2)
        axes[0, 1].set_title('Reconstruction Loss', fontweight='bold')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # VQ Loss
        axes[1, 0].plot(steps, loss_tracker.losses['vq'], color='green', linewidth=2)
        axes[1, 0].set_title('VQ Loss', fontweight='bold')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Perplexity
        axes[1, 1].plot(steps, loss_tracker.losses['perplexity'], color='red', linewidth=2)
        axes[1, 1].set_title('Codebook Perplexity', fontweight='bold')
        axes[1, 1].set_ylabel('Perplexity')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'loss_curves_step_{step}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    @staticmethod
    def plot_metrics_comparison(metrics_dict, save_dir, step):
        """Plot reconstruction metrics over time"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Reconstruction Metrics at Step {step:,}', fontsize=16, fontweight='bold')
        
        if 'step' not in metrics_dict or len(metrics_dict['step']) == 0:
            print("Warning: no metrics history available for plotting")
            return None
        
        steps = metrics_dict['step']
        
        # MSE
        axes[0, 0].plot(steps, metrics_dict['recon_mse'], linewidth=2, marker='o')
        axes[0, 0].set_title('Reconstruction MSE (Lower is Better)', fontweight='bold')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].grid(True, alpha=0.3)
        
        # L1
        axes[0, 1].plot(steps, metrics_dict['recon_l1'], color='orange', linewidth=2, marker='o')
        axes[0, 1].set_title('Reconstruction L1 (Lower is Better)', fontweight='bold')
        axes[0, 1].set_ylabel('L1 Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # PSNR
        axes[1, 0].plot(steps, metrics_dict['psnr'], color='green', linewidth=2, marker='o')
        axes[1, 0].set_title('PSNR (Higher is Better)', fontweight='bold')
        axes[1, 0].set_ylabel('PSNR (dB)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # SSIM
        axes[1, 1].plot(steps, metrics_dict['ssim'], color='red', linewidth=2, marker='o')
        axes[1, 1].set_title('SSIM (Higher is Better)', fontweight='bold')
        axes[1, 1].set_ylabel('SSIM')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'metrics_comparison_step_{step}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    @staticmethod
    def plot_cascade_hierarchy(original, stage_1, stage_2, stage_3, save_dir, step):
        """Visualize the cascaded dropout learning hierarchy"""
        num_samples = min(4, original.shape[0])
        
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(num_samples, 4, figure=fig, hspace=0.3, wspace=0.2)
        fig.suptitle(f'Cascaded Hierarchy at Step {step:,}: Progressive Reconstruction Quality', 
                     fontsize=16, fontweight='bold')
        
        # Denormalize function
        def denorm(x):
            return torch.clamp((x + 1) / 2, 0, 1)
        
        for i in range(num_samples):
            # Original
            ax = fig.add_subplot(gs[i, 0])
            ax.imshow(denorm(original[i]).cpu().permute(1, 2, 0).numpy())
            ax.set_title('Original' if i == 0 else '', fontweight='bold')
            ax.axis('off')
            if i == 0:
                ax.text(-0.15, 0.5, 'Sample', transform=ax.transAxes, 
                       fontsize=10, fontweight='bold', rotation=90, va='center')
            
            # Stage 1 (Top only)
            ax = fig.add_subplot(gs[i, 1])
            ax.imshow(denorm(stage_1[i]).cpu().permute(1, 2, 0).numpy())
            ax.set_title('Stage 1\n(Top: Structure)', fontweight='bold' if i == 0 else 'normal')
            ax.axis('off')
            
            # Stage 2 (Top + Mid)
            ax = fig.add_subplot(gs[i, 2])
            ax.imshow(denorm(stage_2[i]).cpu().permute(1, 2, 0).numpy())
            ax.set_title('Stage 2\n(Top+Mid: Attributes)', fontweight='bold' if i == 0 else 'normal')
            ax.axis('off')
            
            # Stage 3 (All levels)
            ax = fig.add_subplot(gs[i, 3])
            ax.imshow(denorm(stage_3[i]).cpu().permute(1, 2, 0).numpy())
            ax.set_title('Stage 3\n(All: Texture)', fontweight='bold' if i == 0 else 'normal')
            ax.axis('off')
        
        # Add legend
        fig.text(0.5, 0.02, 
                '25% Top Only → 25% Top+Mid → 50% Full | Enforces: Structure → Attributes → Texture',
                ha='center', fontsize=11, style='italic', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.3))
        
        save_path = os.path.join(save_dir, f'cascade_hierarchy_step_{step}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    @staticmethod
    def plot_codebook_usage(codebook_analyzer, save_dir, step):
        """Visualize codebook usage statistics"""
        metrics = codebook_analyzer.get_diversity_metrics()
        
        if not metrics:
            print("Warning: no codebook metrics available")
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f'Codebook Usage at Step {step:,}', fontsize=14, fontweight='bold')
        
        level_names = ['Top (8×8)', 'Mid (16×16)', 'Bottom (32×32)']
        
        for level, ax in enumerate(axes):
            usage_ratio = metrics.get(f'{["Top", "Mid", "Bottom"][level]}_usage_ratio', 0)
            entropy = metrics.get(f'{["Top", "Mid", "Bottom"][level]}_normalized_entropy', 0)
            
            x = ['Usage Ratio', 'Normalized\nEntropy']
            y = [usage_ratio, entropy]
            colors = ['#2ecc71', '#3498db']
            
            bars = ax.bar(x, y, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
            ax.set_ylim([0, 1])
            ax.set_title(level_names[level], fontweight='bold')
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'codebook_usage_step_{step}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path

def create_evaluation_report(loss_tracker, metrics_dict, codebook_analyzer, 
                            original, stage_1, stage_2, stage_3, 
                            save_dir, step):
    """Create comprehensive evaluation report"""
    
    print("\n" + "="*70)
    print(f"Creating evaluation report at step {step:,}")
    print("="*70)
    
    visualizer = TrainingVisualizer()
    
    # Disabled: plot loss curves
    # loss_plot = visualizer.plot_loss_curves(loss_tracker, save_dir, step)
    # if loss_plot:
    #     print(f"Loss curves saved: {os.path.basename(loss_plot)}")
    
    # Plot metrics
    if metrics_dict:
        metrics_plot = visualizer.plot_metrics_comparison(metrics_dict, save_dir, step)
        if metrics_plot:
            print(f"Metrics comparison saved: {os.path.basename(metrics_plot)}")
    
    # Plot cascade hierarchy
    cascade_plot = visualizer.plot_cascade_hierarchy(original, stage_1, stage_2, stage_3, save_dir, step)
    if cascade_plot:
        print(f"Cascade hierarchy saved: {os.path.basename(cascade_plot)}")
    
    # Disabled: plot codebook usage
    # if codebook_analyzer:
    #     codebook_plot = visualizer.plot_codebook_usage(codebook_analyzer, save_dir, step)
    #     if codebook_plot:
    #         print(f"Codebook usage saved: {os.path.basename(codebook_plot)}")
    #     
    #     # Save codebook analysis
    #     analysis_file = codebook_analyzer.save_analysis(save_dir, step)
    #     print(f"Codebook analysis saved: {os.path.basename(analysis_file)}")
    
    print("="*70 + "\n")