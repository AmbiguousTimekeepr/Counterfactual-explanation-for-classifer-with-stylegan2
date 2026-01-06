import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import os
from tqdm import tqdm
from .config import cfg
from .model import HVQVAE_3Level
from .discriminator import NLayerDiscriminator
from .losses import calculate_loss, VGGPerceptualLoss
from .dataset import get_loader
from torchvision.utils import save_image, make_grid
import math
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from .metrics import MetricsCalculator, CodebookAnalyzer
from .visualization import create_evaluation_report

class LossTracker:
    """Track and smooth losses for live display"""
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.losses = {
            'total': deque(maxlen=window_size),
            'recon': deque(maxlen=window_size),
            'vq': deque(maxlen=window_size),
            'perc': deque(maxlen=window_size),
            'disc_g': deque(maxlen=window_size),
            'disc_d': deque(maxlen=window_size),
            'perplexity': deque(maxlen=window_size),
        }
    
    def add(self, total, recon, vq, perc, disc_g=0, disc_d=0, perplexity=0.0):
        self.losses['total'].append(total)
        self.losses['recon'].append(recon)
        self.losses['vq'].append(vq)
        self.losses['perc'].append(perc)
        self.losses['perplexity'].append(perplexity)
        if disc_g > 0:
            self.losses['disc_g'].append(disc_g)
        if disc_d > 0:
            self.losses['disc_d'].append(disc_d)
    
    def get_avg(self, key):
        if len(self.losses[key]) == 0:
            return 0.0
        return sum(self.losses[key]) / len(self.losses[key])
    
    def get_all_avg(self):
        return {k: self.get_avg(k) for k in self.losses}

def calculate_perplexity(encodings, num_embeddings):
    """✅ Calculate perplexity metric for VQ codebook usage
    
    Perplexity measures how many codebook entries are being actively used.
    Higher perplexity = better utilization of the codebook.
    
    Args:
        encodings: one-hot encodings from VQ layer [B*H*W, num_embeddings]
        num_embeddings: total number of codebook entries
    
    Returns:
        perplexity: scalar value (higher is better, max = num_embeddings)
    """
    try:
        # Ensure float dtype (encodings are already one-hot from quantizer)
        encodings = encodings.float()
        
        # Average encoding across batch
        avg_probs = encodings.mean(dim=0)  # [num_embeddings]
        
        # Clamp to avoid log(0)
        avg_probs = torch.clamp(avg_probs, min=1e-10)
        
        # Calculate entropy: -sum(p * log(p))
        entropy = -torch.sum(avg_probs * torch.log(avg_probs))
        
        # Perplexity = exp(entropy)
        perplexity = torch.exp(entropy)
        
        return perplexity.item()
    except Exception as e:
        raise ValueError(f"Perplexity calculation failed: {e}")

def sample_and_save(model, original_imgs, num_samples, save_path, device):
    """✅ FIXED: Generate and save progressive decoding without upsampling codes
    
    Strategy: Pass codes at NATIVE resolutions to decoder
    - q_t: 8x8    -> dec_t (stride=2) -> 16x16
    - q_m: 16x16  -> dec_m (stride=2) -> 32x32
    - q_b: 32x32  -> dec_b (stride=2x2) -> 128x128
    """
    model.eval()
    
    try:
        actual_num_samples = min(num_samples, original_imgs.shape[0])
        original_imgs_subset = original_imgs[:actual_num_samples].to(device)
        
        with torch.no_grad():
            # 1. Encode through all levels
            feat_b = model.enc_b(original_imgs_subset)
            feat_m = model.enc_m(feat_b)
            feat_t = model.enc_t(feat_m)
            
            # 2. Quantize at all levels (at native resolutions)
            q_t, _, _ = model.quant_t(model.quant_conv_t(feat_t))  # 8x8
            q_m, _, _ = model.quant_m(model.quant_conv_m(feat_m))  # 16x16
            q_b, _, _ = model.quant_b(model.quant_conv_b(feat_b))  # 32x32
            
            # 3. Progressive decoding (pass native resolutions directly)
            
            # Stage 1: Only q_t (top level - coarsest)
            z_m = torch.zeros_like(q_m)
            z_b = torch.zeros_like(q_b)
            stage_1 = model.decode_codes(q_t, z_m, z_b)
            
            # Stage 2: q_t + q_m (top + middle - mid-resolution)
            stage_2 = model.decode_codes(q_t, q_m, z_b)
            
            # Stage 3: All levels (q_t + q_m + q_b - finest)
            stage_3 = model.decode_codes(q_t, q_m, q_b)
        
        # ✅ Verify batch sizes match
        assert original_imgs_subset.shape[0] == stage_1.shape[0] == stage_2.shape[0] == stage_3.shape[0], \
            f"Batch mismatch: orig={original_imgs_subset.shape[0]}, s1={stage_1.shape[0]}, s2={stage_2.shape[0]}, s3={stage_3.shape[0]}"
        
        # ✅ Create progressive visualization grid
        rows = []
        for i in range(actual_num_samples):
            row = torch.stack([
                original_imgs_subset[i],
                stage_1[i],
                stage_2[i],
                stage_3[i]
            ], dim=0)
            rows.append(row)
        
        # grid_tensor = torch.cat(rows, dim=0)
        # grid_img = make_grid(grid_tensor, nrow=4, normalize=True, value_range=(-1, 1), padding=2, pad_value=0.5)
        # save_image(grid_img, save_path)
        
        # ✅ Save detailed comparison plot
        save_comparison_plot(
            original_imgs_subset, stage_1, stage_2, stage_3,
            save_path.replace('.png', '_comparison.png'),
            actual_num_samples
        )
        
    except Exception as e:
        print(f"⚠️  Could not save samples: {e}. Skipping sample save.")
    
    model.train()

def save_comparison_plot(original, stage1, stage2, stage3, save_path, num_samples=8):
    """✅ NEW: Create detailed matplotlib comparison visualization"""
    try:
        # Ensure all tensors are on CPU and have matching batch sizes
        original = original.cpu()
        stage1 = stage1.cpu()
        stage2 = stage2.cpu()
        stage3 = stage3.cpu()
        
        # Get actual batch size from original (smallest dimension)
        actual_batch = min(original.shape[0], stage1.shape[0], stage2.shape[0], stage3.shape[0])
        num_samples = min(num_samples, actual_batch)
        
        # Slice all to same batch size
        original = original[:num_samples]
        stage1 = stage1[:num_samples]
        stage2 = stage2[:num_samples]
        stage3 = stage3[:num_samples]
        
        # Denormalize from [-1, 1] to [0, 1]
        def denorm(x):
            return torch.clamp((x + 1) / 2, 0, 1)
        
        original = denorm(original)
        stage1 = denorm(stage1)
        stage2 = denorm(stage2)
        stage3 = denorm(stage3)
        
        # Create figure with subplots: 4 rows (stages) x num_samples columns
        fig, axes = plt.subplots(4, num_samples, figsize=(num_samples*2, 8))
        
        # Handle case where num_samples=1 (axes is 1D instead of 2D)
        if num_samples == 1:
            axes = axes.reshape(4, 1)
        
        stage_names = ['Original', 'Stage 1 (Top)', 'Stage 2 (Top+Mid)', 'Stage 3 (Full)']
        stages = [original, stage1, stage2, stage3]
        
        for row_idx, (stage_name, stage_tensor) in enumerate(zip(stage_names, stages)):
            for col_idx in range(num_samples):
                ax = axes[row_idx, col_idx]
                
                # Convert to numpy for display
                img_np = stage_tensor[col_idx].permute(1, 2, 0).numpy()
                img_np = np.clip(img_np, 0, 1)
                
                ax.imshow(img_np)
                ax.axis('off')
                
                # Label only the first column
                if col_idx == 0:
                    ax.set_ylabel(stage_name, fontsize=10, fontweight='bold')
        
        # Add column labels (sample indices)
        for col_idx in range(num_samples):
            axes[0, col_idx].set_title(f'Sample {col_idx+1}', fontsize=9)
        
        # Overall title
        fig.suptitle('HVQ-GAN Progressive Decoding Process', fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"⚠️  Could not save comparison plot: {e}")

def train():
    # 1. Setup - Create directory structure
    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)
    
    # ✅ NEW: Create checkpoints subfolder
    checkpoints_dir = os.path.join(cfg.save_dir, "checkpoints")
    samples_dir = os.path.join(cfg.save_dir, "samples")
    logs_dir = os.path.join(cfg.save_dir, "logs")
    metrics_viz_dir = os.path.join(cfg.save_dir, "metrics_and_visualisations")
    gan_csv_path = os.path.join(logs_dir, "gan_loss_log.csv")
    
    for dir_path in [checkpoints_dir, samples_dir, logs_dir, metrics_viz_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    if not os.path.exists(gan_csv_path):
        with open(gan_csv_path, "w") as f:
            f.write("step,loss_g,adv_g,disc_d,recon,vq,perc,perplexity\n")
        
    print("="*70)
    print("🔧 HVQ-GAN TRAINING V3 (Step-based, Unsupervised)")
    print("="*70)
    print(f"Device: {cfg.device}")
    print(f"Total Steps: {cfg.total_steps:,}")
    print(f"Batch Size: {cfg.batch_size}")
    print(f"Log Interval: {cfg.log_interval} steps")
    print(f"Save Interval: {cfg.save_interval:,} steps")
    print(f"Sample Interval: {cfg.sample_interval:,} steps")
    print("="*70)
    print(f"📁 Output Directory: {cfg.save_dir}")
    print(f"   ├── checkpoints/              (model checkpoints)")
    print(f"   ├── samples/                  (progressive decoding visualizations)")
    print(f"   ├── metrics_and_visualisations/ (training metrics & analysis)")
    print(f"   └── logs/                     (training logs)")
    print("="*70 + "\n")
    
    # 2. Model & Optimizers
    model = HVQVAE_3Level(cfg).to(cfg.device)
    discriminator = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3).to(cfg.device)
    
    optimizer_g = optim.Adam(model.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=cfg.learning_rate * 0.5, betas=(0.5, 0.999))
    
    scaler_g = GradScaler(enabled=cfg.use_amp)
    scaler_d = GradScaler(enabled=cfg.use_amp)
    
    # 3. Perceptual Loss
    print("Loading VGG for Perceptual Loss...")
    perceptual_fn = VGGPerceptualLoss().to(cfg.device).eval()
    print("✅ VGG loaded\n")
    
    # 4. Data
    dataloader = get_loader(cfg)
    print(f"Training images: {len(dataloader.dataset)}")
    print(f"Batches per cycle: {len(dataloader)}\n")
    
    # 5. Loss tracker
    loss_tracker = LossTracker(window_size=cfg.log_interval)
    
    # ✅ NEW: Initialize metrics and visualization
    metrics_calc = MetricsCalculator(device=cfg.device)
    codebook_analyzer = CodebookAnalyzer(num_embeddings=cfg.num_embeddings, num_levels=3)
    
    # 6. Training Loop
    model.train()
    discriminator.train()
    
    step = 0
    dataloader_iter = iter(dataloader)
    nan_steps = 0
    max_nan_tolerance = 5
    
    with tqdm(total=cfg.total_steps, desc="Training", unit="step") as pbar:
        while step < cfg.total_steps:
            try:
                images = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                images = next(dataloader_iter)
            
            images = images.to(cfg.device)
            
            # ✅ SAFETY: Check input
            if torch.isnan(images).any() or torch.isinf(images).any():
                print(f"\n❌ Step {step}: NaN/Inf in input batch!")
                continue
            
            # ========== GENERATOR UPDATE ==========
            optimizer_g.zero_grad()
            
            with autocast(enabled=cfg.use_amp):
                # ✅ NEW: Enable cascaded dropout (25-25-50 split)
                # dropout_rate=1.0 activates the cascaded dropout logic in model
                recon, vq_loss, encodings = model(images, dropout_rate=1.0)
                
                # ✅ SAFETY: Check recon
                if torch.isnan(recon).any() or torch.isinf(recon).any():
                    print(f"\n❌ Step {step}: NaN/Inf in reconstruction!")
                    nan_steps += 1
                    if nan_steps > max_nan_tolerance:
                        raise RuntimeError(f"Too many NaN steps ({nan_steps}). Training unstable.")
                    continue
                
                loss_vae, recon_l, vq_l, perc_l = calculate_loss(
                    recon, images, vq_loss, cfg.weights, perceptual_fn
                )
                
                # ✅ SAFETY: Check losses
                if torch.isnan(loss_vae) or torch.isinf(loss_vae):
                    print(f"\n❌ Step {step}: NaN/Inf in VAE loss!")
                    nan_steps += 1
                    if nan_steps > max_nan_tolerance:
                        raise RuntimeError(f"Too many NaN steps ({nan_steps}). Training unstable.")
                    continue
                
                # ✅ Calculate perplexity from VQ encodings
                perplexity_val = 0.0
                if encodings is not None:
                    try:
                        if isinstance(encodings, tuple):
                            perp_values = []
                            for enc in encodings:
                                if enc is not None and hasattr(enc, 'shape'):
                                    if len(enc.shape) > 2:
                                        enc_flat = enc.view(-1, enc.size(-1))
                                    else:
                                        enc_flat = enc
                                    perp_values.append(calculate_perplexity(enc_flat, enc_flat.size(-1)))
                            perplexity_val = sum(perp_values) / len(perp_values) if perp_values else 0.0
                        elif hasattr(encodings, 'shape'):
                            if len(encodings.shape) > 2:
                                encodings_flat = encodings.view(-1, encodings.size(-1))
                            else:
                                encodings_flat = encodings
                            perplexity_val = calculate_perplexity(encodings_flat, encodings_flat.size(-1))
                    except Exception as e:
                        print(f"⚠️  Warning: Could not calculate perplexity at step {step}: {e}")
                        perplexity_val = 0.0
                
                disc_active = step >= cfg.disc_start_step
                loss_disc_gen = torch.tensor(0.0, device=images.device)
                if disc_active:
                    disc_recon = discriminator(recon)
                    loss_disc_gen = -disc_recon.mean()
                
                loss_g = loss_vae + cfg.weights['disc'] * loss_disc_gen
            
            scaler_g.scale(loss_g).backward()
            scaler_g.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_g.step(optimizer_g)
            scaler_g.update()
            
            # ========== DISCRIMINATOR UPDATE ==========
            loss_d_val = 0.0
            if disc_active:
                optimizer_d.zero_grad()
                
                with autocast(enabled=cfg.use_amp):
                    disc_real = discriminator(images)
                    disc_fake = discriminator(recon.detach())
                    
                    loss_d_val = F.softplus(-disc_real).mean() + F.softplus(disc_fake).mean()
                
                if torch.isnan(loss_d_val) or torch.isinf(loss_d_val):
                    print(f"\n❌ Step {step}: NaN/Inf in discriminator loss!")
                    nan_steps += 1
                    if nan_steps > max_nan_tolerance:
                        raise RuntimeError(f"Too many NaN steps ({nan_steps}). Training unstable.")
                    continue
                
                scaler_d.scale(loss_d_val).backward()
                scaler_d.unscale_(optimizer_d)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                scaler_d.step(optimizer_d)
                scaler_d.update()
            
            # Reset NaN counter on successful step
            nan_steps = 0
            
            # Track losses
            loss_tracker.add(
                loss_g.item(),
                recon_l.item(),
                vq_l.item(),
                perc_l.item(),
                loss_disc_gen.item() if disc_active else 0,
                loss_d_val.item() if disc_active and torch.is_tensor(loss_d_val) else 0,
                perplexity_val
            )
            
            # Update progress bar
            step += 1
            pbar.update(1)

            adv_val = loss_disc_gen.item() if disc_active else 0.0
            disc_val = loss_d_val.item() if disc_active and torch.is_tensor(loss_d_val) else 0.0
            with open(gan_csv_path, "a") as f:
                f.write(
                    f"{step},{loss_g.item():.6f},{adv_val:.6f},{disc_val:.6f},"
                    f"{recon_l.item():.6f},{vq_l.item():.6f},{perc_l.item():.6f},{perplexity_val:.6f}\n"
                )
            
            # Log every N steps
            if step % cfg.log_interval == 0:
                avg_losses = loss_tracker.get_all_avg()
                postfix = {
                    'Tot': f"{avg_losses['total']:.4f}",
                    'Rec': f"{avg_losses['recon']:.4f}",
                    'VQ': f"{avg_losses['vq']:.4f}",
                    'Perc': f"{avg_losses['perc']:.4f}",
                    'Ppl': f"{avg_losses['perplexity']:.2f}",
                }
                if step >= cfg.disc_start_step:
                    postfix['DiscG'] = f"{avg_losses['disc_g']:.4f}"
                    postfix['DiscD'] = f"{avg_losses['disc_d']:.4f}"
                
                pbar.set_postfix(postfix)
                
                # Log to file
                with open(os.path.join(logs_dir, "training_log.txt"), "a") as f:
                    f.write(f"Step {step:,} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | " + 
                            " | ".join([f"{k}={v}" for k, v in postfix.items()]) + "\n")
            
            # Sample every N steps
            if step % cfg.sample_interval == 0 and step > 0:
                sample_path = os.path.join(samples_dir, f"progressive_step_{step}.png")
                sample_and_save(model, images, cfg.num_samples, sample_path, cfg.device)
            
            # ✅ NEW: Save checkpoint + metrics + visualizations every N steps
            if step % cfg.save_interval == 0 and step > 0:
                # 1. Save checkpoint (existing logic)
                checkpoint = {
                    'step': step,
                    'model_state': model.state_dict(),
                    'discriminator_state': discriminator.state_dict(),
                    'optimizer_g_state': optimizer_g.state_dict(),
                    'optimizer_d_state': optimizer_d.state_dict(),
                }
                ckpt_path = os.path.join(checkpoints_dir, f"checkpoint_step_{step}.pth")
                torch.save(checkpoint, ckpt_path)
                
                # Also save as latest
                latest_path = os.path.join(checkpoints_dir, "model_latest.pth")
                torch.save(model.state_dict(), latest_path)
                
                # ✅ NEW: Compute and save metrics
                with torch.no_grad():
                    sample_metrics = metrics_calc.compute_metrics(images, recon, step)
                    codebook_analyzer.update(encodings)
                
                # ✅ NEW: Generate progressive stages for visualization
                with torch.no_grad():
                    model.eval()
                    feat_b = model.enc_b(images[:4])
                    feat_m = model.enc_m(feat_b)
                    feat_t = model.enc_t(feat_m)
                    q_t, _, _ = model.quant_t(model.quant_conv_t(feat_t))
                    q_m, _, _ = model.quant_m(model.quant_conv_m(feat_m))
                    q_b, _, _ = model.quant_b(model.quant_conv_b(feat_b))
                    
                    z_m = torch.zeros_like(q_m)
                    z_b = torch.zeros_like(q_b)
                    stage_1 = model.decode_codes(q_t, z_m, z_b)
                    stage_2 = model.decode_codes(q_t, q_m, z_b)
                    stage_3 = model.decode_codes(q_t, q_m, q_b)
                    model.train()
                
                # ✅ NEW: Create comprehensive evaluation report (saved to metrics_and_visualisations)
                try:
                    create_evaluation_report(
                        loss_tracker, 
                        metrics_calc.history,
                        codebook_analyzer,
                        images[:4],
                        stage_1,
                        stage_2,
                        stage_3,
                        metrics_viz_dir,
                        step
                    )
                except Exception as e:
                    print(f"⚠️  Could not create evaluation report: {e}")
                
                # ✅ NEW: Save metrics history (to metrics_and_visualisations)
                try:
                    metrics_path = metrics_calc.save_metrics(metrics_viz_dir)
                    print(f"   Metrics history: {os.path.basename(metrics_path)}")
                except Exception as e:
                    print(f"⚠️  Could not save metrics: {e}")
                
                # Print summary
                print(f"\n✅ Step {step:,} Checkpoint:")
                print(f"   Checkpoint: {os.path.basename(ckpt_path)}")
                print(f"   Metrics: MSE={sample_metrics['mse']:.4f}, PSNR={sample_metrics['psnr']:.2f} dB, SSIM={sample_metrics['ssim']:.3f}")
                codebook_metrics = codebook_analyzer.get_diversity_metrics()
                if codebook_metrics:
                    print(f"   Codebook: Top usage={codebook_metrics.get('Top_usage_ratio', 0):.1%}, "
                          f"Mid usage={codebook_metrics.get('Mid_usage_ratio', 0):.1%}, "
                          f"Bottom usage={codebook_metrics.get('Bottom_usage_ratio', 0):.1%}")
    
    print("\n" + "="*70)
    print("✅ HVQ-GAN TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📁 Output Structure:")
    print(f"   {cfg.save_dir}/")
    print(f"   ├── checkpoints/")
    print(f"   │   ├── checkpoint_step_5000.pth")
    print(f"   │   ├── checkpoint_step_10000.pth")
    print(f"   │   └── model_latest.pth")
    print(f"   ├── samples/")
    print(f"   │   ├── progressive_step_2500_comparison.png")
    print(f"   │   ├── progressive_step_5000_comparison.png")
    print(f"   │   └── progressive_step_10000_comparison.png")
    print(f"   ├── metrics_and_visualisations/")
    print(f"   │   ├── loss_curves_step_5000.png")
    print(f"   │   ├── metrics_comparison_step_5000.png")
    print(f"   │   ├── cascade_hierarchy_step_5000.png")
    print(f"   │   ├── codebook_usage_step_5000.png")
    print(f"   │   ├── codebook_analysis_step_5000.txt")
    print(f"   │   └── metrics_history.csv")
    print(f"   └── logs/")
    print(f"       └── training_log.txt")
    print(f"\n📊 Final step: {step:,}/{cfg.total_steps:,}")
    print(f"\n💡 Each sample shows 4 stages:")
    print(f"   1. Original input image")
    print(f"   2. Stage 1 - Top level only (coarsest, 8x8 code)")
    print(f"   3. Stage 2 - Top + Middle (mid-resolution, 16x16 code)")
    print(f"   4. Stage 3 - All levels (finest, full reconstruction)")