import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
import numpy as np
import os
import csv
from datetime import datetime
import json
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from .config import Config
from .generator import CounterfactualGenerator
from .loss_functions import CounterfactualLossManager
from .dataset import CelebADataset, get_loader
from .visualizer import CounterfactualVisualizer


class ComprehensiveTrainer:
    def __init__(self, cfg, experiment_name="counterfactual_training"):
        """
        Initialize comprehensive trainer with logging and visualization.
        
        Args:
            cfg: Config object
            experiment_name: Name for this experiment (used for directories)
        """
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.experiment_name = experiment_name
        
        # Setup directories
        self.setup_directories()
        
        # Initialize models
        self.init_models()
        
        # Setup logging
        self.setup_logging()
        
        # Visualization
        self.visualizer = CounterfactualVisualizer(self.device)
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Attribute name mapping (CelebA 40 attributes)
        self.attr_names = self._get_attribute_names()
        self.decoder_pretrained = False
        
    def setup_directories(self):
        """Create output directories"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(f"outputs/experiments/{self.experiment_name}_{timestamp}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.log_dir = self.exp_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        self.vis_dir = self.exp_dir / "visualizations"
        self.vis_dir.mkdir(exist_ok=True)
        
        print(f"📁 Experiment directory: {self.exp_dir}")
    
    def init_models(self):
        """Initialize all models"""
        self.model = CounterfactualGenerator(
            self.cfg,
            vqvae_path=self.cfg.vqvae_checkpoint,
            classifier_path=self.cfg.classifier_checkpoint,
            device=self.device
        )
        
        self.loss_manager = CounterfactualLossManager(device=self.device)
        
        # Optimizer: only Mutator
        self.optimizer = optim.AdamW(
            self.model.mutator.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay
        )
        
        self.scaler = GradScaler()
        self.decoder_pretrained = False

        decoder_ckpt_path = getattr(self.cfg, 'decoder_checkpoint_path', '')
        if decoder_ckpt_path:
            ckpt_path = Path(decoder_ckpt_path)
            if ckpt_path.is_file():
                self.model.decoder.load_state_dict(torch.load(ckpt_path, map_location=self.device))
                self.decoder_pretrained = True
                print(f"🧭 Loaded decoder weights from {ckpt_path}")
            else:
                if getattr(self.cfg, 'decoder_pretrain_epochs', 0) > 0:
                    print(f"ℹ️ Decoder checkpoint not found at {ckpt_path}; will train the decoder from scratch.")
                else:
                    print(f"⚠️ Decoder checkpoint not found at {ckpt_path}; starting with random weights.")
        else:
            if getattr(self.cfg, 'decoder_pretrain_epochs', 0) > 0:
                print("ℹ️ No decoder checkpoint supplied; decoder will be trained from scratch.")
    
    def setup_logging(self):
        """Setup Tensorboard and CSV logging"""
        self.writer = SummaryWriter(str(self.log_dir))
        
        # CSV logging
        self.csv_path = self.log_dir / "training_log.csv"
        self.csv_file = open(self.csv_path, 'w', newline='')
        
        fieldnames = [
            'epoch', 'batch', 'step', 'total_loss',
            'cf_loss', 'retention_loss', 'latent_prox_loss',
            'ortho_loss', 'sparse_loss', 'lr'
        ] + [f'attr_{i}_cf_loss' for i in range(40)]
        
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        self.csv_file.flush()
        
    def _get_attribute_names(self):
        """Get CelebA attribute names"""
        return [
            '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
            'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
            'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
            'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
            'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
            'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
            'Sideburns', 'Smiling', 'Straight_Hair', 'Stubble', 'Sunglass',
            'Sweating', 'Thick_Lips', 'Thin_Lips', 'Wearing_Earrings', 'Young'
        ]
    
    def train_epoch(self, epoch, loader):
        """Train for one epoch with comprehensive logging"""
        self.model.mutator.train()
        
        epoch_losses = {
            'total': [], 'cf': [], 'retention': [], 'latent_prox': [],
            'ortho': [], 'sparse': []
        }
        attr_losses = {i: [] for i in range(40)}
        
        pbar = tqdm(loader, desc=f"Epoch {epoch} / {self.cfg.num_epochs}")
        accumulation_steps = self.cfg.accumulation_steps
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)  # [B, 40]
            
            # --- Random attribute selection ---
            attr_idx = random.randint(0, 39)
            target_labels = labels.clone()
            target_labels[:, attr_idx] = 1 - target_labels[:, attr_idx]
            
            # --- 2. XAI Extraction (Fixed with Smoothing) ---
            images.requires_grad = True
            self.model.classifier.zero_grad()

            logits = self.model.classifier(images)
            target_score = logits[:, attr_idx].sum()
            grads = torch.autograd.grad(target_score, images, create_graph=False)[0]

            raw_saliency = torch.abs(images * grads).sum(dim=1)
            with torch.no_grad():
                smooth_mask = F.avg_pool2d(raw_saliency.unsqueeze(1), kernel_size=15, stride=1, padding=7)
                flat = smooth_mask.view(smooth_mask.size(0), -1)
                max_val = flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
                mask_norm = smooth_mask / (max_val + 1e-8)
                cam_128 = (mask_norm > 0.2).float().squeeze(1)

                mask_t = F.interpolate(cam_128.unsqueeze(1), size=(8, 8)).squeeze(1)
                mask_m = F.interpolate(cam_128.unsqueeze(1), size=(16, 16)).squeeze(1)
                mask_b = F.interpolate(cam_128.unsqueeze(1), size=(32, 32)).squeeze(1)
                masks = [mask_t, mask_m, mask_b]
                probs = torch.sigmoid(logits.detach())

            images.requires_grad = False
            
            with torch.no_grad():
                z_orig_list = self.model.vqvae.get_codes(images)
            
            with autocast():
                use_hard = (epoch >= 5)

                x_new, z_mutated = self.model(
                    images,
                    raw_saliency.detach(),
                    masks,
                    target_labels,
                    probs,
                    attr_idx,
                    hard=use_hard
                )
                # === DEBUG PROBE START ===
                # if batch_idx % 10 == 0:
                #     print(f"\n--- DEBUG STEP {self.global_step} ---")
                #     print(f"GRAD MAX: {grads.abs().max().item():.6f}")
                #     if grads.abs().max() < 1e-6:
                #         print("⚠️ CRITICAL: Gradients are zero! XAI is blind.")
                #     print(f"MASK MEAN: {masks[1].mean().item():.4f} | MAX: {masks[1].max().item():.4f}")
                #     with torch.no_grad():
                #         diff = (x_new - images).abs().mean().item()
                #     print(f"IMG DIFF: {diff:.6f}")
                #     if diff < 1e-4:
                #         print("⚠️ CRITICAL: Output image is identical to Input! Decoder disconnected.")
                #     else:
                #         print("✅ Image is changing (Signal is flowing)")
                # === DEBUG PROBE END ===
                new_logits = self.model.classifier(x_new)
                
                # Loss computation
                flip_mask = torch.ones(labels.size(0), 40, device=self.device)
                losses = self.loss_manager.generator_loss(
                    images, x_new,
                    new_logits, target_labels, probs,
                    flip_mask,
                    z_orig_list, z_mutated,  # ✅ Now passes actual z_orig_list
                    masks,
                    self.model.mutator.directions
                )
                
                total_loss = losses['total'] / accumulation_steps
            
            # --- Backward pass ---
            self.scaler.scale(total_loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Record losses
                for key in epoch_losses.keys():
                    epoch_losses[key].append(losses[key].item())
                
                attr_losses[attr_idx].append(losses['cf'].item())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}",
                    'cf': f"{losses['cf'].item():.3f}",
                    'ret': f"{losses['retention'].item():.3f}"
                })
            
            # Memory cleanup
            del x_new, new_logits, grads, raw_saliency, losses
            torch.cuda.empty_cache()
        
        # Log epoch summary
        self._log_epoch_summary(epoch, epoch_losses, attr_losses)
        
        return {k: np.mean(v) for k, v in epoch_losses.items()}
    
    def _log_step(self, epoch, batch_idx, losses, attr_idx):
        """Log per-step metrics"""
        self.writer.add_scalar('Loss/total', losses['total'].item(), self.global_step)
        self.writer.add_scalar('Loss/cf', losses['cf'].item(), self.global_step)
        self.writer.add_scalar('Loss/retention', losses['retention'].item(), self.global_step)
        self.writer.add_scalar('Loss/latent_prox', losses['latent_prox'].item(), self.global_step)
        self.writer.add_scalar('Loss/ortho', losses['ortho'].item(), self.global_step)
        self.writer.add_scalar('Loss/sparse', losses['sparse'].item(), self.global_step)
        
        row = {
            'epoch': epoch,
            'batch': batch_idx,
            'step': self.global_step,
            'total_loss': losses['total'].item(),
            'cf_loss': losses['cf'].item(),
            'retention_loss': losses['retention'].item(),
            'latent_prox_loss': losses['latent_prox'].item(),
            'ortho_loss': losses['ortho'].item(),
            'sparse_loss': losses['sparse'].item(),
            'lr': self.optimizer.param_groups[0]['lr'],
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        self.csv_writer.writerow(row)
        self.csv_file.flush()
    
    def _log_epoch_summary(self, epoch, epoch_losses, attr_losses):
        """Log per-epoch summary"""
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} Summary")
        print(f"{'='*60}")
        
        for loss_name, loss_values in epoch_losses.items():
            mean_loss = np.mean(loss_values)
            self.writer.add_scalar(f'Epoch/Loss_{loss_name}', mean_loss, epoch)
            print(f"  {loss_name}: {mean_loss:.4f}")
        
        # Per-attribute loss summary
        attr_means = {}
        for attr_idx, losses in attr_losses.items():
            if losses:
                mean = np.mean(losses)
                attr_means[attr_idx] = mean
                attr_name = self.attr_names[attr_idx]
                if mean < 0.5:
                    print(f"  {attr_name}: {mean:.4f}")
        
        # Save per-attribute stats
        attr_stats = {
            self.attr_names[i]: {
                'mean_loss': float(attr_means.get(i, -1)),
                'num_updates': len(attr_losses[i])
            }
            for i in range(40)
        }
        
        with open(self.log_dir / f'attr_stats_epoch_{epoch}.json', 'w') as f:
            json.dump(attr_stats, f, indent=2)
    
    def validate_and_visualize(self, epoch, loader, num_samples=4):
        """Generate visualizations during validation"""
        self.model.mutator.eval()
        
        print(f"\n🎨 Generating visualizations for epoch {epoch}...")
        
        vis_count = 0
        for images, labels in loader:
            if vis_count >= num_samples:
                break
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            batch_size = images.size(0)
            for sample_idx in range(batch_size):
                if vis_count >= num_samples:
                    break
                
                img = images[sample_idx:sample_idx+1]
                lbl = labels[sample_idx:sample_idx+1]
                
                # Generate 3 counterfactuals with different attributes
                cf_results = self._generate_counterfactuals(img, lbl, num_cf=3)
                
                # Visualize
                fig = self.visualizer.create_comparison_grid(
                    img, cf_results, lbl, self.attr_names
                )
                
                fig.savefig(
                    self.vis_dir / f'epoch_{epoch:03d}_sample_{vis_count:03d}.png',
                    dpi=150, bbox_inches='tight'
                )
                
                self.writer.add_figure(
                    f'Counterfactuals/sample_{vis_count}',
                    fig,
                    epoch
                )
                
                vis_count += 1
        
        print(f"✅ Saved {vis_count} visualizations")
    
    def plot_inference(self, loader, epoch, num_samples=4):
        """Plot inference results"""
        if loader is None:
            return
        self.model.mutator.eval()
        inference_dir = self.vis_dir / "inference"
        inference_dir.mkdir(exist_ok=True)
        data_iter = iter(loader)
        vis_count = 0
        while vis_count < num_samples:
            try:
                images, labels = next(data_iter)
            except StopIteration:
                break
            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)
            for sample_idx in range(batch_size):
                if vis_count >= num_samples:
                    break
                img = images[sample_idx:sample_idx+1]
                lbl = labels[sample_idx:sample_idx+1]
                cf_results = self._generate_counterfactuals(img, lbl, num_cf=3)
                fig = self.visualizer.create_comparison_grid(
                    img, cf_results, lbl, self.attr_names
                )
                fig.savefig(
                    inference_dir / f'epoch_{epoch:03d}_sample_{vis_count:03d}.png',
                    dpi=150, bbox_inches='tight'
                )
                self.writer.add_figure(
                    f'Inference/sample_{vis_count}',
                    fig,
                    epoch
                )
                vis_count += 1
        self.model.mutator.train()
        print(f"🖼️ Inference plots saved: {vis_count}")

    def _generate_counterfactuals(self, img, labels, num_cf=3):
        results = {'original': img.cpu(), 'cfs': []}

        with torch.no_grad():
            probs = torch.sigmoid(self.model.classifier(img))

        for cf_idx in range(num_cf):
            attr_idx = random.randint(0, 39)
            target_labels = labels.clone()
            target_labels[:, attr_idx] = 1 - target_labels[:, attr_idx]

            with torch.enable_grad():
                img_grad = img.clone().detach()
                img_grad.requires_grad = True
                logits = self.model.classifier(img_grad)
                target_score = logits[:, attr_idx].sum()
                grads = torch.autograd.grad(target_score, img_grad, create_graph=False)[0]
                ig_map = torch.abs(img_grad * grads).sum(dim=1)

            with torch.no_grad():
                cam_128 = ig_map
                mask_t = F.interpolate(cam_128.unsqueeze(1), size=(8, 8)).squeeze(1)
                mask_m = F.interpolate(cam_128.unsqueeze(1), size=(16, 16)).squeeze(1)
                mask_b = F.interpolate(cam_128.unsqueeze(1), size=(32, 32)).squeeze(1)
                masks = [mask_t, mask_m, mask_b]

                x_cf, _ = self.model(img, ig_map, masks, target_labels, probs, attr_idx, hard=True)

            results['cfs'].append({
                'image': x_cf.cpu(),
                'attr_idx': attr_idx,
                'attr_name': self.attr_names[attr_idx],
                'cam_mask': cam_128.cpu()
            })

        return results
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'mutator_state': self.model.mutator.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scaler_state': self.scaler.state_dict(),
            'global_step': self.global_step,
            'config': vars(self.cfg) if hasattr(self.cfg, '__dict__') else str(self.cfg)
        }
        
        ckpt_path = self.checkpoint_dir / f'epoch_{epoch:03d}.pth'
        torch.save(checkpoint, ckpt_path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"✨ Best checkpoint saved: {best_path}")
    
    def recon_training(self, loader):
        """Pre-train the decoder with reconstruction loss"""
        decoder_epochs = getattr(self.cfg, 'decoder_pretrain_epochs', 0)
        if decoder_epochs <= 0:
            print("ℹ️ decoder_pretrain_epochs <= 0 — skipping decoder pre-training.")
            return

        decoder_lr = getattr(self.cfg, 'decoder_lr', 1e-3)
        decoder_betas = getattr(self.cfg, 'decoder_betas', (0.0, 0.99))
        ckpt_dir = Path(getattr(self.cfg, 'decoder_checkpoint_dir', self.checkpoint_dir / "decoder"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        primary_ckpt = ckpt_dir / "decoder_pretrained.pth"
        latest_ckpt = ckpt_dir / "latest.pth"
        preview_dir = ckpt_dir / "previews"
        preview_dir.mkdir(exist_ok=True)

        preview_imgs_cpu = next(iter(loader))[0][:4].clone()

        print(f"🎯 Starting StyleGAN decoder pre-training for {decoder_epochs} epoch(s)")
        self.model.decoder.train()
        optimizer = optim.Adam(self.model.decoder.parameters(), lr=decoder_lr, betas=decoder_betas)
        scaler = GradScaler()
        l1_loss = torch.nn.L1Loss()
        lpips_loss = self.loss_manager.lpips

        for epoch in range(decoder_epochs):
            pbar = tqdm(loader, desc=f"[Decoder Pretrain] {epoch}")
            for images, _ in pbar:
                images = images.to(self.device)
                optimizer.zero_grad()

                with torch.no_grad():
                    feat_b = self.model.vqvae.enc_b(images)
                    feat_m = self.model.vqvae.enc_m(feat_b)
                    feat_t = self.model.vqvae.enc_t(feat_m)
                    q_t, _, _ = self.model.vqvae.quant_t(self.model.vqvae.quant_conv_t(feat_t))
                    q_m, _, _ = self.model.vqvae.quant_m(self.model.vqvae.quant_conv_m(feat_m))
                    q_b, _, _ = self.model.vqvae.quant_b(self.model.vqvae.quant_conv_b(feat_b))
                    z_list = [q_t, q_m, q_b]

                with autocast():
                    recon = self.model.decoder(z_list)
                    loss_l1 = l1_loss(recon, images)
                    loss_lpips = lpips_loss(recon, images).mean()
                    loss = loss_l1 + loss_lpips

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                pbar.set_postfix({'L1': loss_l1.item(), 'LPIPS': loss_lpips.item()})

            print("🖼️ Generating reconstruction preview...")
            self.model.decoder.eval()
            with torch.no_grad():
                preview = preview_imgs_cpu.to(self.device)
                feat_b = self.model.vqvae.enc_b(preview)
                feat_m = self.model.vqvae.enc_m(feat_b)
                feat_t = self.model.vqvae.enc_t(feat_m)
                q_t, _, _ = self.model.vqvae.quant_t(self.model.vqvae.quant_conv_t(feat_t))
                q_m, _, _ = self.model.vqvae.quant_m(self.model.vqvae.quant_conv_m(feat_m))
                q_b, _, _ = self.model.vqvae.quant_b(self.model.vqvae.quant_conv_b(feat_b))
                recon_preview = self.model.decoder([q_t, q_m, q_b]).detach().cpu()

            grid = vutils.make_grid(
                torch.cat([preview_imgs_cpu, recon_preview], dim=0),
                nrow=4,
                normalize=True,
                value_range=(-1, 1)
            )
            fig = plt.figure(figsize=(10, 5))
            plt.title(f"Epoch {epoch} | Original (top) vs Reconstruction (bottom)")
            plt.axis('off')
            plt.imshow(grid.permute(1, 2, 0).numpy())
            preview_path = preview_dir / f"epoch_{epoch:03d}.png"
            fig.savefig(preview_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"🖼️ Saved decoder preview: {preview_path}")
            self.model.decoder.train()

            if epoch % 5 == 0:
                torch.save(self.model.decoder.state_dict(), primary_ckpt)
                torch.save(self.model.decoder.state_dict(), latest_ckpt)
                print(f"💾 Saved decoder checkpoints to {primary_ckpt} and {latest_ckpt}")

        self.model.decoder.eval()
        self.decoder_pretrained = True
        self.cfg.decoder_checkpoint_path = str(latest_ckpt)
        print("✅ Decoder pre-training complete")

    def train(self, num_epochs, loader_train, loader_val=None):
        """Full training loop"""
        print("🚀 Starting Comprehensive Counterfactual Training...")
        print(f"📊 Logging to: {self.log_dir}")
        print(f"💾 Checkpoints to: {self.checkpoint_dir}")
        print(f"🎨 Visualizations to: {self.vis_dir}")
        
        for epoch in range(num_epochs):
            epoch_losses = self.train_epoch(epoch, loader_train)
            
            # ✅ SAVE CHECKPOINT EVERY EPOCH
            is_best = epoch_losses['total'] < self.best_loss
            if is_best:
                self.best_loss = epoch_losses['total']
            self.save_checkpoint(epoch, is_best=is_best)
            
            if loader_val:
                self.validate_and_visualize(epoch, loader_val, num_samples=4)
            if epoch % 5 == 0:
                self.plot_inference(loader_val or loader_train, epoch, num_samples=4)
            if hasattr(self.cfg, 'lr_schedule'):
                self._update_learning_rate(epoch)
        
        print("\n✅ Training complete!")
        print(f"📁 Results saved to: {self.exp_dir}")
        
        self.writer.close()
        self.csv_file.close()
    
    def _update_learning_rate(self, epoch):
        """Update learning rate based on schedule"""
        if hasattr(self.cfg, 'lr_schedule'):
            schedule = self.cfg.lr_schedule
            if epoch in schedule:
                new_lr = schedule[epoch]
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"📉 Learning rate updated to {new_lr}")


def main():
    """Main training entry point"""
    cfg = Config()
    
    loader_train = get_loader(cfg, split='train', batch_size=cfg.batch_size)
    loader_val = get_loader(cfg, split='val', batch_size=cfg.batch_size)
    
    trainer = ComprehensiveTrainer(cfg, experiment_name="ceGAN_counterfactual")
    
    trainer.train(
        num_epochs=cfg.num_epochs,
        loader_train=loader_train,
        loader_val=loader_val
    )


if __name__ == '__main__':
    main()
