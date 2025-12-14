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
from .dataset import CelebADataset, get_loader, get_attribute_names
from .visualizer import CounterfactualVisualizer
from .discriminator_manager import PatchGANDiscriminator


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
        
        # Attribute name mapping (restricted CelebA subset)
        self.attr_names = self._get_attribute_names()
        self._initialize_active_attributes()

        # Setup logging
        self.setup_logging()
        
        # Visualization
        self.visualizer = CounterfactualVisualizer(self.device)
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        
        self.decoder_pretrained = False
        
    def setup_directories(self):
        """Create output directories"""
        timestamp = datetime.now().strftime("%Y%m%d")
        self.exp_dir = Path(f"outputs/synth_network/CF_generator/{self.experiment_name}_{timestamp}")
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
        
        for p in self.model.decoder.parameters():
            p.requires_grad = True

        self.optimizer = optim.AdamW(
            [
                {'params': self.model.mutator.parameters(), 'lr': self.cfg.learning_rate},
                {'params': self.model.decoder.parameters(), 'lr': self.cfg.learning_rate * 0.1}
            ],
            weight_decay=self.cfg.weight_decay
        )

        self.discriminator = PatchGANDiscriminator().to(self.device)
        self.discriminator.train()
        self.opt_d = optim.AdamW(
            self.discriminator.parameters(),
            lr=self.cfg.learning_rate * 0.5,
            betas=(0.5, 0.999)
        )
        self.adv_weight = getattr(self.cfg, 'adv_weight', 1.0)

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
    
    def _initialize_active_attributes(self):
        """Derive active attribute indices based on configuration."""
        desired_attrs = getattr(self.cfg, 'active_attributes', None)
        if desired_attrs:
            missing = [attr for attr in desired_attrs if attr not in self.attr_names]
            if missing:
                raise ValueError(f"Unknown attributes in config.active_attributes: {missing}")
            self.active_attr_names = list(desired_attrs)
        else:
            self.active_attr_names = list(self.attr_names)

        self.active_attr_indices = [self.attr_names.index(name) for name in self.active_attr_names]
        print(f"🎯 Active synthesis attributes ({len(self.active_attr_indices)}): {', '.join(self.active_attr_names)}")

    def setup_logging(self):
        """Setup Tensorboard and CSV logging"""
        self.writer = SummaryWriter(str(self.log_dir))
        
        # CSV logging
        self.csv_path = self.log_dir / "training_log.csv"
        self.csv_file = open(self.csv_path, 'w', newline='')
        
        attr_fields = [f"attr_{self.attr_names[idx]}_cf_loss" for idx in self.active_attr_indices]

        fieldnames = [
            'epoch', 'batch', 'step', 'total_loss',
            'cf_loss', 'retention_loss', 'latent_prox_loss',
            'ortho_loss', 'sparse_loss', 'adv_loss', 'disc_loss', 'lr',
            'embed_orig_top', 'embed_orig_mid', 'embed_orig_bot',
            'embed_cf_top', 'embed_cf_mid', 'embed_cf_bot',
            'step_top', 'step_mid', 'step_bot'
        ] + attr_fields
        
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        self.csv_file.flush()
        
    def _get_attribute_names(self):
        """Get active CelebA attribute names"""
        return get_attribute_names()

    def _set_discriminator_grad(self, requires_grad: bool):
        """Toggle gradient computation for discriminator parameters."""
        for param in self.discriminator.parameters():
            param.requires_grad_(requires_grad)
    
    def train_epoch(self, epoch, loader):
        """Train for one epoch with comprehensive logging"""
        self.model.mutator.train()
        self.model.decoder.train()
        self.discriminator.train()
        
        epoch_losses = {
            'total': [], 'cf': [], 'retention': [], 'latent_prox': [],
            'ortho': [], 'sparse': [], 'adv': [], 'disc': []
        }
        attr_losses = {i: [] for i in self.active_attr_indices}
        
        pbar = tqdm(loader, desc=f"Epoch {epoch} / {self.cfg.num_epochs}")
        accumulation_steps = self.cfg.accumulation_steps
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # --- Random attribute selection ---
            attr_idx = random.choice(self.active_attr_indices)
            target_labels = labels.clone()
            target_labels[:, attr_idx] = 1 - target_labels[:, attr_idx]
            
            # ====================================================
            # 1. XAI Extraction (Smoothed & Robust)
            # ====================================================
            images.requires_grad = True
            self.model.classifier.zero_grad()

            logits = self.model.classifier(images)
            target_score = logits[:, attr_idx].sum()
            grads = torch.autograd.grad(target_score, images, create_graph=False)[0]

            # --- SIGNAL 1: IG (Texture/Intensity) ---
            # Used by Mutator to decide "How much to change"
            # Keep this sharp/high-frequency
            raw_saliency = torch.abs(images * grads).sum(dim=1)
            ig_map = raw_saliency

            # --- SIGNAL 2: Pseudo-CAM (Shape/Location) ---
            # Used by Loss to decide "Where to allow changes"
            # We must turn the wireframe into a BLOB to allow full-region editing.
            smooth_mask = F.avg_pool2d(
                raw_saliency.unsqueeze(1),
                kernel_size=7,
                stride=1,
                padding=3
            )
            smooth_mask = F.avg_pool2d(smooth_mask, kernel_size=3, stride=1, padding=1)

            norm = smooth_mask / (smooth_mask.amax(dim=[1, 2, 3], keepdim=True) + 1e-8)
            norm = norm.pow(2)
            cam_binary = (norm > 0.45).float()

            kernel = torch.ones(1, 1, 3, 3, device=images.device)
            eroded = F.conv2d(cam_binary, kernel, padding=1)
            cam_128 = (eroded == 9).float()
            cam_soft = cam_128.squeeze(1)

            mask_t = F.interpolate(cam_128, size=(8, 8)).squeeze(1)
            mask_m = F.interpolate(cam_128, size=(16, 16)).squeeze(1)
            mask_b = F.interpolate(cam_128, size=(32, 32)).squeeze(1)
            masks = [mask_t, mask_m, mask_b]

            probs = torch.sigmoid(logits.detach())

            images.requires_grad = False

            with torch.no_grad():
                z_orig_list = self.model.vqvae.get_codes(images)

            use_hard = (epoch >= 5)

            # ====================================================
            # PHASE 1: Train Discriminator
            # ====================================================
            self._set_discriminator_grad(True)
            self.opt_d.zero_grad()

            with torch.no_grad():
                x_fake_detached, _, _ = self.model(
                    images,
                    ig_map,
                    masks,
                    target_labels,
                    probs,
                    attr_idx,
                    hard=use_hard
                )

            logits_real = self.discriminator(images)
            d_loss_real = F.relu(1.0 - logits_real).mean()

            logits_fake = self.discriminator(x_fake_detached)
            d_loss_fake = F.relu(1.0 + logits_fake).mean()

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.opt_d.step()
            disc_loss_value = d_loss.detach()

            # ====================================================
            # PHASE 2: Train Generator (Mutator + Decoder)
            # ====================================================
            self._set_discriminator_grad(False)

            # ====================================================
            # 2. Forward Pass (Curriculum Learning)
            # ====================================================
            with autocast():
                x_new, z_mutated, step_values = self.model(
                    images,
                    ig_map,
                    masks,
                    target_labels,
                    probs,
                    attr_idx,
                    hard=use_hard
                )

                new_logits = self.model.classifier(x_new)

                flip_mask = torch.zeros_like(labels)
                flip_mask[:, attr_idx] = 1.0

                losses = self.loss_manager.generator_loss(
                    images, x_new,
                    new_logits, target_labels, probs,
                    flip_mask,
                    z_orig_list, z_mutated,
                    masks,
                    self.model.mutator.directions,
                    weights=self._get_dynamic_loss_weights(epoch)
                )

                logits_fake_for_g = self.discriminator(x_new)
                g_adv_loss = -logits_fake_for_g.mean()
                losses['adv'] = g_adv_loss

                base_total = losses['total']
                losses['total'] = base_total + self.adv_weight * g_adv_loss
                losses['disc'] = disc_loss_value

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

                embed_log = self._prepare_embedding_log(z_orig_list, z_mutated, step_values)
                self._log_step(epoch, batch_idx, losses, attr_idx, embed_log)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}",
                    'cf': f"{losses['cf'].item():.3f}",
                    'ret': f"{losses['retention'].item():.3f}",
                    'adv': f"{losses['adv'].item():.3f}",
                    'd': f"{losses['disc'].item():.3f}"
                })
            
            self._set_discriminator_grad(True)

            torch.cuda.empty_cache()
        
        # Log epoch summary
        self._log_epoch_summary(epoch, epoch_losses, attr_losses)
        
        return {k: np.mean(v) for k, v in epoch_losses.items()}

    def _get_dynamic_loss_weights(self, epoch):
        base_weights = self.loss_manager.base_weights
        dynamic = base_weights.copy()
        dynamic['cf'] = base_weights['cf'] * 2.0
        if epoch < 10:
            dynamic['retention'] = 0.0
            dynamic['latent_prox'] = 0.0
        return dynamic

    def _log_step(self, epoch, batch_idx, losses, attr_idx, embedding_info):
        """Log per-step metrics"""
        self.writer.add_scalar('Loss/total', losses['total'].item(), self.global_step)
        self.writer.add_scalar('Loss/cf', losses['cf'].item(), self.global_step)
        self.writer.add_scalar('Loss/retention', losses['retention'].item(), self.global_step)
        self.writer.add_scalar('Loss/latent_prox', losses['latent_prox'].item(), self.global_step)
        self.writer.add_scalar('Loss/ortho', losses['ortho'].item(), self.global_step)
        self.writer.add_scalar('Loss/sparse', losses['sparse'].item(), self.global_step)
        self.writer.add_scalar('Loss/adv', losses['adv'].item(), self.global_step)
        self.writer.add_scalar('Loss/disc', losses['disc'].item(), self.global_step)
        
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
            'adv_loss': losses['adv'].item(),
            'disc_loss': losses['disc'].item(),
            'lr': self.optimizer.param_groups[0]['lr'],
            'embed_orig_top': embedding_info['orig'][0],
            'embed_orig_mid': embedding_info['orig'][1],
            'embed_orig_bot': embedding_info['orig'][2],
            'embed_cf_top': embedding_info['cf'][0],
            'embed_cf_mid': embedding_info['cf'][1],
            'embed_cf_bot': embedding_info['cf'][2],
            'step_top': embedding_info['steps'][0],
            'step_mid': embedding_info['steps'][1],
            'step_bot': embedding_info['steps'][2]
        }
        
        self.csv_writer.writerow(row)
        self.csv_file.flush()
    
    def _prepare_embedding_log(self, z_orig_list, z_mutated_list, step_values):
        """Prepare compact embedding summaries and step magnitudes for CSV logging."""
        embed_orig = []
        embed_cf = []
        for z_orig, z_cf in zip(z_orig_list, z_mutated_list):
            orig_vec = z_orig[0].detach().float().mean(dim=(1, 2)).cpu().tolist()
            cf_vec = z_cf[0].detach().float().mean(dim=(1, 2)).cpu().tolist()
            embed_orig.append(json.dumps(orig_vec))
            embed_cf.append(json.dumps(cf_vec))

        steps = []
        for step in step_values:
            step_scalar = float(step[0].detach().float().item())
            steps.append(step_scalar)

        return {
            'orig': embed_orig,
            'cf': embed_cf,
            'steps': steps
        }

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
        for attr_idx in self.active_attr_indices:
            losses = attr_losses.get(attr_idx, [])
            if losses:
                mean = np.mean(losses)
                attr_means[attr_idx] = mean
                attr_name = self.attr_names[attr_idx]
                if mean < 0.5:
                    print(f"  {attr_name}: {mean:.4f}")
        
        # Save per-attribute stats for active attributes only
        attr_stats = {
            self.attr_names[idx]: {
                'mean_loss': float(attr_means.get(idx, -1)),
                'num_updates': len(attr_losses.get(idx, []))
            }
            for idx in self.active_attr_indices
        }
        
        with open(self.log_dir / f'attr_stats_epoch_{epoch}.json', 'w') as f:
            json.dump(attr_stats, f, indent=2)
    
    def validate_and_visualize(self, epoch, loader, num_samples=4):
        """Generate visualizations during validation"""
        self.model.mutator.eval()

        print(f"\n🎨 Generating visualizations for epoch {epoch}...")

        inference_dir = self.vis_dir / "validation"
        inference_dir.mkdir(exist_ok=True)

        vis_count = 0
        with torch.no_grad():
            for batch in loader:
                if vis_count >= num_samples:
                    break

                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    images, labels, filenames = batch
                else:
                    images, labels = batch
                    filenames = None

                images = images.to(self.device)
                labels = labels.to(self.device)

                batch_size = images.size(0)
                for sample_idx in range(batch_size):
                    if vis_count >= num_samples:
                        break

                    img = images[sample_idx:sample_idx + 1]
                    lbl = labels[sample_idx:sample_idx + 1]
                    img_name = None
                    if filenames is not None:
                        try:
                            img_name = filenames[sample_idx]
                        except Exception:
                            img_name = None

                    cf_results = self._generate_counterfactuals(img, lbl, num_cf=3)
                    fig = self.visualizer.create_comparison_grid(
                        img.cpu(),
                        cf_results,
                        lbl.cpu(),
                        self.attr_names,
                        active_indices=self.active_attr_indices,
                        active_names=self.active_attr_names,
                        image_name=img_name
                    )

                    fig.savefig(
                        inference_dir / f'epoch_{epoch:03d}_sample_{vis_count:03d}.png',
                        dpi=150,
                        bbox_inches='tight'
                    )
                    self.writer.add_figure(
                        f'Validation/sample_{vis_count}',
                        fig,
                        epoch
                    )
                    plt.close(fig)
                    vis_count += 1

        self.model.mutator.train()
        print(f"🖼️ Validation samples logged: {vis_count}")

    def _generate_counterfactuals(self, img, labels, num_cf=3):
        results = {'original': img.detach().cpu(), 'cfs': []}

        with torch.no_grad():
            probs = torch.sigmoid(self.model.classifier(img))

        for _ in range(num_cf):
            attr_idx = random.choice(self.active_attr_indices)
            target_labels = labels.clone()
            target_labels[:, attr_idx] = 1 - target_labels[:, attr_idx]

            with torch.enable_grad():
                img_grad = img.clone().detach().requires_grad_(True)
                logits = self.model.classifier(img_grad)
                target_score = logits[:, attr_idx].sum()
                grads = torch.autograd.grad(target_score, img_grad, create_graph=False)[0]
                ig_map = torch.abs(img_grad * grads).sum(dim=1)
                max_vals = ig_map.view(ig_map.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
                ig_map = ig_map / (max_vals + 1e-8)
                ig_map = ig_map.detach()

            with torch.no_grad():
                cam_blob = F.avg_pool2d(ig_map.unsqueeze(1), kernel_size=11, stride=1, padding=5)
                cam_flat = cam_blob.view(cam_blob.size(0), -1)
                cam_max = cam_flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1)
                cam_norm = cam_blob.squeeze(1) / (cam_max + 1e-8)

                cam_binary = (cam_norm > 0.05).float()
                mask_t = F.interpolate(cam_binary.unsqueeze(1), size=(8, 8)).squeeze(1)
                mask_m = F.interpolate(cam_binary.unsqueeze(1), size=(16, 16)).squeeze(1)
                mask_b = F.interpolate(cam_binary.unsqueeze(1), size=(32, 32)).squeeze(1)
                masks = [mask_t, mask_m, mask_b]

                overlay_mask = F.interpolate(
                    cam_norm.unsqueeze(1),
                    size=img.shape[2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1).clamp(0, 1)

                x_cf, _, _ = self.model(
                    img,
                    ig_map,
                    masks,
                    target_labels,
                    probs,
                    attr_idx,
                    hard=True
                )

            results['cfs'].append({
                'image': x_cf.cpu(),
                'attr_idx': attr_idx,
                'attr_name': self.attr_names[attr_idx],
                'ig_map': ig_map.cpu(),
                'cam_map': overlay_mask.cpu()
            })

        return results
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'mutator_state': self.model.mutator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'opt_d_state': self.opt_d.state_dict(),
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

            is_best = epoch_losses['total'] < self.best_loss
            if is_best:
                self.best_loss = epoch_losses['total']
            self.save_checkpoint(epoch, is_best=is_best)

            if loader_val:
                self.validate_and_visualize(epoch, loader_val, num_samples=4)

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
    loader_val = get_loader(cfg, split='val', batch_size=cfg.batch_size, return_filename=True)
    
    trainer = ComprehensiveTrainer(cfg, experiment_name="ceGAN_counterfactual")
    
    trainer.train(
        num_epochs=cfg.num_epochs,
        loader_train=loader_train,
        loader_val=loader_val
    )


if __name__ == '__main__':
    main()
