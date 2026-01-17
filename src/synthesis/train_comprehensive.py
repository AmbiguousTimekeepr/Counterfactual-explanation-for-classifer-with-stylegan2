import torch
import torch.optim as optim
from torch.cuda.amp import autocast
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
from collections import OrderedDict
import itertools
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import cv2

from .config import Config
from .generator import CounterfactualGenerator
from .loss_functions import CounterfactualLossManager
from .dataset import CelebADataset, get_loader, get_attribute_names
from .visualizer import CounterfactualVisualizer
from .discriminator_manager import PatchGANDiscriminator
from ..classifiers.integrated_gradients import integrated_gradients_batch

class ComprehensiveTrainer:
    def __init__(self, cfg, experiment_name="counterfactual_training", device=None):
        """
        Initialize comprehensive trainer with logging and visualization.
        
        Args:
            cfg: Config object
            experiment_name: Name for this experiment (used for directories)
        """
        self.cfg = cfg
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.experiment_name = experiment_name
        
        # Setup directories
        self.setup_directories()
        
        # Initialize models
        self.init_models()

        # XAI helpers (Integrated Gradients + Sharpened mask)
        self.mask_threshold = getattr(self.cfg, 'cam_threshold', 0.35)
        self.ig_steps = getattr(self.cfg, 'ig_steps', 16)
        self.use_ig = getattr(self.cfg, 'use_ig', True)
        self.saliency_cache = OrderedDict()
        self.saliency_cache_size = getattr(self.cfg, 'saliency_cache_size', 256)
        
        # Attribute name mapping (restricted CelebA subset)
        self.attr_names = self._get_attribute_names()
        self._initialize_active_attributes()
        self.attr_sample_size = min(
            len(self.active_attr_indices),
            getattr(self.cfg, 'max_active_attributes_per_epoch', len(self.active_attr_indices))
        )
        if self.attr_sample_size <= 0:
            raise ValueError("max_active_attributes_per_epoch must be >= 1")

        # Setup logging
        self.setup_logging()
        
        # Visualization
        self.visualizer = CounterfactualVisualizer(self.device)
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        
        self.decoder_pretrained = False
    
    def _make_cache_key(self, filename, attr_idx, target_class):
        """Create a unique cache key for saliency computation."""
        if filename is None:
            return None
        filename_str = str(filename) if not isinstance(filename, str) else filename
        filename_safe = self._sanitize_filename(Path(filename_str).stem)
        return f"{filename_safe}_{attr_idx}_{int(target_class)}"

    def _cache_get(self, cache_key):
        """Retrieve cached saliency signals."""
        if cache_key is None:
            return None
        return self.saliency_cache.get(cache_key, None)

    def _cache_set(self, cache_key, value):
        """Store saliency signals in cache with LRU eviction."""
        if cache_key is None:
            return
        self.saliency_cache[cache_key] = value
        # LRU eviction: nếu cache quá lớn, xóa oldest entry
        if len(self.saliency_cache) > self.saliency_cache_size:
            self.saliency_cache.popitem(last=False)
    
    def setup_directories(self):
        """Create output directories"""
        timestamp = datetime.now().strftime("%Y%m%d")
        suffix = "_no_ig" if not self.cfg.use_ig else ""
        self.exp_dir = Path(f"outputs/synth_network/CF_generator/{self.experiment_name}_{timestamp}{suffix}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.log_dir = self.exp_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        self.vis_dir = self.exp_dir / "visualizations"
        self.vis_dir.mkdir(exist_ok=True)
        
        print(f"Experiment directory: {self.exp_dir}")
    
    def init_models(self):
        """Initialize all models"""
        self.model = CounterfactualGenerator(
            self.cfg,
            vqvae_path=self.cfg.vqvae_checkpoint,
            classifier_path=self.cfg.classifier_checkpoint,
            device=self.device
        )
        
        self.loss_manager = CounterfactualLossManager(device=self.device)
        
        # Enable fine-tuning for decoder
        for p in self.model.decoder.parameters():
            p.requires_grad = True

        # Optimizer: Mutator (full LR) + Decoder (reduced LR)
        self.optimizer = optim.AdamW(
            [
                {'params': self.model.mutator.parameters(), 'lr': self.cfg.learning_rate},
                {'params': self.model.decoder.parameters(), 'lr': self.cfg.learning_rate * 0.1}
            ],
            weight_decay=self.cfg.weight_decay
        )

        # Adversarial discriminator (realism regularizer)
        self.discriminator = PatchGANDiscriminator().to(self.device)
        self.discriminator.train()
        self.opt_d = optim.AdamW(
            self.discriminator.parameters(),
            lr=self.cfg.learning_rate * 0.1,
            betas=(0.5, 0.999)
        )
        self.adv_weight = getattr(self.cfg, 'adv_weight', 0.5)
        print(f"Adversarial loss weight: {self.adv_weight}")
        
        use_amp = bool(getattr(self.cfg, 'use_amp', False)) and torch.cuda.is_available()
        try:
            # Prefer newer torch.amp API when available
            self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            # Backward compatibility for older torch versions
            self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.use_amp = use_amp
        self.decoder_pretrained = False
        sharpened_path = getattr(self.cfg, 'sharpened_decoder_path', '')
        decoder_ckpt_path = getattr(self.cfg, 'decoder_checkpoint_path', '')

        chosen_path = None
        chosen_label = None

        if sharpened_path:
            sharp_path = Path(sharpened_path)
            if sharp_path.is_file():
                chosen_path = sharp_path
                chosen_label = "sharpened decoder checkpoint"
            else:
                print(f"Sharpened decoder checkpoint not found at {sharp_path}")

        if chosen_path is None and decoder_ckpt_path:
            ckpt_path = Path(decoder_ckpt_path)
            if ckpt_path.is_file():
                chosen_path = ckpt_path
                chosen_label = "decoder checkpoint"
            else:
                print(f"Decoder checkpoint not found at {ckpt_path}")

        if chosen_path is not None:
            self.model.load_decoder_checkpoint(chosen_path, map_location=self.device)
            self.decoder_pretrained = True
            print(f"Loaded decoder weights from {chosen_path} ({chosen_label})")
        else:
            if getattr(self.cfg, 'decoder_pretrain_epochs', 0) > 0:
                print("No decoder checkpoint supplied; decoder will be trained from scratch.")
            else:
                print("No decoder checkpoint supplied; starting with random decoder weights.")
    
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
        print(f"Active synthesis attributes ({len(self.active_attr_indices)}): {', '.join(self.active_attr_names)}")

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
            'ortho_loss', 'sparse_loss', 'perc_loss', 'adv_loss', 'disc_loss', 'lr',
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

    def _compute_saliency_for_sample(self, image_tensor, attr_idx, target_class, cache_key=None):
        """Compute IG map and sharpened retention masks for a single sample."""
        if not self.use_ig:
            h, w = image_tensor.shape[-2:]
            ig_map = torch.ones(h, w, device=image_tensor.device)
            mask_t = torch.ones(8, 8, device=image_tensor.device)
            mask_m = torch.ones(16, 16, device=image_tensor.device)
            mask_b = torch.ones(32, 32, device=image_tensor.device)
            cam_soft = torch.zeros(h, w, device=image_tensor.device)
            return ig_map, [mask_t, mask_m, mask_b], cam_soft

        cached = self._cache_get(cache_key)
        if cached is not None:
            ig_cached, masks_cached, cam_cached = cached
            ig_map = ig_cached.to(self.device)
            masks = [mask.to(self.device) for mask in masks_cached]
            cam_soft = cam_cached.to(self.device)
            return ig_map, masks, cam_soft

        # Use integrated_gradients_batch instead
        ig_attr = integrated_gradients_batch(
            model=self.model.classifier,
            input_batch=image_tensor,
            attribute_idx=attr_idx,
            target_classes=torch.tensor([int(target_class)], device=self.device),
            steps=self.ig_steps,
            device=self.device
        )

        raw_saliency = torch.abs(ig_attr).sum(dim=1, keepdim=True)
        flat_max = raw_saliency.view(raw_saliency.size(0), -1).amax(dim=1, keepdim=True).view(-1, 1, 1, 1)
        ig_map = (raw_saliency / (flat_max + 1e-8)).squeeze(0).squeeze(0).detach()

        smooth = F.avg_pool2d(raw_saliency, kernel_size=11, stride=1, padding=5)
        norm = smooth / (smooth.amax(dim=[1, 2, 3], keepdim=True) + 1e-8)
        norm = norm.pow(1.5)
        norm_map = F.interpolate(
            norm,
            size=image_tensor.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)
        norm_map = norm_map.clamp(0.0, 1.0)

        binary_mask = (norm_map > self.mask_threshold).float()
        mask_t = F.interpolate(
            binary_mask.unsqueeze(0).unsqueeze(0),
            size=(8, 8),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)
        mask_m = F.interpolate(
            binary_mask.unsqueeze(0).unsqueeze(0),
            size=(16, 16),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)
        mask_b = F.interpolate(
            binary_mask.unsqueeze(0).unsqueeze(0),
            size=(32, 32),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)

        ig_to_store = ig_map.detach().cpu()
        masks_to_store = [mask_t.detach().cpu(), mask_m.detach().cpu(), mask_b.detach().cpu()]
        cam_to_store = norm_map.detach().cpu()
        self._cache_set(cache_key, (ig_to_store, masks_to_store, cam_to_store))

        return ig_map, [mask_t.detach(), mask_m.detach(), mask_b.detach()], norm_map.detach()

    def _prepare_saliency_signals(self, images, target_labels, attr_idx, filenames=None):
        """Compute IG maps and sharpened retention masks for a batch (vectorized)."""
        batch_size = images.size(0)
        device = images.device

        if not self.use_ig:
            h, w = images.shape[-2:]
            ig_maps = torch.ones(batch_size, h, w, device=device)
            mask_t = torch.ones(batch_size, 8, 8, device=device)
            mask_m = torch.ones(batch_size, 16, 16, device=device)
            mask_b = torch.ones(batch_size, 32, 32, device=device)
            return ig_maps, [mask_t, mask_m, mask_b]
        
        # Check cache for all samples first
        cache_keys = []
        cached_indices = []
        uncached_indices = []
        
        target_classes = target_labels[:, attr_idx]  # [B]
        
        if filenames is not None:
            for i in range(batch_size):
                cache_key = self._make_cache_key(filenames[i], attr_idx, int(target_classes[i].item()))
                cache_keys.append(cache_key)
                cached = self._cache_get(cache_key)
                if cached is not None:
                    cached_indices.append((i, cached))
                else:
                    uncached_indices.append(i)
        else:
            cache_keys = [None] * batch_size
            uncached_indices = list(range(batch_size))
        
        # Initialize output tensors
        ig_maps = torch.zeros(batch_size, images.shape[-2], images.shape[-1], device=device)
        mask_t = torch.zeros(batch_size, 8, 8, device=device)
        mask_m = torch.zeros(batch_size, 16, 16, device=device)
        mask_b = torch.zeros(batch_size, 32, 32, device=device)
        
        # Fill in cached results
        for idx, (ig_cached, masks_cached, _) in cached_indices:
            ig_maps[idx] = ig_cached.to(device)
            mask_t[idx] = masks_cached[0].to(device)
            mask_m[idx] = masks_cached[1].to(device)
            mask_b[idx] = masks_cached[2].to(device)
        
        # Process uncached samples in batch
        if uncached_indices:
            uncached_images = images[uncached_indices]  # [U, C, H, W]
            uncached_targets = target_classes[uncached_indices]  # [U]
            
            # Batch integrated gradients
            ig_batch = integrated_gradients_batch(
                model=self.model.classifier,
                input_batch=uncached_images,
                attribute_idx=attr_idx,
                target_classes=uncached_targets,
                steps=self.ig_steps,
                device=device
            )  # [U, C, H, W]
            
            # Compute IG maps (vectorized)
            raw_saliency = torch.abs(ig_batch).sum(dim=1, keepdim=True)  # [U, 1, H, W]
            flat_max = raw_saliency.view(raw_saliency.size(0), -1).amax(dim=1, keepdim=True)
            flat_max = flat_max.view(-1, 1, 1, 1)
            ig_normalized = (raw_saliency / (flat_max + 1e-8)).squeeze(1)  # [U, H, W]
            
            # Compute sharpened masks (vectorized)
            smooth = F.avg_pool2d(raw_saliency, kernel_size=11, stride=1, padding=5)  # [U, 1, H, W]
            norm = smooth / (smooth.amax(dim=[1, 2, 3], keepdim=True) + 1e-8)
            norm = norm.pow(1.5)
            norm_map = F.interpolate(
                norm,
                size=uncached_images.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # [U, H, W]
            norm_map = norm_map.clamp(0.0, 1.0)
            
            # Binary masks
            binary_mask = (norm_map > self.mask_threshold).float()  # [U, H, W]
            
            # Resize to different scales (vectorized)
            binary_4d = binary_mask.unsqueeze(1)  # [U, 1, H, W]
            masks_t_batch = F.interpolate(binary_4d, size=(8, 8), mode='bilinear', align_corners=False).squeeze(1)
            masks_m_batch = F.interpolate(binary_4d, size=(16, 16), mode='bilinear', align_corners=False).squeeze(1)
            masks_b_batch = F.interpolate(binary_4d, size=(32, 32), mode='bilinear', align_corners=False).squeeze(1)
            
            # Fill in results and update cache
            for local_idx, global_idx in enumerate(uncached_indices):
                ig_maps[global_idx] = ig_normalized[local_idx].detach()
                mask_t[global_idx] = masks_t_batch[local_idx].detach()
                mask_m[global_idx] = masks_m_batch[local_idx].detach()
                mask_b[global_idx] = masks_b_batch[local_idx].detach()
                
                # Cache the result
                if cache_keys[global_idx] is not None:
                    ig_to_store = ig_normalized[local_idx].detach().cpu()
                    masks_to_store = [
                        masks_t_batch[local_idx].detach().cpu(),
                        masks_m_batch[local_idx].detach().cpu(),
                        masks_b_batch[local_idx].detach().cpu()
                    ]
                    cam_to_store = norm_map[local_idx].detach().cpu()
                    self._cache_set(cache_keys[global_idx], (ig_to_store, masks_to_store, cam_to_store))
        
        masks_batch = [mask_t, mask_m, mask_b]
        return ig_maps, masks_batch

    def _sanitize_filename(self, name):
        """Sanitize filename component by keeping alphanumerics, dash, underscore."""
        return ''.join(ch if ch.isalnum() or ch in ('_', '-') else '_' for ch in name)

    def _run_epoch_inference(self, epoch, loader_test, num_images=4):
        """Run inference visualizations on test split and save overlays per attribute."""
        if loader_test is None:
            return

        was_mutator_training = self.model.mutator.training
        was_decoder_training = self.model.decoder.training
        self.model.mutator.eval()
        self.model.decoder.eval()

        inference_root = self.vis_dir / "inferences"
        epoch_dir = inference_root / f"epoch_{epoch + 1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        images_logged = 0
        for batch in loader_test:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, labels, filenames = batch
            else:
                images, labels = batch
                filenames = None
            if filenames is not None and not isinstance(filenames, (list, tuple)):
                filenames = list(filenames)
            if filenames is not None and len(filenames) != images.size(0):
                filenames = [None] * images.size(0)
            if filenames is not None and len(filenames) != images.size(0):
                filenames = [None] * images.size(0)
            if filenames is not None and not isinstance(filenames, (list, tuple)):
                filenames = list(filenames)

            images = images.to(self.device)
            labels = labels.to(self.device)

            batch_size = images.size(0)
            for sample_idx in range(batch_size):
                if images_logged >= num_images:
                    break

                img = images[sample_idx:sample_idx + 1]
                lbl = labels[sample_idx:sample_idx + 1]

                if filenames is not None:
                    base_name = Path(filenames[sample_idx]).stem
                else:
                    base_name = f"image{images_logged + 1}"
                base_name = self._sanitize_filename(base_name)

                with torch.no_grad():
                    probs = torch.sigmoid(self.model.classifier(img))

                base_np = self._tensor_to_image(img)

                for attr_idx in self.active_attr_indices:
                    attr_name = self.attr_names[attr_idx]
                    target_labels = lbl.clone()
                    target_labels[:, attr_idx] = 1 - target_labels[:, attr_idx]
                    target_class = target_labels[0, attr_idx].item()

                    cache_source = filenames[sample_idx] if filenames is not None else base_name
                    cache_key = self._make_cache_key(cache_source, attr_idx, target_class)
                    ig_map, mask_levels, cam_soft = self._compute_saliency_for_sample(
                        img.detach(),
                        attr_idx,
                        target_class,
                        cache_key=cache_key
                    )

                    ig_batch = ig_map.unsqueeze(0)
                    masks = [mask.unsqueeze(0) for mask in mask_levels]

                    with torch.no_grad():
                        x_cf, _, _ = self.model(
                            img,
                            ig_batch,
                            masks,
                            target_labels,
                            probs,
                            attr_idx,
                            hard=True
                        )

                    ig_overlay = self._create_overlay(base_np, ig_map)
                    cam_overlay = self._create_overlay(base_np, cam_soft)
                    cf_np = self._tensor_to_image(x_cf)

                    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                    axes[0].imshow(base_np)
                    axes[0].set_title("Original")
                    axes[0].axis('off')

                    axes[1].imshow(ig_overlay)
                    axes[1].set_title("IG Overlay")
                    axes[1].axis('off')

                    axes[2].imshow(cam_overlay)
                    axes[2].set_title("Sharpened CAM Overlay")
                    axes[2].axis('off')

                    axes[3].imshow(cf_np)
                    axes[3].set_title("Counterfactual")
                    axes[3].axis('off')

                    fig.suptitle(f"{base_name} - {attr_name}", fontsize=12)
                    fig.tight_layout(rect=[0, 0, 1, 0.95])

                    output_path = epoch_dir / f"{base_name}_{attr_name}.png"
                    fig.savefig(output_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)

                images_logged += 1

                if images_logged >= num_images:
                    break

            if images_logged >= num_images:
                break

        if was_mutator_training:
            self.model.mutator.train()
        if was_decoder_training:
            self.model.decoder.train()

    def _set_discriminator_grad(self, requires_grad: bool):
        """Toggle gradient computation for discriminator parameters."""
        for param in self.discriminator.parameters():
            param.requires_grad_(requires_grad)
    
    def train_epoch(self, epoch, loader):
        """Train for one epoch with comprehensive logging"""
        # Reset saliency/mask cache each epoch to avoid reusing stale IG maps
        self.saliency_cache.clear()
        self.model.mutator.train()
        self.model.decoder.train()
        self.discriminator.train()
        
        epoch_losses = {
            'total': [], 'cf': [], 'retention': [], 'latent_prox': [],
            'ortho': [], 'sparse': [], 'perc': [], 'adv': [], 'disc': []
        }
        attr_losses = {i: [] for i in self.active_attr_indices}
        
        pbar = tqdm(loader, desc=f"Epoch {epoch} / {self.cfg.num_epochs}")
        accumulation_steps = self.cfg.accumulation_steps
        
        attr_cycle = itertools.cycle(random.sample(self.active_attr_indices, self.attr_sample_size))
        d_clip = getattr(self.cfg, 'd_grad_clip', 0.0) or 0.0
        g_clip = getattr(self.cfg, 'grad_clip', 0.0) or 0.0
        r1_gamma = getattr(self.cfg, 'adv_r1_gamma', 0.0) or 0.0

        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, labels, filenames = batch
            else:
                images, labels = batch
                filenames = None
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # --- Random attribute selection ---
            attr_idx = next(attr_cycle)
            target_labels = labels.clone()
            target_labels[:, attr_idx] = 1 - target_labels[:, attr_idx]
            
            # ====================================================
            # 1. XAI Extraction (IG + Sharpened Mask)
            # ====================================================
            filenames_batch = list(filenames) if filenames is not None else [None] * images.size(0)
            ig_map, masks = self._prepare_saliency_signals(
                images.detach(),
                target_labels,
                attr_idx,
                filenames=filenames_batch
            )

            with torch.no_grad():
                logits = self.model.classifier(images)
                probs = torch.sigmoid(logits)

            with torch.no_grad():
                z_orig_list = self.model.vqvae.get_codes(images)

            use_hard = (epoch >= 5)

            # ====================================================
            # PHASE 1: Train Discriminator (Realism enforcement)
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

            real_imgs = images.detach().requires_grad_(r1_gamma > 0)
            logits_real = self.discriminator(real_imgs)
            d_loss_real = F.relu(1.0 - logits_real).mean()

            logits_fake = self.discriminator(x_fake_detached)
            d_loss_fake = F.relu(1.0 + logits_fake).mean()

            d_loss = d_loss_real + d_loss_fake

            if r1_gamma > 0:
                grad_real = torch.autograd.grad(outputs=logits_real.sum(), inputs=real_imgs, create_graph=True)[0]
                grad_penalty = grad_real.pow(2).view(grad_real.size(0), -1).sum(dim=1).mean()
                d_loss = d_loss + (r1_gamma / 2.0) * grad_penalty

            d_loss.backward()
            if d_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), d_clip)
            self.opt_d.step()
            disc_loss_value = d_loss.detach()

            # ====================================================
            # PHASE 2: Train Generator (Mutator + Decoder)
            # ====================================================
            self._set_discriminator_grad(False)

            # ====================================================
            # 2. Forward Pass (Curriculum Learning)
            # ====================================================
            with autocast(enabled=self.use_amp):
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
                # Unscale before gradient clipping
                self.scaler.unscale_(self.optimizer)
                if g_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), g_clip)
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
                    'total': f"{losses['total'].item():.4f}",
                    'cf': f"{losses['cf'].item():.3f}",
                    'ret': f"{losses['retention'].item():.3f}",
                    'lat': f"{losses['latent_prox'].item():.3f}",
                    'ortho': f"{losses['ortho'].item():.3f}",
                    'sparse': f"{losses['sparse'].item():.4f}",
                    'perc': f"{losses['perc'].item():.3f}",
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
        self.writer.add_scalar('Loss/perc', losses['perc'].item(), self.global_step)
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
            'perc_loss': losses['perc'].item(),
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
        """Visualize original, saliency overlay, and counterfactual for a few samples each epoch."""
        # Ensure validation uses fresh saliency computations
        self.saliency_cache.clear()
        self.model.mutator.eval()
        inference_dir = self.vis_dir / "validation_simple"
        inference_dir.mkdir(exist_ok=True)

        logged = 0
        for batch in loader:
            if logged >= num_samples:
                break

            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, labels, filenames = batch
            else:
                images, labels = batch
                filenames = None

            images = images.to(self.device)
            labels = labels.to(self.device)

            for i in range(images.size(0)):
                if logged >= num_samples:
                    break

                img = images[i:i+1]
                lbl = labels[i:i+1]
                name = filenames[i] if filenames is not None else f"sample_{logged:03d}"
                sanitized = self._sanitize_filename(str(name))
                attr_list = [self.attr_names.index(attr_name) for attr_name in self.active_attr_names]
                # attr_list = [self.attr_names.index('Mustache'),
                # ]
                attr_idx = attr_list[logged % len(attr_list)]
                target_labels = lbl.clone()
                target_labels[:, attr_idx] = 1 - target_labels[:, attr_idx]
                target_class = target_labels[0, attr_idx].item()
                cache_key = self._make_cache_key(name, attr_idx, target_class)

                # Saliency + masks
                ig_map, mask_levels, cam_soft = self._compute_saliency_for_sample(
                    img.detach(), attr_idx, target_class, cache_key=cache_key
                )
                ig_batch = ig_map.unsqueeze(0)
                masks = [m.unsqueeze(0) for m in mask_levels]

                with torch.no_grad():
                    probs = torch.sigmoid(self.model.classifier(img))
                    x_cf, _, _ = self.model(
                        img, ig_batch, masks, target_labels, probs, attr_idx, hard=True
                    )

                # To numpy
                def to_np(x):
                    x = x.detach().cpu().clamp(0, 1)
                    return x[0].permute(1, 2, 0).numpy()

                orig_np = to_np((img + 1) * 0.5)  # nếu training ở [-1,1]; điều chỉnh nếu 0-1
                cf_np = to_np((x_cf + 1) * 0.5)

                # Create overlay
                def _create_overlay(base_img, saliency_map, alpha=0.4):
                    """Create overlay image with saliency heatmap."""
                    heatmap = cv2.applyColorMap(
                        (saliency_map.detach().cpu().numpy() * 255).astype(np.uint8),
                        cv2.COLORMAP_JET
                    )
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
                    overlay = alpha * heatmap + (1 - alpha) * base_img
                    return np.clip(overlay, 0, 1)

                overlay_np = _create_overlay(orig_np, cam_soft)

                fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                axes[0].imshow(orig_np)
                axes[0].set_title("Original")
                axes[0].axis("off")

                axes[1].imshow(overlay_np)
                axes[1].set_title(f"Saliency Overlay ({self.attr_names[attr_idx]})")
                axes[1].axis("off")

                axes[2].imshow(cf_np)
                axes[2].set_title(f"CE (flip {self.attr_names[attr_idx]})")
                axes[2].axis("off")

                fig.suptitle(f"Epoch {epoch} - {sanitized}", fontsize=10)
                fig.tight_layout()

                out_path = inference_dir / f"epoch_{epoch:03d}_{sanitized}.png"
                fig.savefig(out_path, dpi=120, bbox_inches="tight")
                self.writer.add_figure(f"ValSimple/{sanitized}", fig, epoch)
                plt.close(fig)

                # Lưu ảnh gốc và cf riêng
                orig_save_path = inference_dir / f"orig_epoch_{epoch:03d}_sample_{logged:03d}.png"
                cf_save_path = inference_dir / f"cf_epoch_{epoch:03d}_sample_{logged:03d}.png"
                plt.imsave(orig_save_path, orig_np)
                plt.imsave(cf_save_path, cf_np)

                logged += 1

            self.model.mutator.train()
            print(f"Simple validation samples logged: {logged}")

    def _generate_counterfactuals(self, img, labels, num_cf=3, filename=None):
        results = {'original': img.detach().cpu(), 'cfs': []}

        with torch.no_grad():
            probs = torch.sigmoid(self.model.classifier(img))

        for _ in range(num_cf):
            attr_idx = random.choice(self.active_attr_indices)
            target_labels = labels.clone()
            target_labels[:, attr_idx] = 1 - target_labels[:, attr_idx]

            target_class = target_labels[0, attr_idx].item()
            cache_key = self._make_cache_key(filename, attr_idx, target_class)
            ig_single, mask_levels, cam_soft = self._compute_saliency_for_sample(
                img.detach(),
                attr_idx,
                target_class,
                cache_key=cache_key
            )

            ig_map = ig_single.unsqueeze(0)
            masks = [mask.unsqueeze(0) for mask in mask_levels]

            overlay_mask = cam_soft.clamp(0, 1)

            with torch.no_grad():
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
            'decoder_state': self.model.decoder.state_dict(),
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
            print(f"Best checkpoint saved: {best_path}")
    
    def recon_training(self, loader):
        """Pre-train the decoder with reconstruction loss"""
        decoder_epochs = getattr(self.cfg, 'decoder_pretrain_epochs', 0)
        if decoder_epochs <= 0:
            print("decoder_pretrain_epochs <= 0 - skipping decoder pre-training.")
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

        print(f"Starting StyleGAN decoder pre-training for {decoder_epochs} epoch(s)")
        self.model.decoder.train()
        optimizer = optim.Adam(self.model.decoder.parameters(), lr=decoder_lr, betas=decoder_betas)
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
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

                with autocast(enabled=self.use_amp):
                    recon = self.model.decoder(z_list)
                    loss_l1 = l1_loss(recon, images)
                    loss_lpips = lpips_loss(recon, images).mean()
                    loss = loss_l1 + loss_lpips

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                pbar.set_postfix({'L1': loss_l1.item(), 'LPIPS': loss_lpips.item()})

            print("Generating reconstruction preview...")
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
            print(f"Saved decoder preview: {preview_path}")
            self.model.decoder.train()

            if epoch % 5 == 0:
                torch.save(self.model.decoder.state_dict(), primary_ckpt)
                torch.save(self.model.decoder.state_dict(), latest_ckpt)
                print(f"Saved decoder checkpoints to {primary_ckpt} and {latest_ckpt}")

        self.model.decoder.eval()
        self.decoder_pretrained = True
        self.cfg.decoder_checkpoint_path = str(latest_ckpt)
        print("Decoder pre-training complete")

    def train(self, num_epochs, loader_train, loader_val=None, loader_test=None):
        """Full training loop"""
        print("Starting Comprehensive Counterfactual Training...")
        print(f"Logging to: {self.log_dir}")
        print(f"Checkpoints to: {self.checkpoint_dir}")
        print(f"Visualizations to: {self.vis_dir}")

        for epoch in range(num_epochs):
            epoch_losses = self.train_epoch(epoch, loader_train)

            is_best = epoch_losses['total'] < self.best_loss
            if is_best:
                self.best_loss = epoch_losses['total']
            self.save_checkpoint(epoch, is_best=is_best)

            if loader_val is not None:
                self.validate_and_visualize(epoch, loader_val, num_samples=4)

            if loader_test is not None:
                self._run_epoch_inference(epoch, loader_test, num_images=4)

            if hasattr(self.cfg, 'lr_schedule'):
                schedule = self.cfg.lr_schedule
                if epoch in schedule:
                    new_lr = schedule[epoch]
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"Learning rate updated to {new_lr}")


def main():
    """Main training entry point"""
    cfg = Config()
    
    loader_train = get_loader(cfg, split='train', batch_size=cfg.batch_size, return_filename=True)
    loader_val = get_loader(cfg, split='val', batch_size=cfg.batch_size, return_filename=True)
    loader_test = get_loader(cfg, split='test', batch_size=4, return_filename=True)
    
    trainer = ComprehensiveTrainer(cfg, experiment_name="ceGAN_counterfactual")
    
    trainer.train(
        num_epochs=cfg.num_epochs,
        loader_train=loader_train,
        loader_val=loader_val,
        loader_test=loader_test
    )


if __name__ == '__main__':
    main()
