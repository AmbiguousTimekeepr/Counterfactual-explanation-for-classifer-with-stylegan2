"""Latent Mutator training script.

This standalone trainer stitches together the latent editor, the HVQ-VAE encoder/
decoder stack, and a pretrained attribute classifier.

Key features:
- Aggressive target flip loss
- CAM as spatial mask, IG as strength modulator
- Continuous mutation (no re-quant during training)
- Perceptual + margin + orthogonality losses
- Multi-attribute per batch support
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, List, TextIO
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from torchvision import transforms as T
from torchvision import models
from torchvision.utils import save_image

from tqdm.auto import tqdm
from .module import LatentMutator
from ..unsupervised_latentspace.model import HVQVAE_3Level
from ..unsupervised_latentspace.config import Config as VQConfig
from ..classifiers.dataset import CelebADataset, load_attribute_dataframe
from ..classifiers.model import SquarePadResize
from ..classifiers.resnet18_CBAM.model import ResNet18_CBAM
from ..classifiers.resnet18_CBAM.integrated_gradients import integrated_gradients
from ..classifiers.resnet18_CBAM.gradcam import GradCAMPlusPlus


class VGGPerceptualLoss(nn.Module):
    """VGG16-based perceptual loss (frozen)."""
    def __init__(self, device: str = "cuda"):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16].to(device)  # up to relu3_3 or relu4_1
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.eval()
        self.mse = nn.MSELoss()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.mse(self.vgg(x), self.vgg(y))


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


@dataclass
class MutatorTrainingConfig:
    data_root: str
    attributes_csv: str
    attribute_names: Sequence[str]
    hvqvae_checkpoint: str
    classifier_checkpoint: str
    output_dir: str
    device: str = "cuda"
    batch_size: int = 8
    num_workers: int = 6
    hvqvae_image_size: int = 128
    classifier_image_size: int = 224
    learning_rate: float = 2e-4
    total_steps: int = 10000
    accumulation_steps: int = 8
    log_interval: int = 50
    sample_interval: int = 500
    checkpoint_interval: int = 2000
    ig_steps: int = 100
    amp: bool = True
    seed: Optional[int] = 42
    save_debug_samples: int = 8
    # Live logging / tensorboard
    live_logging: bool = True
    tensorboard: bool = False
    log_dir: Optional[str] = None


class MutatorTrainer:
    def __init__(self, cfg: MutatorTrainingConfig):
        self.cfg = cfg
        self.attribute_cycle = list(range(len(self.cfg.attribute_names)))
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            torch.cuda.manual_seed_all(cfg.seed)

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "samples").mkdir(exist_ok=True)

        self._load_models()
        self._setup_perceptual()
        self._setup_data()
        self.optimizer = torch.optim.Adam(self.mutator.parameters(), lr=cfg.learning_rate)
        self.scaler = GradScaler(enabled=cfg.amp)
        self.global_step = 0
        # Plain-text logging (optional). The `tensorboard` cfg flag now
        # enables writing a simple CSV-like log file to `log_dir`.
        self.log_file: Optional[TextIO] = None
        if getattr(self.cfg, "tensorboard", False):
            log_dir = Path(self.cfg.log_dir) if self.cfg.log_dir else (self.output_dir / "logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "training.log"
            # open in append mode so multiple runs accumulate
            self.log_file = open(log_path, "a", encoding="utf-8")
            # header with timestamp for run separation
            self.log_file.write(f"=== Mutator training log started: {datetime.now().isoformat()} ===\n")
            # CSV header for scalars (only if file was empty would be ideal, but keep simple)
            self.log_file.write("step,total,target,margin,l1,perceptual,latent_prox,ortho\n")
            self.log_file.flush()

    def _load_models(self):
        # HVQ-VAE (frozen) - create minimal config
        from ..unsupervised_latentspace.config import Config as VQConfig
        vq_cfg = VQConfig()
        vq_cfg.device = str(self.device)
        # Optionally set other params if needed, e.g.:
        # vq_cfg.image_size = self.cfg.hvqvae_image_size

        self.hvqvae = HVQVAE_3Level(vq_cfg).to(self.device)

        # Load checkpoint
        state = torch.load(self.cfg.hvqvae_checkpoint, map_location=self.device)
        model_state = state.get("model", state.get("state_dict", state))
        self.hvqvae.load_state_dict(model_state, strict=False)
        self.hvqvae.eval()
        for p in self.hvqvae.parameters():
            p.requires_grad = False

        # Classifier (frozen)
        num_attr = len(self.cfg.attribute_names)
        self.classifier = ResNet18_CBAM(num_classes=num_attr).to(self.device)
        state = torch.load(self.cfg.classifier_checkpoint, map_location=self.device)
        classifier_state = state.get("model", state.get("state_dict", state))
        self.classifier.load_state_dict(classifier_state, strict=False)
        self.classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad = False

        # GradCAM++ extractor
        self.cam_extractor = GradCAMPlusPlus(self.classifier, self.classifier.layer4)

        # Mutator
        embed_dim = self.hvqvae.quant_conv_t.out_channels  # usually 64
        self.mutator = LatentMutator(num_attributes=num_attr, embed_dim=embed_dim).to(self.device)

    def _setup_perceptual(self):
        self.perceptual_fn = VGGPerceptualLoss(device=self.device).to(self.device)
        self.perceptual_fn.eval()
        for p in self.perceptual_fn.parameters():
            p.requires_grad = False

    def _setup_data(self):
        df = load_attribute_dataframe(self.cfg.attributes_csv, self.cfg.attribute_names)

        base_dataset = CelebADataset(
            df=df,
            root_dir=self.cfg.data_root,
            transform=None,
        )

        hvq_transform = T.Compose([
            SquarePadResize(self.cfg.hvqvae_image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

        classifier_transform = T.Compose([
            SquarePadResize(self.cfg.classifier_image_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
        ])

        class DualViewDataset(torch.utils.data.Dataset):
            def __init__(self, base, hvq_t, cls_t):
                self.base = base
                self.hvq_t = hvq_t
                self.cls_t = cls_t
            def __len__(self): return len(self.base)
            def __getitem__(self, i):
                img, label = self.base[i]
                return self.hvq_t(img), self.cls_t(img), label.float()

        self.dataset = DualViewDataset(base_dataset, hvq_transform, classifier_transform)
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def _get_maps(self, classifier_inputs: Tensor, targets: Tensor, active_attr_idxs) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Compute CAM and IG maps for a set of attribute indices.
        If multiple attributes are requested, maps are combined via elementwise max.

        active_attr_idxs: int or Sequence[int]
        Returns: cam_resized, ig_resized where each is a list over latent levels
        """
        B = classifier_inputs.shape[0]
        if isinstance(active_attr_idxs, int):
            active_attr_idxs = [int(active_attr_idxs)]

        cam_maps_per_attr = []
        ig_maps_per_attr = []

        classifier_inputs = classifier_inputs.detach().clone()
        classifier_inputs.requires_grad_(True)

        self.cam_extractor.register_hooks()
        try:
            for attr_idx in active_attr_idxs:
                cam_maps = []
                ig_maps = []
                for i in range(B):
                    img = classifier_inputs[i:i+1]
                    target_label = targets[i, attr_idx]

                    logits = self.classifier(img)
                    score = logits[:, attr_idx]

                    self.classifier.zero_grad()
                    if img.grad is not None:
                        img.grad.zero_()
                    score.backward()

                    cam_np = self.cam_extractor.generate_cam(
                        img,
                        attribute_idx=attr_idx,
                        target_class=int(target_label.item()),
                    )
                    cam = torch.from_numpy(cam_np).unsqueeze(0).unsqueeze(0).to(self.device)  # [1,1,H,W]
                    cam_maps.append(cam)

                    ig = integrated_gradients(
                        model=self.classifier,
                        input_image=img,
                        attribute_idx=attr_idx,
                        target_class=int(target_label.item()),
                        steps=self.cfg.ig_steps,
                        device=self.device,
                    )
                    ig = ig.abs().sum(dim=1, keepdim=True)  # [1,1,H,W]
                    ig_maps.append(ig)

                cam_maps_per_attr.append(torch.cat(cam_maps, dim=0))
                ig_maps_per_attr.append(torch.cat(ig_maps, dim=0))
        finally:
            self.cam_extractor.remove_hooks()

        # Combine attribute maps by taking element-wise max across attributes
        cam_maps_combined = torch.stack(cam_maps_per_attr, dim=0).amax(dim=0)
        ig_maps_combined = torch.stack(ig_maps_per_attr, dim=0).amax(dim=0)

        # Resize to hierarchical latent sizes (adjust if your sizes differ)
        latent_sizes = [(8, 8), (16, 16), (32, 32)]
        cam_resized = [F.interpolate(cam_maps_combined, size=s, mode='bilinear', align_corners=False) for s in latent_sizes]
        ig_resized = [F.interpolate(ig_maps_combined, size=s, mode='bilinear', align_corners=False) for s in latent_sizes]

        return cam_resized, ig_resized

    def train(self):
        pbar = None
        if getattr(self.cfg, "live_logging", False):
            pbar = tqdm(total=self.cfg.total_steps, desc="Mutator Training", initial=self.global_step)

        # gradient accumulation control
        accum_steps = max(1, int(getattr(self.cfg, "accumulation_steps", 1)))
        batch_idx = 0

        # ensure grads start at zero
        self.optimizer.zero_grad()

        try:
            for hvq_images, classifier_inputs, targets in self.loader:
                hvq_images = hvq_images.to(self.device)
                classifier_inputs = classifier_inputs.to(self.device)
                targets = targets.to(self.device)

                with torch.no_grad():
                    codes = self.hvqvae.get_codes(hvq_images)  # list of 3
                    current_logits = self.classifier(classifier_inputs)
                    current_probs = torch.sigmoid(current_logits)

                # Choose 1..3 attributes at random each step (to train single and multi-attribute edits)
                num_attrs = len(self.cfg.attribute_names)
                k = random.randint(1, min(3, num_attrs))
                active_attrs = torch.tensor(torch.randperm(num_attrs)[:k], device=self.device, dtype=torch.long)
                cam_maps, ig_maps = self._get_maps(classifier_inputs, targets, [int(a.item()) for a in active_attrs])

                with autocast(enabled=self.cfg.amp):
                    mutated_codes, _ = self.mutator(
                        z_list=codes,
                        cam_maps=cam_maps,
                        ig_maps=ig_maps,
                        target_vec=targets,
                        current_probs=current_probs,
                        active_attrs=active_attrs,
                    )

                    mutated_images = self.hvqvae.decode_codes(*mutated_codes)
                    mutated_images = torch.clamp(mutated_images, -1, 1)

                    mut_inputs = F.interpolate(
                        (mutated_images + 1) / 2,
                        size=self.cfg.classifier_image_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    mut_inputs = (mut_inputs - IMAGENET_MEAN.to(self.device)[:, None, None]) / IMAGENET_STD.to(self.device)[:, None, None]
                    mut_logits = self.classifier(mut_inputs)
                    mut_probs = torch.sigmoid(mut_logits)

                    # === AGGRESSIVE LOSS ===
                    total_loss = 0.0

                    # Target flip
                    target_loss = F.binary_cross_entropy_with_logits(mut_logits, targets)
                    total_loss += 40.0 * target_loss

                    # Other attributes (loose)
                    other_loss = F.binary_cross_entropy_with_logits(mut_logits, current_probs.detach())
                    total_loss += 1.0 * other_loss

                    # Fidelity
                    l1_loss = F.l1_loss(mutated_images, hvq_images)
                    total_loss += 1.0 * l1_loss

                    perc_loss = self.perceptual_fn(mutated_images, hvq_images)
                    total_loss += 0.5 * perc_loss

                    latent_prox = sum(F.mse_loss(m, c.detach()) for m, c in zip(mutated_codes, codes)) / 3
                    total_loss += 1.0 * latent_prox

                    # Margin
                    margin_loss = F.relu(0.35 - torch.abs(mut_probs - 0.5)).mean()
                    total_loss += 10.0 * margin_loss

                    # Orthogonality
                    ortho = (F.cosine_similarity(
                        self.mutator.directions.unsqueeze(1),
                        self.mutator.directions.unsqueeze(0),
                        dim=-1
                    ).triu(1)).abs().mean()
                    total_loss += 3.0 * ortho

                # scale the loss down for accumulation
                loss_for_backwards = total_loss / float(accum_steps)
                self.scaler.scale(loss_for_backwards).backward()

                batch_idx += 1
                # perform optimizer step once we've accumulated enough micro-batches
                if (batch_idx % accum_steps) == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    # update progress (per optimizer step)
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(loss=total_loss.item(), target=target_loss.item(), margin=margin_loss.item())

                    # stdout logging (as fallback)
                    if not getattr(self.cfg, "live_logging", False) or self.global_step % self.cfg.log_interval == 0:
                        print(f"Step {self.global_step} | Loss {total_loss.item():.3f} | "
                              f"Target {target_loss.item():.3f} | Margin {margin_loss.item():.3f}")

                    # Plain-text scalar logging (CSV style)
                    if getattr(self, "log_file", None) is not None:
                        # step,total,target,margin,l1,perceptual,latent_prox,ortho
                        line = f"{self.global_step},{total_loss.item():.6f},{target_loss.item():.6f},{margin_loss.item():.6f},{l1_loss.item():.6f},{perc_loss.item():.6f},{latent_prox.item():.6f},{ortho.item():.6f}\n"
                        try:
                            self.log_file.write(line)
                            self.log_file.flush()
                        except Exception:
                            # avoid crashing training on logging issues
                            pass

                    if self.global_step % self.cfg.sample_interval == 0:
                        # save samples including CAM overlays and attribute names
                        self._save_samples(hvq_images, mutated_images, cam_maps[-1], active_attrs)

                    if self.global_step % self.cfg.checkpoint_interval == 0:
                        torch.save({
                            "mutator": self.mutator.state_dict(),
                            "step": self.global_step,
                        }, self.output_dir / "checkpoints" / f"mutator_step_{self.global_step}.pth")

                    if self.global_step >= self.cfg.total_steps:
                        break
        finally:
            if pbar is not None:
                pbar.close()
            if getattr(self, "log_file", None) is not None:
                try:
                    self.log_file.write(f"=== Mutator training log ended: {datetime.now().isoformat()} ===\n")
                    self.log_file.flush()
                except Exception:
                    pass
                try:
                    self.log_file.close()
                except Exception:
                    pass

    def _save_samples(self, orig, mut, cam_map, active_attrs):
        """Save a grid of original / mutated images and CAM overlays.

        cam_map: Tensor [B,1,H_latent,W_latent] (largest latent level used)
        active_attrs: Tensor of attribute indices used for this step
        """
        from torchvision.transforms import ToPILImage
        from PIL import Image, ImageDraw

        B = orig.shape[0]
        n = min(4, B, self.cfg.save_debug_samples)

        orig_vis = (orig[:n] + 1) / 2  # to [0,1]
        mut_vis = (mut[:n] + 1) / 2

        # upsample cam to HVQ image size for overlay
        cam_up = F.interpolate(cam_map[:n], size=(self.cfg.hvqvae_image_size, self.cfg.hvqvae_image_size), mode='bilinear', align_corners=False)
        cam_up = cam_up.clamp(0, 1)

        to_pil = ToPILImage()
        images = []
        for i in range(n):
            o = orig_vis[i].cpu()
            m = mut_vis[i].cpu()
            c = cam_up[i].cpu()  # [1,H,W]

            # Normalize cam to 0..1
            cmin = float(c.min())
            cmax = float(c.max())
            if cmax - cmin > 1e-6:
                c_norm = (c - cmin) / (cmax - cmin)
            else:
                c_norm = c * 0.0

            # create red heatmap (R channel) and blend with mutated image
            heat = torch.cat([c_norm, c_norm * 0.0, c_norm * 0.0], dim=0)
            overlay = (m * 0.6) + (heat * 0.4)
            overlay = overlay.clamp(0, 1)

            pil_o = to_pil(o)
            pil_m = to_pil(m)
            pil_overlay = to_pil(overlay)

            # draw attribute names on overlay image
            draw = ImageDraw.Draw(pil_overlay)
            attr_names = [self.cfg.attribute_names[int(a.item())] for a in active_attrs]
            label = ",".join(attr_names)
            draw.text((6, 6), label, fill=(255, 255, 255))

            # combine into a horizontal strip: orig | mutated | overlay
            w, h = pil_o.size
            strip = Image.new('RGB', (w * 3, h))
            strip.paste(pil_o, (0, 0))
            strip.paste(pil_m, (w, 0))
            strip.paste(pil_overlay, (w * 2, 0))

            images.append(torch.from_numpy(np.array(strip)).permute(2, 0, 1).float() / 255.0)

        if len(images) == 0:
            return

        grid = torch.stack(images, dim=0)
        save_image(grid.cpu(), self.output_dir / "samples" / f"step_{self.global_step}.png",
                   nrow=1, normalize=False)


if __name__ == "__main__":
    # Simple CLI can be added if needed
    cfg = MutatorTrainingConfig(
        data_root="path/to/images",
        attributes_csv="path/to/attr.csv",
        attribute_names=[
        "Bald",
        "Bangs",
        "Black_Hair",
        "Blond_Hair",
        "Brown_Hair",
        "Bushy_Eyebrows",
        "Eyeglasses",
        "Male",
        "Mouth_Slightly_Open",
        "Mustache",
        "Pale_Skin",
        "Young",
        ],
        hvqvae_checkpoint="path/to/hvqvae.pth",
        classifier_checkpoint="path/to/classifier.pth",
        output_dir="outputs/mutator_final",
    )
    trainer = MutatorTrainer(cfg)
    trainer.train()