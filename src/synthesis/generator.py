import torch
import torch.nn as nn
import torch.nn.functional as F
from ..unsupervised_latentspace.model import HVQVAE_3Level
from ..classifiers.model import ResNet50_CBAM,ResNet18_CBAM
from .latent_mutator import LatentMutator
from .synthesis import HierarchicalSynthesisNet
from pathlib import Path
from collections import OrderedDict

class CounterfactualGenerator(nn.Module):
    def __init__(self, cfg, vqvae_path, classifier_path, device='cuda'):
        super().__init__()
        self.device = device
        # Blend configuration (implemented in latent space before decoding)
        self.use_latent_blend = bool(getattr(cfg, "use_decoder_blend", False))
        self.blend_kernel = max(1, int(getattr(cfg, "blend_kernel_size", 3)))

        # Optional: keep decoder style (w) computed from original latents to reduce global leakage.
        self.decoder_w_from_orig = bool(getattr(cfg, "decoder_w_from_orig", False))
        
        # 1. Frozen VQ-VAE (Encoder only mostly)
        self.vqvae = HVQVAE_3Level(cfg).to(device)

        ckpt = torch.load(vqvae_path, map_location=device)
        if isinstance(ckpt, dict):
            if 'model_state' in ckpt:
                state_dict = ckpt['model_state']
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
        else:
            state_dict = ckpt

        incompatible = self.vqvae.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print(f"⚠️ VQ-VAE missing {len(incompatible.missing_keys)} keys (first 3 shown):")
            for k in incompatible.missing_keys[:3]:
                print(f"   • {k}")
        if incompatible.unexpected_keys:
            print(f"⚠️ VQ-VAE unexpected {len(incompatible.unexpected_keys)} keys (first 3 shown):")
            for k in incompatible.unexpected_keys[:3]:
                print(f"   • {k}")

        self.vqvae.eval()
        for p in self.vqvae.parameters():
            p.requires_grad = False

        clf_ckpt = torch.load(classifier_path, map_location=device)
        clf_state = self._extract_state_dict(clf_ckpt)

        classifier_cls = self._infer_classifier_cls(clf_state)
        self.classifier = classifier_cls(num_classes=cfg.num_classes).to(device)
        print(f"Number of classes for classifier: {cfg.num_classes} | Using {classifier_cls.__name__}")

        incompatible = self.classifier.load_state_dict(clf_state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            msg = []
            if incompatible.missing_keys:
                msg.append(f"missing {len(incompatible.missing_keys)} keys (first 3): {incompatible.missing_keys[:3]}")
            if incompatible.unexpected_keys:
                msg.append(f"unexpected {len(incompatible.unexpected_keys)} keys (first 3): {incompatible.unexpected_keys[:3]}")
            raise RuntimeError(f"Classifier checkpoint does not match model: {'; '.join(msg)}")

        self.classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad = False

        self.decoder = HierarchicalSynthesisNet(style_dim=512, latent_dim=64).to(device)

        sharpen_candidates = []
        sharpen_path = getattr(cfg, "sharpened_decoder_path", None)
        if sharpen_path:
            sharpen_candidates.append(Path(sharpen_path))

        sharpen_root = Path(getattr(cfg, "sharpening_checkpoint_dir", "outputs/synth_network/stylegan_decoder_sharpened"))
        latest_sharp = sharpen_root / "latest_sharp.pth"
        if latest_sharp.is_file():
            sharpen_candidates.append(latest_sharp)
        if sharpen_root.is_dir():
            epoch_ckpts = sorted(sharpen_root.glob("sharp_epoch_*.pth"), reverse=True)
            sharpen_candidates.extend(epoch_ckpts)

        decoder_loaded = False
        for candidate in sharpen_candidates:
            if candidate.is_file():
                print(f"Loading sharpened StyleGAN decoder from {candidate}...")
                self.load_decoder_checkpoint(candidate, map_location=device)
                decoder_loaded = True
                break

        if not decoder_loaded:
            decoder_ckpt_path = getattr(cfg, "decoder_checkpoint_path", None)
            if decoder_ckpt_path:
                ckpt_path = Path(decoder_ckpt_path)
                if ckpt_path.is_file():
                    print(f"Falling back to decoder weights at {ckpt_path}")
                    self.load_decoder_checkpoint(ckpt_path, map_location=device)
                    decoder_loaded = True
                else:
                    print(f"Decoder checkpoint not found at {ckpt_path}; starting with random weights.")
            else:
                print("No sharpened decoder path provided; starting with random weights.")

        for p in self.decoder.parameters():
            p.requires_grad = False

        self.mutator = LatentMutator(
            embed_dim=cfg.embed_dim,
            num_attributes=cfg.num_attributes,
            num_levels=3,
            step_min=float(getattr(cfg, "mutator_step_min", 0.1)),
            step_scale=float(getattr(cfg, "mutator_step_scale", 2.5)),
            mid_mult=float(getattr(cfg, "mutator_mid_mult", 1.5)),
            bias_init=float(getattr(cfg, "mutator_step_bias_init", 5.0)),
            ig_step_gain=float(getattr(cfg, "ig_step_gain", 0.0)),
        ).to(device)

    @staticmethod
    def _extract_state_dict(ckpt):
        if isinstance(ckpt, dict):
            if 'model_state' in ckpt:
                return ckpt['model_state']
            if 'state_dict' in ckpt:
                return ckpt['state_dict']
            if 'model_state_dict' in ckpt:
                return ckpt['model_state_dict']
            # fallback: tensor-like entries
            return {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
        return ckpt

    @staticmethod
    def _infer_classifier_cls(state_dict):
        # Heuristic: ResNet50 checkpoints (Bottleneck) contain conv3 in layer blocks
        has_conv3 = any("conv3" in k for k in state_dict.keys())
        return ResNet50_CBAM if has_conv3 else ResNet18_CBAM

    def _prepare_decoder_state(self, raw_state):
        """Normalize decoder checkpoints created before the dual-mapping refactor."""
        if isinstance(raw_state, dict):
            if 'state_dict' in raw_state:
                state_dict = raw_state['state_dict']
            elif 'model_state' in raw_state:
                state_dict = raw_state['model_state']
            else:
                state_dict = OrderedDict((k, v) for k, v in raw_state.items() if isinstance(v, torch.Tensor))
        else:
            state_dict = raw_state

        remapped = OrderedDict()
        has_mapping_m = any(k.startswith('mapping_m') for k in state_dict.keys())

        for key, tensor in state_dict.items():
            new_key = key
            if key.startswith('mapping.') and not key.startswith('mapping.net.'):
                suffix = key.split('.', 1)[1]
                new_key = f"mapping.net.{suffix}"
            elif key.startswith('mapping_m.') and not key.startswith('mapping_m.net.'):
                suffix = key.split('.', 1)[1]
                new_key = f"mapping_m.net.{suffix}"

            remapped[new_key] = tensor

        if not has_mapping_m:
            # Legacy checkpoints only had a single mapping network; mirror weights to mapping_m.
            for key, tensor in list(remapped.items()):
                if key.startswith('mapping.net.'):
                    suffix = key.split('mapping.net.', 1)[1]
                    mirror_key = f"mapping_m.net.{suffix}"
                    remapped.setdefault(mirror_key, tensor.clone())

        return remapped

    def _report_decoder_load_issues(self, incompatible, source_path):
        missing = incompatible.missing_keys
        unexpected = incompatible.unexpected_keys
        if not missing and not unexpected:
            return

        print(f"⚠️ Decoder load divergence for {source_path}:")
        if missing:
            print(f"   • Missing {len(missing)} keys (first 3): {[k for k in missing[:3]]}")
        if unexpected:
            print(f"   • Unexpected {len(unexpected)} keys (first 3): {[k for k in unexpected[:3]]}")

    def load_decoder_checkpoint(self, checkpoint_path, map_location=None):
        map_location = map_location or self.device
        raw_state = torch.load(checkpoint_path, map_location=map_location)
        state_dict = self._prepare_decoder_state(raw_state)
        incompatible = self.decoder.load_state_dict(state_dict, strict=False)
        self._report_decoder_load_issues(incompatible, checkpoint_path)
        return incompatible

    def re_quantize(self, z, quantizer):
        """Standard quantization"""
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        flat_z = z_perm.view(-1, z.shape[1])
        dist = torch.cdist(flat_z, quantizer.embedding.float())
        indices = torch.argmin(dist, dim=1)
        z_q = torch.nn.functional.embedding(indices, quantizer.embedding).view(z_perm.shape)
        z_q = z_perm + (z_q - z_perm).detach()
        return z_q.permute(0, 3, 1, 2).contiguous()

    def _smooth_mask(self, mask_4d: torch.Tensor) -> torch.Tensor:
        """Optionally smooth a [B,1,H,W] mask with avg pooling."""
        if self.blend_kernel <= 1:
            return mask_4d

        h, w = mask_4d.shape[-2], mask_4d.shape[-1]
        kernel = min(self.blend_kernel, h, w)
        if kernel <= 1:
            return mask_4d
        if kernel % 2 == 0:
            kernel = max(1, kernel - 1)
        if kernel <= 1:
            return mask_4d

        pad = kernel // 2
        return F.avg_pool2d(mask_4d, kernel_size=kernel, stride=1, padding=pad)

    def forward(self, x, ig_map, cam_masks, target_vec, current_probs, attr_idx, hard=True):
        # A. Encode
        with torch.no_grad():
            feat_b = self.vqvae.enc_b(x)
            feat_m = self.vqvae.enc_m(feat_b)
            feat_t = self.vqvae.enc_t(feat_m)

            q_t, _, _ = self.vqvae.quant_t(self.vqvae.quant_conv_t(feat_t))
            q_m, _, _ = self.vqvae.quant_m(self.vqvae.quant_conv_m(feat_m))
            q_b, _, _ = self.vqvae.quant_b(self.vqvae.quant_conv_b(feat_b))
            z_list = [q_t, q_m, q_b]

        # B. Mutate
        # The Mutator uses the IG/CAM guidance to modify the latents
        z_mutated, step_values = self.mutator(z_list, ig_map, cam_masks, target_vec, current_probs, attr_idx)
        
        # C. Re-Quantize (Optional based on 'hard' flag)
        if hard:
            z_final_t = self.re_quantize(z_mutated[0], self.vqvae.quant_t)
            z_final_m = self.re_quantize(z_mutated[1], self.vqvae.quant_m)
            z_final_b = self.re_quantize(z_mutated[2], self.vqvae.quant_b)
            z_final = [z_final_t, z_final_m, z_final_b]
        else:
            z_final = z_mutated # Continuous for gradients

        # C2. Latent-space blending (replace pixel-space blend):
        # Keep original latents outside the saliency mask, apply edits only inside.
        if self.use_latent_blend and cam_masks is not None:
            blended = []
            for level_idx, (z_o, z_e, m) in enumerate(zip(z_list, z_final, cam_masks)):
                # cam_masks[level] is expected as [B, H_l, W_l]
                if m is None:
                    blended.append(z_e)
                    continue
                m4 = m.unsqueeze(1).to(z_e.dtype)
                m4 = m4.clamp(0.0, 1.0)
                m4 = self._smooth_mask(m4)
                blended.append(m4 * z_e + (1.0 - m4) * z_o)
            z_final = blended

        # D. Decode (New StyleGAN Decoder)
        # Note: The decoder does NOT need the masks. It renders whatever z_final says.
        # The 'Surgical Precision' comes from the fact that z_final was only edited
        # in specific regions by the Mutator.

        w_override = None
        if self.decoder_w_from_orig:
            # Compute global style from original latents to avoid global modulation changes.
            # Note: z_list comes from a no_grad encode path, so this stabilizes style w.r.t edits
            # while still allowing decoder weights to train.
            w_t = self.decoder.mapping(z_list[0])
            w_m = self.decoder.mapping_m(z_list[1])
            w_override = w_t + w_m

        img_out = self.decoder(z_final, w_override=w_override)
        
        return img_out, z_mutated, step_values