import torch
import torch.nn as nn
import torch.nn.functional as F
from ..unsupervised_latentspace.model import HVQVAE_3Level
from ..classifiers.model import ResNet50_CBAM
from .latent_mutator import LatentMutator
from .synthesis import HierarchicalSynthesisNet
from pathlib import Path

class CounterfactualGenerator(nn.Module):
    def __init__(self, cfg, vqvae_path, classifier_path, device='cuda'):
        super().__init__()
        self.device = device
        
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

        self.classifier = ResNet50_CBAM(num_classes=cfg.num_classes).to(device)

        clf_ckpt = torch.load(classifier_path, map_location=device)
        if isinstance(clf_ckpt, dict):
            if 'model_state' in clf_ckpt:
                clf_state = clf_ckpt['model_state']
            elif 'state_dict' in clf_ckpt:
                clf_state = clf_ckpt['state_dict']
            else:
                clf_state = {k: v for k, v in clf_ckpt.items() if isinstance(v, torch.Tensor)}
        else:
            clf_state = clf_ckpt

        incompatible = self.classifier.load_state_dict(clf_state, strict=False)
        if incompatible.missing_keys:
            print(f"⚠️ Classifier missing {len(incompatible.missing_keys)} keys (first 3 shown):")
            for k in incompatible.missing_keys[:3]:
                print(f"   • {k}")
        if incompatible.unexpected_keys:
            print(f"⚠️ Classifier unexpected {len(incompatible.unexpected_keys)} keys (first 3 shown):")
            for k in incompatible.unexpected_keys[:3]:
                print(f"   • {k}")

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
                print(f"🔄 Loading Sharpened StyleGAN decoder from {candidate}...")
                state_dict = torch.load(candidate, map_location=device)
                self.decoder.load_state_dict(state_dict)
                decoder_loaded = True
                break

        if not decoder_loaded:
            decoder_ckpt_path = getattr(cfg, "decoder_checkpoint_path", None)
            if decoder_ckpt_path:
                ckpt_path = Path(decoder_ckpt_path)
                if ckpt_path.is_file():
                    print(f"🔍 Falling back to decoder weights at {ckpt_path}")
                    self.decoder.load_state_dict(torch.load(ckpt_path, map_location=device))
                    decoder_loaded = True
                else:
                    print(f"⚠️ Decoder checkpoint not found at {ckpt_path}; starting with random weights.")
            else:
                print("⚠️ No sharpened decoder path provided; starting with random weights.")

        for p in self.decoder.parameters():
            p.requires_grad = False

        self.mutator = LatentMutator(embed_dim=cfg.embed_dim, num_attributes=cfg.num_attributes).to(device)

    def re_quantize(self, z, quantizer):
        """Standard quantization"""
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        flat_z = z_perm.view(-1, z.shape[1])
        dist = torch.cdist(flat_z, quantizer.embedding.float())
        indices = torch.argmin(dist, dim=1)
        z_q = torch.nn.functional.embedding(indices, quantizer.embedding).view(z_perm.shape)
        z_q = z_perm + (z_q - z_perm).detach()
        return z_q.permute(0, 3, 1, 2).contiguous()

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

        # D. Decode (New StyleGAN Decoder)
        # Note: The decoder does NOT need the masks. It renders whatever z_final says.
        # The 'Surgical Precision' comes from the fact that z_final was only edited
        # in specific regions by the Mutator.
        img_out = self.decoder(z_final)
        
        return img_out, z_mutated, step_values