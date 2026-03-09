import argparse
import copy
import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import time
from PIL import Image
from torchvision import transforms

from src.classifiers.gradcam import gradcamplusplus_batch
from src.classifiers.integrated_gradients import integrated_gradients_batch
from src.synthesis.config import Config
from src.synthesis.dataset import SELECTED_ATTRIBUTES
from src.synthesis.generator import CounterfactualGenerator
from src.synthesis.loss_functions import CounterfactualLossManager

active_attribute = "Mouth_Slightly_Open" # Only one attribute at a time for inference
IMAGE_DIR = Path("/mnt/c/for exports/ceGAN/Dataset/celeba_70percent_721/test/img_align_celeba")
attive_cf_path = active_attribute.lower()
print(f"Generating counterfactuals for attribute: {attive_cf_path}")
MUTATOR_CKPT = Path(
	f"outputs/synth_network/CF_generator/ceGAN_counterfactual_20260204/checkpoints/epoch_029.pth"
)
OUTPUT_DIR = Path("outputs/inference_batch")
MAX_IMAGES = 100
# Optional: define no-IG checkpoints here so you don't need to pass via CLI.
# Set to a Path(...) to use, or leave as None to rely on CLI or reuse the primary checkpoints.
NO_IG_MUTATOR_CKPT = Path("outputs/synth_network/CF_generator/ceGAN_counterfactual_20260204_no_ig/checkpoints/epoch_029.pth")

# Saliency guidance for the "saliency-on" branch.
# Options: "ig" (Integrated Gradients) or "gradcampp" (Grad-CAM++).
SALIENCY_METHOD = "ig"

# Optional: define a separate mutator checkpoint to use when running Grad-CAM++ saliency.
# Set to a Path(...) to use, or leave as None to reuse the primary mutator.
GRADCAMPP_MUTATOR_CKPT = Path("outputs/synth_network/CF_generator/ceGAN_counterfactual_20260204_gradcampp/checkpoints/epoch_029.pth")


# -------------------------
# Per-branch VQVAE overrides (optional)
# -------------------------
# Decoder is ALWAYS loaded from the mutator checkpoint's `decoder_state`.
# If you want ONLY a branch to use a different VQVAE, set it here.
# Leave as None to use the base cfg value.
IG_VQVAE_CHECKPOINT: str | None = "outputs/synth_network/hr_vqvae/best_gan.pth"
NO_IG_VQVAE_CHECKPOINT: str | None = "outputs/synth_network/hr_vqvae/best_gan.pth"
GRADCAMPP_VQVAE_CHECKPOINT: str | None = "outputs/synth_network/hr_vqvae/best_gan.pth"


def get_gradcampp_target_layer(classifier: torch.nn.Module) -> torch.nn.Module:
	"""Pick a reasonable target layer for Grad-CAM++ (CBAM > layer4).

	Matches the intent in train_comprehensive.py but is defensive.
	"""
	if hasattr(classifier, "cbam4"):
		return getattr(classifier, "cbam4")
	if hasattr(classifier, "layer4"):
		return getattr(classifier, "layer4")
	raise RuntimeError("Grad-CAM++ target layer not found on classifier (expected cbam4 or layer4).")



def load_image(image_path: Path, device: torch.device, image_size: int = 128) -> torch.Tensor:
	transform = transforms.Compose(
		[
			transforms.Resize((image_size, image_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
		]
	)
	image = Image.open(image_path).convert("RGB")
	return transform(image).unsqueeze(0).to(device)


def compute_saliency(
	classifier: torch.nn.Module,
	image_tensor: torch.Tensor,
	attr_idx: int,
	target_class: int,
	ig_steps: int,
	mask_threshold: float,
	method: str = "ig",
	gradcampp_target_layer: torch.nn.Module | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
	method = (method or "ig").lower()
	if method == "ig":
		ig_attr = integrated_gradients_batch(
			model=classifier,
			input_batch=image_tensor,
			attribute_idx=attr_idx,
			target_classes=torch.tensor([int(target_class)], device=image_tensor.device),
			steps=ig_steps,
			device=image_tensor.device,
		)

		# Match training: keep positive contributions only (direction-aware)
		raw_saliency = F.relu(ig_attr).sum(dim=1, keepdim=True)
		flat_max = raw_saliency.view(raw_saliency.size(0), -1).amax(dim=1, keepdim=True)
		flat_max = flat_max.view(-1, 1, 1, 1)
		ig_map = (raw_saliency / (flat_max + 1e-8)).squeeze(0).squeeze(0)
	elif method == "gradcampp":
		if gradcampp_target_layer is None:
			raise RuntimeError("gradcampp_target_layer is required when method='gradcampp'.")
		cam_np = gradcamplusplus_batch(
			model=classifier,
			target_layer=gradcampp_target_layer,
			input_batch=image_tensor,
			attribute_idx=attr_idx,
			target_classes=torch.tensor([int(target_class)], device=image_tensor.device),
			device=image_tensor.device,
		)
		cam_t = torch.from_numpy(cam_np).to(image_tensor.device).unsqueeze(1)  # [1,1,h,w]
		cam_t = F.interpolate(
			cam_t,
			size=image_tensor.shape[-2:],
			mode="bilinear",
			align_corners=False,
		)
		raw_saliency = cam_t
		ig_map = cam_t.squeeze(0).squeeze(0)
	else:
		raise ValueError(f"Unknown saliency method: {method!r}. Use 'ig' or 'gradcampp'.")

	smooth = F.avg_pool2d(raw_saliency, kernel_size=11, stride=1, padding=5)
	norm = smooth / (smooth.amax(dim=[1, 2, 3], keepdim=True) + 1e-8)
	norm = norm.pow(1.5)
	norm_map = F.interpolate(
		norm, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False
	).squeeze(0).squeeze(0)
	norm_map = norm_map.clamp(0.0, 1.0)

	binary_mask = (norm_map > mask_threshold).float()
	# Use nearest for binary mask resizing (matches training implementation)
	mask_t = F.interpolate(
		binary_mask.unsqueeze(0).unsqueeze(0),
		size=(8, 8),
		mode="nearest",
	).squeeze(0).squeeze(0)
	mask_m = F.interpolate(
		binary_mask.unsqueeze(0).unsqueeze(0),
		size=(16, 16),
		mode="nearest",
	).squeeze(0).squeeze(0)
	mask_b = F.interpolate(
		binary_mask.unsqueeze(0).unsqueeze(0),
		size=(32, 32),
		mode="nearest",
	).squeeze(0).squeeze(0)

	return ig_map.detach(), [mask_t.detach(), mask_m.detach(), mask_b.detach()], norm_map.detach()


def make_full_attention_masks(image_tensor: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
	h, w = image_tensor.shape[-2:]
	# Use zero saliency map to simulate "no IG" while keeping masks active
	ones_map = torch.ones((h, w), device=image_tensor.device)
	mask_t = torch.ones((8, 8), device=image_tensor.device)
	mask_m = torch.ones((16, 16), device=image_tensor.device)
	mask_b = torch.ones((32, 32), device=image_tensor.device)
	return ones_map, [mask_t, mask_m, mask_b], ones_map

def to_numpy_image(t: torch.Tensor) -> np.ndarray:
	x = t.detach().cpu().clamp(0, 1)[0]
	return x.permute(1, 2, 0).numpy()


def create_overlay(base_img: np.ndarray, saliency_map: torch.Tensor, alpha: float = 0.4) -> np.ndarray:
	heatmap = cv2.applyColorMap(
		(saliency_map.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8),
		cv2.COLORMAP_JET,
	)
	heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
	overlay = alpha * heatmap + (1 - alpha) * base_img
	return np.clip(overlay, 0, 1)


def create_diff_heatmap(base_img: np.ndarray, cf_img: np.ndarray) -> np.ndarray:
	"""Create a jet heatmap showing absolute per-pixel difference between `cf_img` and `base_img`.

	Both inputs are HxWx3 in [0,1]. Returns HxWx3 RGB heatmap in [0,1].
	"""
	diff = np.abs(cf_img - base_img)
	# Aggregate across channels
	diff_map = diff.mean(axis=2)
	mn = diff_map.min()
	mx = diff_map.max()
	norm = (diff_map - mn) / (mx - mn + 1e-8)
	heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
	heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB) / 255.0
	return heat


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--max-images", type=int, default=MAX_IMAGES, help="Number of images to process")
	parser.add_argument(
		"--saliency-method",
		type=str,
		default=None,
		help=f"Saliency for the saliency-on branch: 'ig' or 'gradcampp' (default: {SALIENCY_METHOD})",
	)
	parser.add_argument(
		"--no-ig-mutator",
		type=str,
		default=None,
		help="Path to an alternate mutator checkpoint to use for the no-IG generator",
	)
	parser.add_argument(
		"--gradcampp-mutator",
		type=str,
		default=None,
		help="Path to an alternate mutator checkpoint to use for the Grad-CAM++ saliency branch",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	# overall start timer for the whole inference run
	start_all = time.perf_counter()

	# Resolve final no-IG mutator: prefer CLI, fall back to file-level constant
	args_no_ig_mutator = args.no_ig_mutator if getattr(args, "no_ig_mutator", None) else (
		str(NO_IG_MUTATOR_CKPT) if NO_IG_MUTATOR_CKPT is not None else None
	)

	saliency_method = (args.saliency_method or SALIENCY_METHOD).lower()
	if saliency_method not in {"ig", "gradcampp"}:
		raise ValueError(f"Invalid saliency method: {saliency_method!r}. Use 'ig' or 'gradcampp'.")

	cfg = Config()
	cfg.vqvae_checkpoint = "outputs/checkpoints_production/checkpoints/checkpoint_step_22500.pth"
	cfg.classifier_checkpoint = "outputs/cnn_classfier/resnet18_cbam_epoch_10_128.pth"

	# Inference-relevant config values (set from your supplied list where applicable).
	# Only include parameters that affect inference behavior; ignore training-only params and paths.
	cfg.use_ig = getattr(cfg, 'use_ig', True)  # informational here; this script still compares IG vs no-IG
	cfg.ig_steps = getattr(cfg, 'ig_steps', 16)
	cfg.cam_threshold = getattr(cfg, 'cam_threshold', 0.35)

	# These DO affect inference output quality (used inside CounterfactualGenerator / LatentMutator)
	# Match your training config to avoid color-shift / blur.
	cfg.use_decoder_blend = True
	cfg.decoder_w_from_orig = True
	cfg.mutator_step_bias_init = -2.0
	cfg.mutator_step_scale = 1.2
	cfg.mutator_mid_mult = 1.2
	# Keep attribute selection narrow (you requested Mouth_Slightly_Open only)
	cfg.active_attributes = [active_attribute]
	cfg.max_active_attributes_per_epoch = 1

	# --- IG branch generator (with optional per-branch VQVAE override) ---
	cfg_ig = copy.deepcopy(cfg)
	if IG_VQVAE_CHECKPOINT is not None:
		cfg_ig.vqvae_checkpoint = IG_VQVAE_CHECKPOINT

	generator = CounterfactualGenerator(
		cfg_ig, vqvae_path=cfg_ig.vqvae_checkpoint, classifier_path=cfg_ig.classifier_checkpoint, device=device
	)

	if not MUTATOR_CKPT.is_file():
		raise FileNotFoundError(f"Mutator checkpoint missing: {MUTATOR_CKPT}")

	ckpt = torch.load(MUTATOR_CKPT, map_location=device)
	if "mutator_state" in ckpt:
		generator.mutator.load_state_dict(ckpt["mutator_state"], strict=True)
	else:
		generator.mutator.load_state_dict(ckpt, strict=False)

	# Load decoder weights from the mutator checkpoint (always)
	decoder_state = ckpt.get("decoder_state") if isinstance(ckpt, dict) else None
	if decoder_state is None:
		raise FileNotFoundError(f"decoder_state missing in mutator checkpoint: {MUTATOR_CKPT}")
	generator.decoder.load_state_dict(decoder_state, strict=False)

	generator.eval()
	generator.mutator.eval()

	# We always compute Grad-CAM++ saliency/CF for plotting, so always resolve the target layer.
	gradcampp_target_layer = get_gradcampp_target_layer(generator.classifier)

	# Metrics helpers
	loss_mgr = CounterfactualLossManager(device=device)

	# Optionally prepare a separate "no IG" generator that can use a different mutator/vqvae
	no_ig_need_overrides = (NO_IG_VQVAE_CHECKPOINT is not None)
	need_no_ig_generator = (args_no_ig_mutator is not None) or no_ig_need_overrides

	generator_no_ig = generator
	if need_no_ig_generator:
		cfg_no_ig = copy.deepcopy(cfg)
		if NO_IG_VQVAE_CHECKPOINT is not None:
			cfg_no_ig.vqvae_checkpoint = NO_IG_VQVAE_CHECKPOINT

		generator_no_ig = CounterfactualGenerator(
			cfg_no_ig,
			vqvae_path=cfg_no_ig.vqvae_checkpoint,
			classifier_path=cfg_no_ig.classifier_checkpoint,
			device=device,
		)

		# load mutator state
		if args_no_ig_mutator:
			path = Path(args_no_ig_mutator)
			if not path.is_file():
				raise FileNotFoundError(f"No-IG mutator checkpoint missing: {path}")
			ckpt_no_ig = torch.load(path, map_location=device)
			if "mutator_state" in ckpt_no_ig:
				generator_no_ig.mutator.load_state_dict(ckpt_no_ig["mutator_state"], strict=True)
			else:
				generator_no_ig.mutator.load_state_dict(ckpt_no_ig, strict=False)

			decoder_state_no_ig = ckpt_no_ig.get("decoder_state") if isinstance(ckpt_no_ig, dict) else None
			if decoder_state_no_ig is None:
				raise FileNotFoundError(f"decoder_state missing in no-IG mutator checkpoint: {path}")
			generator_no_ig.decoder.load_state_dict(decoder_state_no_ig, strict=False)
		else:
			# If no separate mutator provided, reuse the loaded mutator weights from primary
			generator_no_ig.mutator.load_state_dict(generator.mutator.state_dict(), strict=False)
			if decoder_state is None:
				raise FileNotFoundError(f"decoder_state missing in mutator checkpoint: {MUTATOR_CKPT}")
			generator_no_ig.decoder.load_state_dict(decoder_state, strict=False)

		generator_no_ig.eval()
		generator_no_ig.mutator.eval()

	# Resolve final gradcampp mutator: prefer CLI, fall back to file-level constant
	args_gradcampp_mutator = args.gradcampp_mutator if getattr(args, "gradcampp_mutator", None) else (
		str(GRADCAMPP_MUTATOR_CKPT) if GRADCAMPP_MUTATOR_CKPT is not None else None
	)

	gradcampp_need_overrides = (GRADCAMPP_VQVAE_CHECKPOINT is not None)
	need_gradcampp_generator = (args_gradcampp_mutator is not None) or gradcampp_need_overrides

	# Optionally prepare a separate generator tuned for the Grad-CAM++ branch
	generator_gradcampp = generator
	if need_gradcampp_generator:
		cfg_gc = copy.deepcopy(cfg)
		if GRADCAMPP_VQVAE_CHECKPOINT is not None:
			cfg_gc.vqvae_checkpoint = GRADCAMPP_VQVAE_CHECKPOINT
		generator_gradcampp = CounterfactualGenerator(
			cfg_gc,
			vqvae_path=cfg_gc.vqvae_checkpoint,
			classifier_path=cfg_gc.classifier_checkpoint,
			device=device,
		)

		# load mutator state
		if args_gradcampp_mutator:
			path = Path(args_gradcampp_mutator)
			if not path.is_file():
				raise FileNotFoundError(f"GradCAMpp mutator checkpoint missing: {path}")
			ckpt_gc = torch.load(path, map_location=device)
			if "mutator_state" in ckpt_gc:
				generator_gradcampp.mutator.load_state_dict(ckpt_gc["mutator_state"], strict=True)
			else:
				generator_gradcampp.mutator.load_state_dict(ckpt_gc, strict=False)

			decoder_state_gc = ckpt_gc.get("decoder_state") if isinstance(ckpt_gc, dict) else None
			if decoder_state_gc is None:
				raise FileNotFoundError(f"decoder_state missing in gradcampp mutator checkpoint: {path}")
			generator_gradcampp.decoder.load_state_dict(decoder_state_gc, strict=False)
		else:
			generator_gradcampp.mutator.load_state_dict(generator.mutator.state_dict(), strict=False)
			if decoder_state is None:
				raise FileNotFoundError(f"decoder_state missing in mutator checkpoint: {MUTATOR_CKPT}")
			generator_gradcampp.decoder.load_state_dict(decoder_state, strict=False)

		generator_gradcampp.eval()
		generator_gradcampp.mutator.eval()

	attr_name = active_attribute
	attr_idx = SELECTED_ATTRIBUTES.index(attr_name)
	attr_dir = OUTPUT_DIR / attr_name
	attr_dir.mkdir(parents=True, exist_ok=True)

	image_paths = sorted(glob.glob(str(IMAGE_DIR / "*.jpg")))[: args.max_images]
	if not image_paths:
		raise FileNotFoundError(f"No images found in {IMAGE_DIR}")

	chunk_size = 5
	chunk_idx = 0

	# timing accumulators (seconds)
	saliency_times = []
	gc_times = []
	cf_ig_times = []
	cf_gc_times = []
	cf_no_ig_times = []
	metrics_all = []
	for start in range(0, len(image_paths), chunk_size):
		rows = []
		for img_path in image_paths[start : start + chunk_size]:
			img_path = Path(img_path)
			image_tensor = load_image(img_path, device, image_size=cfg.image_size)

			with torch.no_grad():
				logits = generator.classifier(image_tensor)
				probs = torch.sigmoid(logits)
				base_labels = (probs > 0.5).float()

			target_labels = base_labels.clone()
			target_labels[:, attr_idx] = 1 - target_labels[:, attr_idx]
			target_class = int(target_labels[0, attr_idx].item())

			# time saliency computation (IG and Grad-CAM++)
			t0 = time.perf_counter()
			ig_map, masks_ig, cam_soft_ig = compute_saliency(
				generator.classifier,
				image_tensor,
				attr_idx=attr_idx,
				target_class=target_class,
				ig_steps=getattr(cfg, "ig_steps", 16),
				mask_threshold=getattr(cfg, "cam_threshold", 0.35),
				method="ig",
				gradcampp_target_layer=gradcampp_target_layer,
			)
			t1 = time.perf_counter()
			ig_time = t1 - t0
			saliency_times.append(ig_time)
			ig_batch = ig_map.unsqueeze(0)
			masks_batch_ig = [m.unsqueeze(0) for m in masks_ig]

			# Also compute Grad-CAM++ saliency (for dedicated visualization and CF)
			t0_gc = time.perf_counter()
			gc_map, masks_gc, cam_soft_gc = compute_saliency(
				generator.classifier,
				image_tensor,
				attr_idx=attr_idx,
				target_class=target_class,
				ig_steps=getattr(cfg, "ig_steps", 16),
				mask_threshold=getattr(cfg, "cam_threshold", 0.35),
				method="gradcampp",
				gradcampp_target_layer=gradcampp_target_layer,
			)
			t1_gc = time.perf_counter()
			gc_time = t1_gc - t0_gc
			gc_times.append(gc_time)
			gc_batch = gc_map.unsqueeze(0)
			masks_batch_gc = [m.unsqueeze(0) for m in masks_gc]

			# prepare full masks (simulate no IG attention) for no-IG generator
			ones_map, masks_ones, _ = make_full_attention_masks(image_tensor.squeeze(0))
			ones_batch = ones_map.unsqueeze(0)
			masks_batch_no_ig = [m.unsqueeze(0) for m in masks_ones]

			with torch.no_grad():
				# time CF generation using IG saliency (use primary generator)
				t2 = time.perf_counter()
				x_cf_ig, _, _ = generator(
					image_tensor,
					ig_batch,
					masks_batch_ig,
					target_labels,
					probs,
					attr_idx,
					hard=False,
				)
				t3 = time.perf_counter()
				cf_ig_time = t3 - t2
				cf_ig_times.append(cf_ig_time)

				# time CF generation using Grad-CAM++ saliency (use gradcampp-specific generator if available)
				t2_gc = time.perf_counter()
				x_cf_gc, _, _ = generator_gradcampp(
					image_tensor,
					gc_batch,
					masks_batch_gc,
					target_labels,
					probs,
					attr_idx,
					hard=False,
				)
				t3_gc = time.perf_counter()
				cf_gc_time = t3_gc - t2_gc
				cf_gc_times.append(cf_gc_time)

				# compute counterfactual without using IG (using full/ones masks) with the chosen generator_no_ig
				t4 = time.perf_counter()
				x_cf_no_ig, _, _ = generator_no_ig(
					image_tensor,
					ones_batch,
					masks_batch_no_ig,
					target_labels,
					probs,
					attr_idx,
					hard=False,
				)
				t5 = time.perf_counter()
				cf_no_ig_time = t5 - t4
				cf_no_ig_times.append(cf_no_ig_time)

				prob_orig = torch.sigmoid(generator.classifier(image_tensor))[0, attr_idx].item()
				prob_cf_ig = torch.sigmoid(generator.classifier(x_cf_ig))[0, attr_idx].item()
				prob_cf_gc = torch.sigmoid(generator.classifier(x_cf_gc))[0, attr_idx].item()

				# Reference IG mask (for comparing locality of both methods)
				# Build combined mask (upsampled to image size) from masks_ig (which are 2D tensors)
				h, w = image_tensor.shape[-2:]
				masks_4d = [m.unsqueeze(0).unsqueeze(0) if m.ndim == 2 else m.unsqueeze(0) for m in masks_ig]
				combined_mask = torch.clamp_max(torch.stack([
					F.interpolate(m.float(), size=(h, w), mode='nearest') for m in masks_4d
				]).sum(0), 1.0).squeeze(0)  # [1,H,W] -> squeeze -> [H,W]
				preserve_mask = 1.0 - combined_mask.unsqueeze(0)

				# Compute change maps for both CF variants (in torch)
				x_orig = (image_tensor * 0.5) + 0.5
				x_cf_ig_t = (x_cf_ig * 0.5) + 0.5
				x_cf_no_ig_t = (x_cf_no_ig * 0.5) + 0.5

				# L2 / L1 energies and outside ratios (use reference preserve_mask)
				def energy_stats(xa, xb, preserve_mask):
					dx = (xb - xa).float()
					dx_l2_map = dx.pow(2).sum(dim=1, keepdim=True)  # [1,1,H,W]
					dx_l1_map = dx.abs().sum(dim=1, keepdim=True)
					outside = preserve_mask.unsqueeze(1)
					eps = 1e-8
					l2_total = dx_l2_map.sum()
					l2_out = (dx_l2_map * outside).sum()
					l1_total = dx_l1_map.sum()
					l1_out = (dx_l1_map * outside).sum()
					return {
						'dx_l2_total': float(l2_total.cpu().item()),
						'dx_l2_outside': float(l2_out.cpu().item()),
						'dx_l2_outside_ratio': float((l2_out / (l2_total + eps)).cpu().item()),
						'dx_l1_total': float(l1_total.cpu().item()),
						'dx_l1_outside': float(l1_out.cpu().item()),
						'dx_l1_outside_ratio': float((l1_out / (l1_total + eps)).cpu().item()),
					}

				stats_ig = energy_stats(x_orig, x_cf_ig_t, preserve_mask)
				stats_no_ig = energy_stats(x_orig, x_cf_no_ig_t, preserve_mask)

				# LPIPS overall and outside (spatial LPIPS available)
				lpips_overall_ig = None
				lpips_overall_no_ig = None
				lpips_out_ig = None
				lpips_out_no_ig = None
				try:
					# lpips returns [1,1,H_lp,W_lp] if spatial=True
					lp_ig_map = loss_mgr.lpips(x_cf_ig_t, x_orig)  # [1,1,h,w]
					lp_no_ig_map = loss_mgr.lpips(x_cf_no_ig_t, x_orig)
					# Resize preserve_mask to lpips spatial size
					h_lp, w_lp = lp_ig_map.shape[2:]
					mask_lp = F.interpolate(preserve_mask.unsqueeze(1), size=(h_lp, w_lp), mode='nearest')
					lpips_overall_ig = float(lp_ig_map.mean().cpu().item())
					lpips_overall_no_ig = float(lp_no_ig_map.mean().cpu().item())
					# outside mean
					mask_sum = mask_lp.sum().clamp_min(1e-6)
					lpips_out_ig = float((lp_ig_map * mask_lp).sum().cpu().item() / mask_sum.cpu().item())
					lpips_out_no_ig = float((lp_no_ig_map * mask_lp).sum().cpu().item() / mask_sum.cpu().item())
				except Exception:
					pass

				# (metrics will be appended together with image fields below)

			orig_np = to_numpy_image((image_tensor * 0.5) + 0.5)
			cf_ig_np = to_numpy_image((x_cf_ig * 0.5) + 0.5)
			cf_gc_np = to_numpy_image((x_cf_gc * 0.5) + 0.5)
			cf_no_ig_np = to_numpy_image((x_cf_no_ig * 0.5) + 0.5)
			overlay_np = create_overlay(orig_np, cam_soft_ig)
			# overlays for CF images (apply same saliency maps on CF outputs)
			overlay_cf_ig = create_overlay(cf_ig_np, cam_soft_ig)
			overlay_cf_gc = create_overlay(cf_gc_np, cam_soft_gc)
			overlay_cf_no_ig = create_overlay(cf_no_ig_np, cam_soft_ig)

			# difference heatmaps (CF - original)
			diff_ig_map = create_diff_heatmap(orig_np, cf_ig_np)
			diff_gc_map = create_diff_heatmap(orig_np, cf_gc_np)
			diff_no_ig_map = create_diff_heatmap(orig_np, cf_no_ig_np)

			row = {
				"orig": orig_np,
				"saliency": overlay_np,
				"saliency_ig": overlay_np,
				"saliency_gc": create_overlay(orig_np, cam_soft_gc),
				"cf_ig": overlay_cf_ig,
				"cf_ig_raw": cf_ig_np,
				"cf_gc": overlay_cf_gc,
				"cf_gc_raw": cf_gc_np,
				"cf_no_ig": overlay_cf_no_ig,
				"cf_no_ig_raw": cf_no_ig_np,
				"conf_orig": prob_orig,
				"conf_cf_ig": prob_cf_ig,
				"conf_cf_gc": prob_cf_gc,
				"conf_cf_no_ig": torch.sigmoid(generator.classifier(x_cf_no_ig))[0, attr_idx].item(),
				"saliency_method": saliency_method,
				"time_ig": ig_time,
				"time_gc": gc_time,
				"time_cf_ig": cf_ig_time,
				"time_cf_gc": cf_gc_time,
				"time_cf_no_ig": cf_no_ig_time,
				"diff_ig": diff_ig_map,
				"diff_gc": diff_gc_map,
				"diff_no_ig": diff_no_ig_map,
				"stem": img_path.stem,
				# Metrics
				"dx_l2_total_ig": stats_ig['dx_l2_total'],
				"dx_l2_outside_ig": stats_ig['dx_l2_outside'],
				"dx_l2_outside_ratio_ig": stats_ig['dx_l2_outside_ratio'],
				"dx_l1_total_ig": stats_ig['dx_l1_total'],
				"dx_l1_outside_ig": stats_ig['dx_l1_outside'],
				"dx_l1_outside_ratio_ig": stats_ig['dx_l1_outside_ratio'],
				"dx_l2_total_no_ig": stats_no_ig['dx_l2_total'],
				"dx_l2_outside_no_ig": stats_no_ig['dx_l2_outside'],
				"dx_l2_outside_ratio_no_ig": stats_no_ig['dx_l2_outside_ratio'],
				"dx_l1_total_no_ig": stats_no_ig['dx_l1_total'],
				"dx_l1_outside_no_ig": stats_no_ig['dx_l1_outside'],
				"dx_l1_outside_ratio_no_ig": stats_no_ig['dx_l1_outside_ratio'],
				"lpips_overall_ig": lpips_overall_ig,
				"lpips_out_ig": lpips_out_ig,
				"lpips_overall_no_ig": lpips_overall_no_ig,
				"lpips_out_no_ig": lpips_out_no_ig,
			}
			rows.append(row)
			metrics_all.append(row)

			# print per-image timings
			print(
				f"{img_path.stem}: IG {ig_time:.3f}s | GC {gc_time:.3f}s | CF(IG) {cf_ig_time:.3f}s | CF(GC) {cf_gc_time:.3f}s | CF(no-IG) {cf_no_ig_time:.3f}s"
			)

		if not rows:
			continue

		n_rows = len(rows)
		# Columns: orig, saliency_ig, cf_ig, cf_ig_raw, saliency_gc, cf_gc, cf_gc_raw,
		# cf_no_ig, cf_no_ig_raw, diff_ig, diff_gc, diff_no_ig
		n_cols = 12
		fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
		if n_rows == 1:
			axes = np.expand_dims(axes, axis=0)

		for r_idx, row in enumerate(rows):
			axes[r_idx, 0].imshow(row["orig"])
			axes[r_idx, 0].set_title(f"Original\nconf={row['conf_orig']:.3f}")
			axes[r_idx, 0].axis("off")
			axes[r_idx, 1].imshow(row["saliency_ig"])
			axes[r_idx, 1].set_title("Saliency (IG)")
			axes[r_idx, 1].axis("off")

			axes[r_idx, 2].imshow(row["cf_ig"])
			axes[r_idx, 2].set_title(f"CE (IG) overlay\nconf={row['conf_cf_ig']:.3f}")
			axes[r_idx, 2].axis("off")

			axes[r_idx, 3].imshow(row["cf_ig_raw"])
			axes[r_idx, 3].set_title(f"CE (IG) raw\nconf={row['conf_cf_ig']:.3f}")
			axes[r_idx, 3].axis("off")

			axes[r_idx, 4].imshow(row["saliency_gc"]) 
			axes[r_idx, 4].set_title("Saliency (Grad-CAM++)")
			axes[r_idx, 4].axis("off")

			axes[r_idx, 5].imshow(row["cf_gc"]) 
			axes[r_idx, 5].set_title(f"CE (GC) overlay\nconf={row['conf_cf_gc']:.3f}")
			axes[r_idx, 5].axis("off")

			axes[r_idx, 6].imshow(row["cf_gc_raw"]) 
			axes[r_idx, 6].set_title(f"CE (GC) raw\nconf={row['conf_cf_gc']:.3f}")
			axes[r_idx, 6].axis("off")

			axes[r_idx, 7].imshow(row["cf_no_ig"])
			axes[r_idx, 7].set_title(f"CE (no-IG) overlay\nconf={row['conf_cf_no_ig']:.3f}")
			axes[r_idx, 7].axis("off")

			axes[r_idx, 8].imshow(row["cf_no_ig_raw"])
			axes[r_idx, 8].set_title(f"CE (no-IG) raw\nconf={row['conf_cf_no_ig']:.3f}")
			axes[r_idx, 8].axis("off")

			axes[r_idx, 9].imshow(row["diff_ig"]) 
			axes[r_idx, 9].set_title("Diff (CE_IG - orig)")
			axes[r_idx, 9].axis("off")

			axes[r_idx, 10].imshow(row["diff_gc"]) 
			axes[r_idx, 10].set_title("Diff (CE_GC - orig)")
			axes[r_idx, 10].axis("off")

			axes[r_idx, 11].imshow(row["diff_no_ig"]) 
			axes[r_idx, 11].set_title("Diff (CE_noIG - orig)")
			axes[r_idx, 11].axis("off")

		fig.tight_layout()
		composite_path = attr_dir / f"{attr_name}_composite_{chunk_idx:03d}.png"
		fig.savefig(composite_path, dpi=140, bbox_inches="tight")
		plt.close(fig)
		chunk_idx += 1

	# --- Aggregate numeric metrics across all saved rows and write CSV + plots ---
	all_metrics = []
	for cid in range(chunk_idx):
		# The rows were appended per chunk into image files; we don't persist rows across chunks,
		# but we wrote per-chunk composites. Instead, recompute per-image metrics by re-running a
		# lightweight pass would be expensive. Simpler: gather metrics from the last `rows` buffer
		# available in the last processed chunk (best-effort). If you need full-run CSV, re-run with
		# smaller chunk_size and capture `rows` to disk inside the loop. Here we save the last-chunk CSV.
		break

	if metrics_all:
		# Save full-run CSV of numeric metrics
		csv_path = attr_dir / f"{attr_name}_ablation_metrics.csv"
		import csv as _csv
		sample = metrics_all[0]
		blacklist = ("orig","saliency","saliency_ig","saliency_gc","cf_ig","cf_ig_raw","cf_gc","cf_gc_raw","cf_no_ig","cf_no_ig_raw","diff_ig","diff_gc","diff_no_ig")
		fieldnames = [k for k in sample.keys() if not k in blacklist]
		with open(csv_path, 'w', newline='') as cf:
			writer = _csv.DictWriter(cf, fieldnames=fieldnames)
			writer.writeheader()
			for r in metrics_all:
				rowout = {k: (r[k] if k in r else None) for k in fieldnames}
				writer.writerow(rowout)

		# Plot comparisons (mean values across run)
		import numpy as _np
		metrics_to_plot = [
			('dx_l2_outside_ratio_ig', 'dx_l2_outside_ratio_no_ig', 'L2 outside ratio'),
			('dx_l1_outside_ratio_ig', 'dx_l1_outside_ratio_no_ig', 'L1 outside ratio'),
			('lpips_out_ig', 'lpips_out_no_ig', 'LPIPS outside (masked)'),
		]
		means_ig = []
		means_no = []
		labels = []
		for a, b, lab in metrics_to_plot:
			vals_a = _np.array([r[a] for r in metrics_all if r.get(a) is not None])
			vals_b = _np.array([r[b] for r in metrics_all if r.get(b) is not None])
			if vals_a.size == 0 and vals_b.size == 0:
				continue
			means_ig.append(vals_a.mean() if vals_a.size else _np.nan)
			means_no.append(vals_b.mean() if vals_b.size else _np.nan)
			labels.append(lab)

		if labels:
			fig2, ax2 = plt.subplots(1, 1, figsize=(6, 3 + 0.6 * len(labels)))
			xpos = _np.arange(len(labels))
			width = 0.35
			ax2.bar(xpos - width/2, _np.array(means_ig), width, label=f'{saliency_method}-on')
			ax2.bar(xpos + width/2, _np.array(means_no), width, label='IG-off (mask=1)')
			ax2.set_xticks(xpos)
			ax2.set_xticklabels(labels, rotation=20, ha='right')
			ax2.set_ylabel('Mean')
			ax2.legend()
			plot_path = attr_dir / f"{attr_name}_ablation_summary.png"
			fig2.tight_layout()
			fig2.savefig(plot_path, dpi=150, bbox_inches='tight')
			plt.close(fig2)

	print(f"Saved {chunk_idx} composite images to {attr_dir}")

	# print average timings
	if saliency_times:
		print(f"Average IG time per image: {sum(saliency_times)/len(saliency_times):.3f}s")
	if gc_times:
		print(f"Average Grad-CAM++ time per image: {sum(gc_times)/len(gc_times):.3f}s")
	if cf_ig_times:
		print(f"Average CF (IG) time per image: {sum(cf_ig_times)/len(cf_ig_times):.3f}s")
	if cf_gc_times:
		print(f"Average CF (Grad-CAM++) time per image: {sum(cf_gc_times)/len(cf_gc_times):.3f}s")
	if cf_no_ig_times:
		print(f"Average CF (no-IG) time per image: {sum(cf_no_ig_times)/len(cf_no_ig_times):.3f}s")

	# print total timings and overall elapsed time
	if saliency_times:
		total_ig = sum(saliency_times)
		print(f"Total IG time: {total_ig:.3f}s for {len(saliency_times)} images")
	if gc_times:
		total_gc = sum(gc_times)
		print(f"Total Grad-CAM++ time: {total_gc:.3f}s for {len(gc_times)} images")
	if cf_ig_times:
		total_cf_ig = sum(cf_ig_times)
		print(f"Total CF (IG) time: {total_cf_ig:.3f}s for {len(cf_ig_times)} images")
	if cf_gc_times:
		total_cf_gc = sum(cf_gc_times)
		print(f"Total CF (Grad-CAM++) time: {total_cf_gc:.3f}s for {len(cf_gc_times)} images")
	if cf_no_ig_times:
		total_cf_no_ig = sum(cf_no_ig_times)
		print(f"Total CF (no-IG) time: {total_cf_no_ig:.3f}s for {len(cf_no_ig_times)} images")

	end_all = time.perf_counter()
	print(f"Overall inference run time: {end_all - start_all:.3f}s")


if __name__ == "__main__":
	main()
