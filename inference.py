import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

from src.synthesis.config import Config
from src.synthesis.generator import CounterfactualGenerator
from src.classifiers.integrated_gradients import integrated_gradients_batch
from src.synthesis.dataset import SELECTED_ATTRIBUTES


try:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	_HAS_MPL = True
except Exception:
	plt = None  # type: ignore
	_HAS_MPL = False


IMAGE_PATH = Path(
	"/mnt/c/for exports/ceGAN/Dataset/celeba_70percent_721/test/img_align_celeba/007410.jpg"
)
MUTATOR_CKPT = Path(
	"outputs/synth_network/CF_generator/ceGAN_counterfactual_20260103/checkpoints/epoch_010.pth"
)
DECODER_CKPT = Path("outputs/synth_network/stylegan_decoder/latest.pth")
OUTPUT_DIR = Path("outputs/inference_single_image")


def _normalize_windows_path_maybe(path_str: str) -> str:
	s = path_str.strip().strip('"').strip("'")
	if not s:
		return s
	# Heuristic for Windows path like C:\foo\bar or C:/foo/bar
	if len(s) >= 3 and s[1] == ":" and (s[2] == "\\" or s[2] == "/"):
		drive = s[0].lower()
		rest = s[2:].replace("\\", "/")
		return f"/mnt/{drive}{rest}"
	return s


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Single-image counterfactual inference")
	p.add_argument("--image", type=str, default=str(IMAGE_PATH))
	p.add_argument("--attribute", type=str, default="Mouth_Slightly_Open", choices=list(SELECTED_ATTRIBUTES))
	p.add_argument(
		"--mutator",
		type=str,
		default="outputs/synth_network/CF_generator/ceGAN_counterfactual_20260204_Mouth_Slightly_Open/checkpoints/epoch_029.pth",
	)
	# inference_batch.py loads decoder ONLY from the mutator checkpoint's `decoder_state`.
	# Keep an escape hatch for manual decoder loading, but default to disabled.
	p.add_argument(
		"--decoder",
		type=str,
		default="",
		help="Optional: path to a decoder checkpoint to load before applying mutator's decoder_state (default: disabled)",
	)
	p.add_argument("--outdir", type=str, default=str(OUTPUT_DIR))

	# Key change: allow custom VQ-VAE checkpoint
	p.add_argument(
		"--vqvae",
		type=str,
		# Matches inference_batch.py IG_VQVAE_CHECKPOINT default
		default="outputs/synth_network/hr_vqvae/best_gan.pth",
		help="Path to a custom VQ-VAE checkpoint (.pth)",
	)
	p.add_argument(
		"--classifier",
		type=str,
		default="outputs/cnn_classfier/resnet18_cbam_epoch_10_128.pth",
		help="Path to classifier checkpoint (.pth)",
	)

	p.add_argument("--ig-steps", type=int, default=16)
	p.add_argument("--cam-thr", type=float, default=0.35)
	p.add_argument(
		"--no-ig",
		action="store_true",
		help="Disable saliency and use full attention mask (mask=1 baseline, matches inference_batch)"
	)

	return p.parse_args()


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
	ig_steps: int = 16,
	mask_threshold: float = 0.35,
	use_ig: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
	if not use_ig:
		h, w = image_tensor.shape[-2:]
		device = image_tensor.device
		# Match inference_batch.py baseline: full attention mask (edit everywhere)
		ig_map = torch.ones(h, w, device=device)
		mask_t = torch.ones(8, 8, device=device)
		mask_m = torch.ones(16, 16, device=device)
		mask_b = torch.ones(32, 32, device=device)
		cam_soft = torch.ones(h, w, device=device)
		return ig_map, [mask_t, mask_m, mask_b], cam_soft

	with torch.enable_grad():
		ig_attr = integrated_gradients_batch(
			model=classifier,
			input_batch=image_tensor,
			attribute_idx=attr_idx,
			target_classes=torch.tensor([int(target_class)], device=image_tensor.device),
			steps=ig_steps,
			device=image_tensor.device,
		)

	# Match training/inference_batch: positive contributions only
	raw_saliency = F.relu(ig_attr).sum(dim=1, keepdim=True)
	flat_max = raw_saliency.view(raw_saliency.size(0), -1).amax(dim=1, keepdim=True)
	flat_max = flat_max.view(-1, 1, 1, 1)
	ig_map = (raw_saliency / (flat_max + 1e-8)).squeeze(0).squeeze(0)

	smooth = F.avg_pool2d(raw_saliency, kernel_size=11, stride=1, padding=5)
	norm = smooth / (smooth.amax(dim=[1, 2, 3], keepdim=True) + 1e-8)
	norm = norm.pow(1.5)
	norm_map = F.interpolate(
		norm, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False
	).squeeze(0).squeeze(0)
	norm_map = norm_map.clamp(0.0, 1.0)

	binary_mask = (norm_map > mask_threshold).float()
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


def main() -> None:
	args = parse_args()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	outdir = Path(args.outdir)
	outdir.mkdir(parents=True, exist_ok=True)

	cfg = Config()
	cfg.vqvae_checkpoint = args.vqvae
	cfg.classifier_checkpoint = args.classifier
	cfg.active_attributes = [args.attribute]
	cfg.max_active_attributes_per_epoch = 1
	cfg.ig_steps = int(args.ig_steps)
	cfg.cam_threshold = float(args.cam_thr)

	# Match inference_batch.py inference behavior knobs
	cfg.use_decoder_blend = False
	cfg.decoder_w_from_orig = True
	cfg.mutator_step_bias_init = -2.0
	cfg.mutator_step_scale = 1.2
	cfg.mutator_mid_mult = 1.2

	generator = CounterfactualGenerator(
		cfg, vqvae_path=cfg.vqvae_checkpoint, classifier_path=cfg.classifier_checkpoint, device=device
	)

	if str(args.decoder).strip():
		decoder_ckpt = Path(_normalize_windows_path_maybe(args.decoder))
		if decoder_ckpt.is_file():
			generator.load_decoder_checkpoint(decoder_ckpt, map_location=device)
		else:
			raise FileNotFoundError(f"Decoder checkpoint missing: {decoder_ckpt}")

	mutator_ckpt = Path(_normalize_windows_path_maybe(args.mutator))
	if not mutator_ckpt.is_file():
		raise FileNotFoundError(f"Mutator checkpoint missing: {mutator_ckpt}")

	mutator_state = torch.load(mutator_ckpt, map_location=device)
	if "mutator_state" in mutator_state:
		generator.mutator.load_state_dict(mutator_state["mutator_state"], strict=True)
	else:
		generator.mutator.load_state_dict(mutator_state, strict=False)

	# Match inference_batch.py: decoder MUST come from mutator checkpoint
	decoder_state = mutator_state.get("decoder_state") if isinstance(mutator_state, dict) else None
	if decoder_state is None:
		raise FileNotFoundError(f"decoder_state missing in mutator checkpoint: {mutator_ckpt}")
	generator.decoder.load_state_dict(decoder_state, strict=False)

	generator.eval()
	generator.mutator.eval()

	image_path = Path(_normalize_windows_path_maybe(args.image))
	image_tensor = load_image(image_path, device, image_size=getattr(cfg, "image_size", 128))

	with torch.no_grad():
		logits = generator.classifier(image_tensor)
		probs = torch.sigmoid(logits)
		base_labels = (probs > 0.5).float()

	attr_names = SELECTED_ATTRIBUTES
	attr_idx = attr_names.index(args.attribute)
	conf_orig = float(probs[0, attr_idx].item())

	target_labels = base_labels.clone()
	target_labels[:, attr_idx] = 1 - target_labels[:, attr_idx]
	target_class = int(target_labels[0, attr_idx].item())

	ig_map, masks, cam_soft = compute_saliency(
		generator.classifier,
		image_tensor,
		attr_idx=attr_idx,
		target_class=target_class,
		ig_steps=int(getattr(cfg, "ig_steps", args.ig_steps)),
		mask_threshold=float(getattr(cfg, "cam_threshold", args.cam_thr)),
		use_ig=not bool(args.no_ig),
	)

	ig_batch = ig_map.unsqueeze(0)
	masks_batch = [m.unsqueeze(0) for m in masks]

	with torch.no_grad():
		x_cf, _, _ = generator(
			image_tensor,
			ig_batch,
			masks_batch,
			target_labels,
			probs,
			attr_idx,
			hard=False,
		)
		conf_cf = float(torch.sigmoid(generator.classifier(x_cf))[0, attr_idx].item())

	# Persist + print confidence scores
	scores_path = outdir / f"{args.attribute.lower()}_{image_path.stem}_scores.txt"
	with open(scores_path, "w", encoding="utf-8") as f:
		f.write(f"attribute: {args.attribute}\n")
		f.write(f"image: {image_path}\n")
		f.write(f"conf_orig: {conf_orig:.6f}\n")
		f.write(f"conf_cf: {conf_cf:.6f}\n")
		f.write(f"flip_target: {int(target_labels[0, attr_idx].item())}\n")

	print(f"Attribute: {args.attribute} | conf_orig={conf_orig:.3f} -> conf_cf={conf_cf:.3f} | scores: {scores_path}")

	orig_np = to_numpy_image((image_tensor * 0.5) + 0.5)
	cf_np = to_numpy_image((x_cf * 0.5) + 0.5)
	overlay_np = create_overlay(orig_np, cam_soft)

	composite_path = outdir / f"{args.attribute.lower()}_{image_path.stem}_composite.png"
	if _HAS_MPL:
		fig, axes = plt.subplots(1, 3, figsize=(9, 3))
		axes[0].imshow(orig_np)
		axes[0].set_title(f"Original\nconf={conf_orig:.3f}")
		axes[0].axis("off")

		axes[1].imshow(overlay_np)
		axes[1].set_title(f"Saliency Overlay ({args.attribute})")
		axes[1].axis("off")

		axes[2].imshow(cf_np)
		axes[2].set_title(f"Counterfactual\nconf={conf_cf:.3f}")
		axes[2].axis("off")

		fig.tight_layout()
		fig.savefig(composite_path, dpi=120, bbox_inches="tight")
		plt.close(fig)

	plt.imsave(outdir / f"{args.attribute.lower()}_{image_path.stem}_orig.png", orig_np)
	plt.imsave(outdir / f"{args.attribute.lower()}_{image_path.stem}_cf.png", cf_np)
	plt.imsave(outdir / f"{args.attribute.lower()}_{image_path.stem}_overlay.png", overlay_np)

	print(f"Saved composite to {composite_path}")


if __name__ == "__main__":
	main()
