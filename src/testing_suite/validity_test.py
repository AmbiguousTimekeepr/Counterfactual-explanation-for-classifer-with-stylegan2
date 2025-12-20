"""Validity testing for latent mutator counterfactuals.

This module computes the probability drop ("confidence gap") achieved by a
counterfactual generator while tracking the visual distortion introduced. The
primary metric – the Validity Score – is the mean probability shift towards the
requested counterfactual outcome. A quadrant plot (Confidence Drop vs. LPIPS)
helps diagnose whether edits are valid counterfactuals, adversarial artefacts, or
identity-destroying failures.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import lpips
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from .lspace_testing import load_classifier, load_hvq_model
from ..classifiers.dataset import CelebADataset, load_attribute_dataframe
from ..classifiers.model import SquarePadResize
from ..classifiers.resnet18_CBAM.gradcam import GradCAMPlusPlus
from ..classifiers.resnet18_CBAM.integrated_gradients import integrated_gradients
from ..latent_mutator.module import LatentMutator
from ..unsupervised_latentspace.config import Config as HVQConfig

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

matplotlib.use("Agg")  # headless-safe backend for plot export

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
DEFAULT_IMAGE_SIZE = 128
CLASSIFIER_IMAGE_SIZE = 224

# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------


class ValidityDataset(Dataset):
    """Return paired tensors for HVQ-VAE and classifier evaluation."""

    def __init__(
        self,
        base_dataset: CelebADataset,
        hvq_transform: T.Compose,
        classifier_transform: T.Compose,
    ) -> None:
        self.base_dataset = base_dataset
        self.hvq_transform = hvq_transform
        self.classifier_transform = classifier_transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pil_image, attributes = self.base_dataset[index]
        hvq_view = self.hvq_transform(pil_image)
        classifier_view = self.classifier_transform(pil_image)
        return hvq_view, classifier_view, attributes


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def _load_mutator(
    checkpoint_path: Path,
    num_attributes: int,
    embed_dim: int,
    device: torch.device,
) -> LatentMutator:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Mutator checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = None
    if isinstance(payload, dict):
        for key in ("mutator", "model_state", "model_state_dict", "state_dict"):
            maybe = payload.get(key)
            if isinstance(maybe, dict):
                state_dict = maybe
                break
        if state_dict is None and all(isinstance(v, torch.Tensor) for v in payload.values()):
            state_dict = payload
    if state_dict is None:
        raise RuntimeError(f"Mutator checkpoint {checkpoint_path} does not contain weights")

    mutator = LatentMutator(num_attributes=num_attributes, embed_dim=embed_dim)
    mutator.load_state_dict(state_dict, strict=False)
    mutator.to(device)
    mutator.eval()
    for param in mutator.parameters():
        param.requires_grad_(False)
    return mutator


# -----------------------------------------------------------------------------
# Counterfactual generator wrapper
# -----------------------------------------------------------------------------


class MutatorInferenceHelper:
    """Thin wrapper that replicates the trainer's inference utilities."""

    def __init__(
        self,
        hvq_model: torch.nn.Module,
        mutator: LatentMutator,
        classifier: torch.nn.Module,
        ig_steps: int,
        device: torch.device,
    ) -> None:
        self.hvq = hvq_model
        self.mutator = mutator
        self.classifier = classifier
        self.ig_steps = ig_steps
        self.device = device
        self.cls_mean = IMAGENET_MEAN.to(device).view(1, 3, 1, 1)
        self.cls_std = IMAGENET_STD.to(device).view(1, 3, 1, 1)
        self.gradcam = GradCAMPlusPlus(self.classifier, self.classifier.layer4)

    def _compute_guidance_maps(
        self,
        classifier_inputs: torch.Tensor,
        targets: torch.Tensor,
        attr_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ig_maps: List[torch.Tensor] = []
        cam_maps: List[torch.Tensor] = []
        target_vec = targets[:, attr_index]
        self.gradcam.register_hooks()
        try:
            for i in range(classifier_inputs.size(0)):
                input_tensor = classifier_inputs[i : i + 1].detach().clone()
                target_class = int(target_vec[i].item() >= 0.5)
                with torch.enable_grad():
                    input_tensor.requires_grad_(True)
                    ig_attr = integrated_gradients(
                        model=self.classifier,
                        input_image=input_tensor,
                        attribute_idx=attr_index,
                        target_class=target_class,
                        steps=self.ig_steps,
                        device=self.device,
                    )
                    ig_map = ig_attr.abs().sum(dim=1, keepdim=True)
                    ig_map = ig_map / (ig_map.max() + 1e-6)
                    cam_array = self.gradcam.generate_cam(
                        input_tensor,
                        attribute_idx=attr_index,
                        target_class=target_class,
                    )
                cam_tensor = torch.from_numpy(cam_array).to(self.device)
                if cam_tensor.dim() == 2:
                    cam_tensor = cam_tensor.unsqueeze(0)
                cam_tensor = cam_tensor.unsqueeze(0).to(dtype=classifier_inputs.dtype)
                cam_tensor = cam_tensor - cam_tensor.min()
                cam_tensor = cam_tensor / (cam_tensor.max() + 1e-6)
                ig_maps.append(ig_map.detach())
                cam_maps.append(cam_tensor.detach())
        finally:
            self.gradcam.remove_hooks()
        return (
            torch.cat(ig_maps, dim=0).to(self.device),
            torch.cat(cam_maps, dim=0).to(self.device),
        )

    @torch.no_grad()
    def prepare_classifier_inputs(self, images: torch.Tensor) -> torch.Tensor:
        images_01 = (images + 1.0) * 0.5
        resized = F.interpolate(
            images_01,
            size=(CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        return (resized - self.cls_mean) / self.cls_std

    @torch.no_grad()
    def generate(
        self,
        hvq_images: torch.Tensor,
        classifier_inputs: torch.Tensor,
        current_targets: torch.Tensor,
        attr_index: int,
    ) -> torch.Tensor:
        codes_top, codes_mid, codes_bot = self.hvq.get_codes(hvq_images)
        current_probs = torch.sigmoid(self.classifier(classifier_inputs))
        ig_map, cam_map = self._compute_guidance_maps(
            classifier_inputs,
            current_targets,
            attr_index,
        )
        mutated_codes, _ = self.mutator(
            [codes_top, codes_mid, codes_bot],
            ig_map_strength=ig_map,
            cam_map=cam_map,
            target_vec=current_targets,
            current_probs=current_probs,
            active_attr_idx=attr_index,
        )
        mutated_images = torch.clamp(self.hvq.decode_codes(*mutated_codes), -1.0, 1.0)
        return mutated_images


# -----------------------------------------------------------------------------
# Validity tester
# -----------------------------------------------------------------------------


class ValidityTester:
    def __init__(
        self,
        generator: MutatorInferenceHelper,
        classifier: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        lpips_net: str = "vgg",
    ) -> None:
        self.gen = generator
        self.clf = classifier
        self.loader = dataloader
        self.device = device
        self.lpips_fn = lpips.LPIPS(net=lpips_net).to(device).eval()

    def run_validity_test(self, target_attr_idx: int, num_batches: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        print(f"📉 Running Validity Test for attribute index {target_attr_idx}...")

        conf_drops: List[float] = []
        lpips_scores: List[float] = []
        success_flags: List[int] = []

        self.gen.hvq.eval()
        self.gen.mutator.eval()
        self.clf.eval()

        iterator = enumerate(self.loader)
        total_batches = len(self.loader) if num_batches is None else min(len(self.loader), num_batches)

        with torch.no_grad():
            for batch_idx, batch in tqdm(iterator, total=total_batches):
                if num_batches is not None and batch_idx >= num_batches:
                    break

                hvq_images, classifier_inputs, attrs = batch
                hvq_images = hvq_images.to(self.device)
                classifier_inputs = classifier_inputs.to(self.device)
                attrs = attrs.to(self.device)

                logits_orig = self.clf(classifier_inputs)
                probs_orig = torch.sigmoid(logits_orig)[:, target_attr_idx]

                target_vec = attrs.clone()
                target_vec[:, target_attr_idx] = 1 - target_vec[:, target_attr_idx]

                mutated_images = self.gen.generate(hvq_images, classifier_inputs, target_vec, target_attr_idx)
                classifier_inputs_cf = self.gen.prepare_classifier_inputs(mutated_images)
                logits_new = self.clf(classifier_inputs_cf)
                probs_new = torch.sigmoid(logits_new)[:, target_attr_idx]

                target_values = target_vec[:, target_attr_idx]
                delta = torch.where(target_values == 1, probs_new - probs_orig, probs_orig - probs_new)

                d_lpips = self.lpips_fn(hvq_images, mutated_images).view(-1)

                conf_drops.extend(delta.cpu().numpy())
                lpips_scores.extend(d_lpips.cpu().numpy())

                flipped = ((target_values == 1) & (probs_new > 0.5)) | ((target_values == 0) & (probs_new < 0.5))
                success_flags.extend(flipped.int().cpu().numpy())

        return np.array(conf_drops), np.array(lpips_scores), np.array(success_flags)

    @staticmethod
    def plot_results(
        conf_drops: np.ndarray,
        lpips_scores: np.ndarray,
        success_flags: np.ndarray,
        confidence_threshold: float,
        lpips_threshold: float,
        output_path: Path,
    ) -> None:
        plt.figure(figsize=(10, 8))
        colors = ["green" if flag == 1 else "red" for flag in success_flags]
        plt.scatter(lpips_scores, conf_drops, c=colors, alpha=0.6, edgecolors="none")

        plt.axhline(confidence_threshold, color="gray", linestyle="--", label="Confidence threshold")
        plt.axvline(lpips_threshold, color="gray", linestyle="--", label="LPIPS threshold")

        plt.title("Validity Test: Confidence Drop vs. LPIPS")
        plt.xlabel("LPIPS distance (visual change)")
        plt.ylabel("Confidence gain toward target")
        plt.grid(True, alpha=0.3)

        plt.text(0.05 * lpips_threshold, confidence_threshold + 0.2, "Adversarial Attack\n(Hidden noise)", color="red", fontsize=12, fontweight="bold")
        plt.text(lpips_threshold * 2.0, confidence_threshold + 0.2, "Valid Counterfactual\n(Success)", color="green", fontsize=12, fontweight="bold")
        plt.text(lpips_threshold * 2.0, -0.1, "Identity Destruction\n(Failed edit)", color="orange", fontsize=12)
        plt.text(0.05 * lpips_threshold, -0.1, "No Change\n(Null)", color="gray", fontsize=12)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"✅ Validity plot saved to {output_path}")


# -----------------------------------------------------------------------------
# CLI utilities
# -----------------------------------------------------------------------------


def _build_dataloader(
    data_root: Path,
    attribute_csv: Path,
    attribute_names: Sequence[str],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    attribute_df = load_attribute_dataframe(str(attribute_csv), attribute_names)
    base_dataset = CelebADataset(df=attribute_df, root_dir=str(data_root), transform=None)

    hvq_transform = T.Compose([
        SquarePadResize(DEFAULT_IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    classifier_transform = T.Compose([
        SquarePadResize(CLASSIFIER_IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
    ])

    dataset = ValidityDataset(base_dataset, hvq_transform, classifier_transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _resolve_attribute_list(args: argparse.Namespace) -> Sequence[str]:
    if args.attributes:
        return [attr.strip() for attr in args.attributes.split(",") if attr.strip()]
    raise ValueError("Attribute list is required")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run validity test on the latent mutator")
    parser.add_argument("--data-root", required=True, help="Directory with CelebA images (img_align_celeba)")
    parser.add_argument("--attributes-csv", required=True, help="Path to CelebA attribute CSV")
    parser.add_argument("--attributes", required=True, help="Comma-separated attribute names")
    parser.add_argument("--hvqvae-checkpoint", required=True, help="Path to HVQ-VAE checkpoint (best.pth)")
    parser.add_argument("--classifier-checkpoint", required=True, help="Path to classifier weights")
    parser.add_argument("--mutator-checkpoint", required=True, help="Path to trained mutator checkpoint")
    parser.add_argument("--target-attr", type=int, required=True, help="Attribute index to evaluate")
    parser.add_argument("--output-dir", default="outputs/testing_suite/validity", help="Directory for plots and metrics")
    parser.add_argument("--device", default="cuda", help="Computation device")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=None, help="Limit batches for quick runs")
    parser.add_argument("--ig-steps", type=int, default=32, help="Integrated gradients steps")
    parser.add_argument("--lpips-net", default="vgg", help="LPIPS backbone (alex|vgg)")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Threshold for confidence drop quadrant")
    parser.add_argument("--lpips-threshold", type=float, default=0.1, help="Threshold for LPIPS quadrant")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    attribute_names = _resolve_attribute_list(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataloader = _build_dataloader(
        data_root=Path(args.data_root),
        attribute_csv=Path(args.attributes_csv),
        attribute_names=attribute_names,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    hvq_cfg = HVQConfig()
    hvq_cfg.device = device.type
    hvq_model = load_hvq_model(Path(args.hvqvae_checkpoint), hvq_cfg, device)
    hvq_model.to(device)
    hvq_model.eval()
    for param in hvq_model.parameters():
        param.requires_grad_(False)

    classifier, inferred_classes = load_classifier(Path(args.classifier_checkpoint), device)
    classifier.eval()

    if inferred_classes != len(attribute_names):
        print(
            f"⚠️  Classifier output dimension ({inferred_classes}) does not match provided attribute count ({len(attribute_names)})."
        )

    embed_dim = hvq_model.quant_conv_t.out_channels
    mutator = _load_mutator(Path(args.mutator_checkpoint), len(attribute_names), embed_dim, device)

    generator = MutatorInferenceHelper(
        hvq_model=hvq_model,
        mutator=mutator,
        classifier=classifier,
        ig_steps=args.ig_steps,
        device=device,
    )

    tester = ValidityTester(
        generator=generator,
        classifier=classifier,
        dataloader=dataloader,
        device=device,
        lpips_net=args.lpips_net,
    )

    conf_drops, lpips_scores, success_flags = tester.run_validity_test(
        target_attr_idx=args.target_attr,
        num_batches=args.num_batches,
    )

    validity_score = float(np.mean(conf_drops)) if conf_drops.size else float("nan")
    flip_rate = float(np.mean(success_flags)) if success_flags.size else float("nan")
    std_drop = float(np.std(conf_drops)) if conf_drops.size else float("nan")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = output_dir / "validity_quadrant_plot.png"
    tester.plot_results(
        conf_drops,
        lpips_scores,
        success_flags,
        confidence_threshold=args.confidence_threshold,
        lpips_threshold=args.lpips_threshold,
        output_path=plot_path,
    )

    metrics = {
        "validity_score": validity_score,
        "validity_std": std_drop,
        "flip_rate": flip_rate,
        "num_samples": int(conf_drops.size),
        "confidence_threshold": args.confidence_threshold,
        "lpips_threshold": args.lpips_threshold,
    }

    metrics_path = output_dir / "validity_metrics.json"
    import json

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("\n📊 Validity summary:")
    print(f"   Mean probability drop (ΔP): {validity_score:.4f}")
    print(f"   Flip rate (boundary crossings): {flip_rate * 100:.2f}%")
    print(f"   Samples evaluated: {metrics['num_samples']}")
    print(f"   Plot: {plot_path}")
    print(f"   Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
