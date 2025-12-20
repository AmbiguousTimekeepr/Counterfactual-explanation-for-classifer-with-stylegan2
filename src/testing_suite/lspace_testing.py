#!/usr/bin/env python3
"""Comprehensive latent space evaluation suite for HVQ-GAN / HVQ-VAE models."""

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from src.unsupervised_latentspace.config import Config
from src.unsupervised_latentspace.metrics import CodebookAnalyzer, MetricsCalculator
from src.unsupervised_latentspace.model import HVQVAE_3Level
from src.classifiers import SELECTED_ATTRIBUTES
from src.classifiers.resnet18_CBAM.model import ResNet18_CBAM
from src.classifiers.resnet50_CBAM.model import ResNet50_CBAM


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None

try:  # pragma: no cover
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

try:  # pragma: no cover
    import lpips
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "lpips is required for perceptual evaluation. Install with `pip install lpips`."
    ) from exc

try:  # pragma: no cover
    from umap import UMAP
except ImportError:  # pragma: no cover
    UMAP = None

try:  # pragma: no cover
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
except ImportError:  # pragma: no cover
    FrechetInceptionDistance = None
    KernelInceptionDistance = None

try:  # pragma: no cover
    from sklearn.metrics import mutual_info_score
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scikit-learn is required for mutual information. Install with `pip install scikit-learn`."
    ) from exc

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
class CelebAAttributeDataset(Dataset):
    """Dataset returning CelebA images and attribute labels."""

    def __init__(
        self,
        image_root: Path,
        attribute_csv: Path,
        image_size: int,
        selected_attributes: Optional[List[str]] = None,
    ) -> None:
        self.image_root = Path(image_root)
        if not self.image_root.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_root}")

        self.attribute_df = pd.read_csv(attribute_csv, index_col=0)
        self.selected_attributes = list(selected_attributes or SELECTED_ATTRIBUTES)
        missing = [attr for attr in self.selected_attributes if attr not in self.attribute_df.columns]
        if missing:
            raise ValueError(f"Missing required attributes in CSV: {missing}")
        self.attribute_df = self.attribute_df.loc[:, self.selected_attributes]

        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.file_names = [name for name in self.attribute_df.index if (self.image_root / name).exists()]
        if not self.file_names:
            raise RuntimeError("No images matched attribute rows. Check paths and CSV index format.")

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_name = self.file_names[idx]
        img_path = self.image_root / file_name
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as err:
            print(f"Warning: failed to load {img_path}: {err}. Using blank image.")
            pil_img = Image.new("RGB", (self.image_size, self.image_size))

        image = self.transform(pil_img)
        attrs = self.attribute_df.loc[file_name].values.astype(np.float32)
        attrs = (attrs + 1.0) / 2.0  # convert -1/1 to 0/1
        attributes = torch.from_numpy(attrs)

        return {
            "image": image,
            "attributes": attributes,
            "file_name": file_name,
        }


def build_dataset(cfg: Config, csv_path: Path, attributes: Optional[List[str]] = None) -> Dataset:
    image_dir = Path(cfg.data_path) / "img_align_celeba"
    return CelebAAttributeDataset(image_dir, csv_path, cfg.image_size, attributes)
def denormalize(images: torch.Tensor) -> torch.Tensor:
    return torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    return torch.clamp(denormalize(images) * 255.0, 0.0, 255.0).round().to(torch.uint8)


def to_uint8_from_zero_one(images: torch.Tensor) -> torch.Tensor:
    return torch.clamp(images * 255.0, 0.0, 255.0).round().to(torch.uint8)


def load_classifier(ckpt_path: Path, device: torch.device) -> Tuple[torch.nn.Module, int]:
    checkpoint = torch.load(ckpt_path, map_location=device)

    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model_state", "model", "net"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                state_dict = candidate
                break

    if not isinstance(state_dict, dict):
        raise ValueError("Classifier checkpoint does not contain a valid state_dict.")

    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    arch = "resnet18"
    if any(".conv3." in key for key in state_dict):
        arch = "resnet50"

    num_classes = len(SELECTED_ATTRIBUTES)
    fc_weight = state_dict.get("fc.weight")
    if isinstance(fc_weight, torch.Tensor):
        num_classes = fc_weight.shape[0]

    if arch == "resnet50":
        model = ResNet50_CBAM(num_classes=num_classes)
    else:
        model = ResNet18_CBAM(num_classes=num_classes)

    model_state = model.state_dict()
    filtered_state: Dict[str, torch.Tensor] = {}
    mismatched: Dict[str, Tuple[torch.Size, torch.Size]] = {}
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            mismatched[key] = (value.shape, model_state[key].shape)
            continue
        filtered_state[key] = value

    if mismatched:
        preview = ", ".join(list(mismatched.keys())[:5])
        print(
            f"Warning: skipped {len(mismatched)} classifier parameters due to shape mismatch "
            f"(e.g. {preview})."
        )

    incompatible = model.load_state_dict(filtered_state, strict=False)
    if incompatible.unexpected_keys:
        preview = ", ".join(list(incompatible.unexpected_keys)[:5])
        print(
            f"Warning: ignored {len(incompatible.unexpected_keys)} unexpected classifier keys "
            f"(e.g. {preview})."
        )
    if incompatible.missing_keys:
        preview = ", ".join(list(incompatible.missing_keys)[:5])
        print(
            f"Warning: classifier checkpoint missing {len(incompatible.missing_keys)} parameters "
            f"after load (e.g. {preview})."
        )

    model.to(device)
    model.eval()
    return model, num_classes


def load_hvq_model(ckpt_path: Path, cfg: Config, device: torch.device) -> HVQVAE_3Level:
    model = HVQVAE_3Level(cfg).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)

    state_dict = None
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state", "model_state_dict", "state_dict"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        if state_dict is None:
            state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict()

    compatibility = model.load_state_dict(state_dict, strict=False)

    if compatibility.missing_keys:
        alias_missing = [key for key in compatibility.missing_keys if key.startswith("dec_")]
        other_missing = [key for key in compatibility.missing_keys if key not in alias_missing]
        if alias_missing:
            print(
                "Info: HVQ checkpoint omitted alias decoder keys (dec_*). "
                "Weights were loaded via decoder.* modules instead."
            )
        if other_missing:
            preview = ", ".join(other_missing[:5])
            print(
                f"Warning: HVQ checkpoint missing {len(other_missing)} parameter(s) "
                f"(e.g. {preview})."
            )

    if compatibility.unexpected_keys:
        preview = ", ".join(list(compatibility.unexpected_keys)[:5])
        print(
            f"Warning: HVQ checkpoint has {len(compatibility.unexpected_keys)} unexpected key(s) "
            f"(e.g. {preview})."
        )
    model.eval()
    return model


def calculate_perplexity(encodings: torch.Tensor) -> float:
    encodings = encodings.float()
    avg_probs = encodings.mean(dim=0)
    avg_probs = torch.clamp(avg_probs, min=1e-10)
    entropy = -torch.sum(avg_probs * torch.log(avg_probs))
    return torch.exp(entropy).item()


def render_heatmap(diff: torch.Tensor, save_path: Path) -> None:
    diff_np = diff.cpu().numpy()
    plt.figure(figsize=(3, 3))
    plt.imshow(diff_np, cmap="inferno")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_reconstruction_grid(original: torch.Tensor, recon: torch.Tensor, save_path: Path) -> None:
    comparison = torch.stack([denormalize(original), denormalize(recon)], dim=1)
    b, _, c, h, w = comparison.shape
    comparison = comparison.view(b * 2, c, h, w)
    grid = make_grid(comparison, nrow=2, padding=2, normalize=False)
    save_image(grid, save_path)


def save_difference_visual(original: torch.Tensor, recon: torch.Tensor, save_dir: Path, prefix: str) -> None:
    diff = (recon - original).abs().mean(dim=1)
    for idx, diff_map in enumerate(diff):
        heatmap_path = save_dir / f"{prefix}_sample{idx:03d}.png"
        render_heatmap(diff_map, heatmap_path)

def _resolve_usage_tensor(usage: object, level_name: str, level_idx: int) -> torch.Tensor:
    if isinstance(usage, dict):
        normalized = {str(k).lower(): v for k, v in usage.items()}
        for key in (level_name.lower(), f"level_{level_idx}"):
            if key in normalized:
                return normalized[key]
        raise KeyError(f"Usage stats missing for level '{level_name}' (index {level_idx})")

    if level_idx >= len(usage):
        raise KeyError(f"Usage stats list missing index {level_idx} for level '{level_name}'")
    return usage[level_idx]


def log_embedding_diagnostics(model: HVQVAE_3Level, usage: object, save_dir: Path) -> None:
    os.makedirs(save_dir, exist_ok=True)
    level_names = ["top", "mid", "bottom"]
    reducers: List[Tuple[str, Optional[object]]] = [("tsne", TSNE(n_components=2, init="pca", learning_rate="auto"))]
    if UMAP is not None:
        reducers.append(("umap", UMAP(n_components=2, init="spectral")))

    for level_idx, level_name in enumerate(level_names):
        embedding = getattr(model, f"quant_{level_name[0]}").embedding.detach().cpu().numpy()
        usage_tensor = _resolve_usage_tensor(usage, level_name, level_idx)
        usage_counts = usage_tensor.cpu().numpy()

        plt.figure(figsize=(5, 4))
        plt.hist(np.linalg.norm(embedding, axis=1), bins=40, color="steelblue", alpha=0.85)
        plt.title(f"{level_name.capitalize()} embedding norms")
        plt.xlabel("L2 norm")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(save_dir / f"codebook_{level_name}_norm_hist.png", dpi=200)
        plt.close()

        for reducer_name, reducer in reducers:
            try:
                projection = reducer.fit_transform(embedding)
            except Exception as err:
                print(f"Warning: {reducer_name} failed for {level_name}: {err}")
                continue
            plt.figure(figsize=(6, 6))
            plt.scatter(
                projection[:, 0],
                projection[:, 1],
                c=usage_counts,
                cmap="viridis",
                s=10,
                alpha=0.85,
            )
            plt.title(f"{level_name.capitalize()} codebook ({reducer_name.upper()})")
            plt.colorbar(label="usage count")
            plt.tight_layout()
            plt.savefig(save_dir / f"codebook_{level_name}_{reducer_name}.png", dpi=200)
            plt.close()


def summarize_codebook_usage(analyzer: CodebookAnalyzer, num_embeddings: object) -> Dict[str, float]:
    stats: Dict[str, float] = {}

    usage_items: List[Tuple[str, torch.Tensor]]
    if isinstance(analyzer.usage_counts, dict):
        usage_items = list(analyzer.usage_counts.items())
    else:
        default_levels = ["top", "mid", "bottom"]
        usage_items = []
        for idx, counts in enumerate(analyzer.usage_counts):
            level_name = default_levels[idx] if idx < len(default_levels) else f"level_{idx}"
            usage_items.append((level_name, counts))

    if isinstance(num_embeddings, dict):
        embed_lookup = {str(level): int(val) for level, val in num_embeddings.items()}
        default_embed = next(iter(embed_lookup.values()), 0)
    else:
        embed_lookup = {}
        default_embed = int(num_embeddings) if num_embeddings is not None else 0

    for level, counts in usage_items:
        counts_tensor = counts if isinstance(counts, torch.Tensor) else torch.as_tensor(counts)
        if counts_tensor.numel() == 0:
            continue
        counts_cpu = counts_tensor.to(device="cpu", dtype=torch.float32)
        total_assignments = float(counts_cpu.sum().item())

        used = int((counts_cpu > 0).sum().item())
        active_counts = counts_cpu[counts_cpu > 0]
        mean_usage = float(active_counts.mean().item()) if active_counts.numel() > 0 else 0.0

        embeddings_for_level = embed_lookup.get(level, default_embed)
        dead_ratio = 1.0
        if embeddings_for_level:
            dead_ratio = 1.0 - (used / embeddings_for_level)

        stats[f"{level}_dead_code_ratio"] = dead_ratio
        stats[f"{level}_avg_active_usage"] = mean_usage
        stats[f"{level}_total_assignments"] = total_assignments

    return stats


@torch.no_grad()
def run_evaluation(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    pin_memory_flag = args.pin_memory
    if pin_memory_flag is None:
        pin_memory = device.type == "cuda"
    else:
        pin_memory = pin_memory_flag

    if device.type == "cuda" and not args.no_benchmark:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True

    amp_enabled = device.type == "cuda" and (args.amp or args.fast_eval)

    lpips_interval = max(1, args.lpips_interval)
    semantic_interval = max(1, args.semantic_interval)
    if args.fast_eval:
        lpips_interval = max(lpips_interval, 2)
        semantic_interval = max(semantic_interval, 2)

    cfg = Config()
    cfg.device = str(device)
    cfg.data_path = args.data_root
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.image_size = args.image_size
    cfg.pin_memory = pin_memory

    active_attributes = list(SELECTED_ATTRIBUTES)
    classifier: Optional[torch.nn.Module] = None
    classifier_attr_count = len(active_attributes)

    if args.classifier_ckpt:
        try:
            classifier, classifier_attr_count = load_classifier(Path(args.classifier_ckpt), device)
            if classifier_attr_count < len(active_attributes):
                dropped = active_attributes[classifier_attr_count:]
                print(
                    f"Info: classifier outputs {classifier_attr_count} attributes; "
                    "truncating attribute list to match checkpoint."
                )
                if dropped:
                    print(f"Info: dropping attributes {dropped} for semantic evaluation.")
                active_attributes = active_attributes[:classifier_attr_count]
            elif classifier_attr_count > len(active_attributes):
                raise ValueError(
                    "Classifier checkpoint expects more attributes than SELECTED_ATTRIBUTES provides."
                )
        except Exception as err:
            print(f"Warning: failed to load classifier from {args.classifier_ckpt}: {err}")
            classifier = None
            classifier_attr_count = len(active_attributes)

    dataset = build_dataset(cfg, Path(args.attribute_csv), active_attributes)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = load_hvq_model(Path(args.model_ckpt), cfg, device)

    lpips_fn = lpips.LPIPS(net=args.lpips_backbone).to(device)

    out_dir = Path(args.output_dir)
    recon_dir = out_dir / "reconstructions"
    heatmap_dir = out_dir / "heatmaps"
    interp_dir = out_dir / "interpolations"
    embed_dir = out_dir / "embeddings"
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(interp_dir, exist_ok=True)
    os.makedirs(embed_dir, exist_ok=True)

    writer = None
    if args.tensorboard and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))
    elif args.tensorboard:
        print("TensorBoard requested but not available in environment.")

    wandb_run = None
    if args.wandb_project:
        if wandb is None:
            print("Weights & Biases requested but package not installed.")
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run,
                config=vars(args),
                dir=str(out_dir),
            )

    metrics_calc = MetricsCalculator(device=device)
    codebook_analyzer = CodebookAnalyzer(num_embeddings=cfg.num_embeddings, num_levels=3)

    fid_metric = None
    kid_metric = None
    if FrechetInceptionDistance is not None:
        try:
            fid_metric = FrechetInceptionDistance(feature=2048, reset_real_features=False).to(device)
        except ModuleNotFoundError as err:
            print(f"Warning: skipping FID metric because required dependency is missing: {err}")
    else:
        print("FID metric unavailable (torchmetrics not installed).")
    if KernelInceptionDistance is not None:
        try:
            kid_metric = KernelInceptionDistance(subset_size=1000, subset_replacement=False).to(device)
        except (TypeError, ValueError) as err:
            print(
                "Info: KID metric version does not support subset_replacement; "
                "retrying with default arguments."
            )
            kid_metric = KernelInceptionDistance(subset_size=1000).to(device)
        except ModuleNotFoundError as err:
            print(f"Warning: skipping KID metric because required dependency is missing: {err}")
    else:
        print("KID metric unavailable (torchmetrics not installed).")

    semantic_results: Dict[str, List[float]] = {
        "attr_consistency": [],
        "attr_delta": [],
        "mutual_info": [],
    }

    reconstruction_metrics: Dict[str, List[float]] = {
        "psnr": [],
        "ssim": [],
        "lpips": [],
        "mse": [],
        "l1": [],
    }

    perplexities: Dict[str, List[float]] = {"top": [], "mid": [], "bottom": []}
    classifier_norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    stored_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    autocast_kwargs = {"device_type": "cuda", "dtype": torch.float16} if amp_enabled else None

    progress = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating")
    for step, batch in progress:
        images = batch["image"].to(device, non_blocking=pin_memory)
        attributes = batch["attributes"].to(device, non_blocking=pin_memory)

        if autocast_kwargs is not None:
            with torch.autocast(**autocast_kwargs):
                recon, _, codes = model(images)
        else:
            recon, _, codes = model(images)

        recon = recon.float()
        images_fp32 = images if images.dtype == torch.float32 else images.float()
        codes_top, codes_mid, codes_bottom = codes

        metrics = metrics_calc.compute_metrics(images_fp32, recon, step)
        reconstruction_metrics["psnr"].append(metrics["psnr"])
        reconstruction_metrics["ssim"].append(metrics["ssim"])
        reconstruction_metrics["mse"].append(metrics["mse"])
        reconstruction_metrics["l1"].append(metrics["l1"])

        lpips_val: Optional[float] = None
        if step % lpips_interval == 0:
            lpips_val = lpips_fn(images_fp32, recon).mean().item()
            reconstruction_metrics["lpips"].append(lpips_val)

        denorm_real: Optional[torch.Tensor] = None
        denorm_recon: Optional[torch.Tensor] = None

        if fid_metric is not None:
            denorm_real = denormalize(images_fp32)
            denorm_recon = denormalize(recon)
            fid_metric.update(to_uint8_from_zero_one(denorm_real), real=True)
            fid_metric.update(to_uint8_from_zero_one(denorm_recon), real=False)
        if kid_metric is not None:
            if denorm_real is None:
                denorm_real = denormalize(images_fp32)
                denorm_recon = denormalize(recon)
            kid_metric.update(to_uint8_from_zero_one(denorm_real), real=True)
            kid_metric.update(to_uint8_from_zero_one(denorm_recon), real=False)

        index_levels: List[Optional[torch.Tensor]] = []
        for name, code_tensor in zip(("top", "mid", "bottom"), (codes_top, codes_mid, codes_bottom)):
            if code_tensor is None:
                index_levels.append(None)
                continue
            flat_codes = code_tensor.view(-1, code_tensor.size(-1))
            perplexities[name].append(calculate_perplexity(flat_codes))
            index_levels.append(torch.argmax(flat_codes, dim=1))

        codebook_analyzer.update(tuple(index_levels))

        if step % args.visual_interval == 0:
            images_cpu = images_fp32.cpu()
            recon_cpu = recon.cpu()
            save_reconstruction_grid(images_cpu, recon_cpu, recon_dir / f"recon_step{step:05d}.png")
            save_difference_visual(images_cpu, recon_cpu, heatmap_dir, f"step{step:05d}")
            stored_batches.append((images_cpu, recon_cpu))

        if classifier is not None and step % semantic_interval == 0:
            if denorm_real is None:
                denorm_real = denormalize(images_fp32)
                denorm_recon = denormalize(recon)
            cls_input_real = classifier_norm(denorm_real)
            cls_input_recon = classifier_norm(denorm_recon)
            preds_real = torch.sigmoid(classifier(cls_input_real))
            preds_recon = torch.sigmoid(classifier(cls_input_recon))

            consistency = 1.0 - (preds_real - preds_recon).abs().mean().item()
            delta = (preds_recon - attributes).abs().mean().item()
            semantic_results["attr_consistency"].append(consistency)
            semantic_results["attr_delta"].append(delta)

            if index_levels[0] is not None:
                codes_per_image = index_levels[0].view(images_fp32.size(0), -1)
                codes_np = codes_per_image.cpu().numpy()
                attrs_np = attributes.cpu().numpy()
                mi_vals = []
                for attr_idx in range(attrs_np.shape[1]):
                    attr_vals = attrs_np[:, attr_idx]
                    labels = np.repeat(attr_vals, codes_per_image.shape[1])
                    mi_vals.append(mutual_info_score(codes_np.reshape(-1), labels))
                semantic_results["mutual_info"].append(float(np.mean(mi_vals)))

        if writer is not None:
            writer.add_scalar("reconstruction/psnr", metrics["psnr"], step)
            writer.add_scalar("reconstruction/ssim", metrics["ssim"], step)
            if lpips_val is not None:
                writer.add_scalar("reconstruction/lpips", lpips_val, step)

        if wandb_run is not None:
            log_payload = {
                "step": step,
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
            }
            if lpips_val is not None:
                log_payload["lpips"] = lpips_val
            wandb.log(log_payload)

    avg_metrics = {name: float(np.mean(values)) for name, values in reconstruction_metrics.items() if values}
    avg_perplexity = {name: float(np.mean(vals)) for name, vals in perplexities.items() if vals}
    code_metrics = codebook_analyzer.get_diversity_metrics()
    code_metrics.update(summarize_codebook_usage(codebook_analyzer, cfg.num_embeddings))

    summary = {
        "reconstruction": avg_metrics,
        "perplexity": avg_perplexity,
        "codebook": code_metrics,
        "semantic": {name: float(np.mean(vals)) for name, vals in semantic_results.items() if vals},
    }

    if fid_metric is not None:
        summary["fid"] = float(fid_metric.compute())
    if kid_metric is not None:
        kid_mean, kid_std = kid_metric.compute()
        summary["kid_mean"] = float(kid_mean)
        summary["kid_std"] = float(kid_std)

    with open(out_dir / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    if wandb_run is not None:
        flat_summary: Dict[str, float] = {}

        def _flatten(prefix: str, data: Dict[str, object]) -> None:
            for key, value in data.items():
                name = f"{prefix}/{key}" if prefix else str(key)
                if isinstance(value, dict):
                    _flatten(name, value)
                else:
                    flat_summary[name] = value

        _flatten("summary", summary)
        wandb_run.log(flat_summary)

    if writer is not None and "reconstruction" in summary:
        final_step = len(dataloader)
        for key, value in summary["reconstruction"].items():
            writer.add_scalar(f"reconstruction/avg_{key}", value, final_step)

    log_embedding_diagnostics(model, codebook_analyzer.usage_counts, embed_dir)

    if stored_batches:
        generate_interpolations(model, stored_batches[0][0], interp_dir, device)

    if writer is not None:
        writer.close()
    if wandb_run is not None:
        wandb_run.finish()


def interpolate_codes(model: HVQVAE_3Level, image_a: torch.Tensor, image_b: torch.Tensor, steps: int, device: torch.device) -> torch.Tensor:
    image_a = image_a.unsqueeze(0).to(device)
    image_b = image_b.unsqueeze(0).to(device)

    with torch.no_grad():
        feat_b_a = model.enc_b(image_a)
        feat_b_b = model.enc_b(image_b)
        feat_m_a = model.enc_m(feat_b_a)
        feat_m_b = model.enc_m(feat_b_b)
        feat_t_a = model.enc_t(feat_m_a)
        feat_t_b = model.enc_t(feat_m_b)

        qt_a, _, _ = model.quant_t(model.quant_conv_t(feat_t_a))
        qt_b, _, _ = model.quant_t(model.quant_conv_t(feat_t_b))
        qm_a, _, _ = model.quant_m(model.quant_conv_m(feat_m_a))
        qm_b, _, _ = model.quant_m(model.quant_conv_m(feat_m_b))
        qb_a, _, _ = model.quant_b(model.quant_conv_b(feat_b_a))
        qb_b, _, _ = model.quant_b(model.quant_conv_b(feat_b_b))

        blends = []
        for alpha in np.linspace(0, 1, steps):
            qt = torch.lerp(qt_a, qt_b, alpha)
            qm = torch.lerp(qm_a, qm_b, alpha)
            qb = torch.lerp(qb_a, qb_b, alpha)
            blended = model.decode_codes(qt, qm, qb)
            blends.append(blended.squeeze(0).cpu())
    return torch.stack(blends)

def generate_interpolations(model: HVQVAE_3Level, originals: torch.Tensor, save_dir: Path, device: torch.device) -> None:
    os.makedirs(save_dir, exist_ok=True)
    num_pairs = min(3, originals.shape[0])
    indices = list(range(num_pairs))
    for idx_a, idx_b in itertools.combinations(indices, 2):
        interp = interpolate_codes(model, originals[idx_a], originals[idx_b], steps=8, device=device)
        grid = make_grid(denormalize(interp), nrow=interp.shape[0], padding=2)
        save_image(grid, save_dir / f"interp_{idx_a}_{idx_b}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent space testing harness")
    parser.add_argument("--model-ckpt", type=str, required=True, help="Path to trained HVQ checkpoint")
    parser.add_argument("--classifier-ckpt", type=str, default="", help="Optional attribute classifier")
    parser.add_argument("--data-root", type=str, required=True, help="Directory with CelebA split")
    parser.add_argument("--attribute-csv", type=str, required=True, help="Path to attribute CSV")
    parser.add_argument("--output-dir", type=str, default="lspace_eval", help="Directory for reports")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", help="Force DataLoader pinned memory.")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false", help="Disable DataLoader pinned memory.")
    parser.set_defaults(pin_memory=None)
    parser.add_argument("--visual-interval", type=int, default=50)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="")
    parser.add_argument("--wandb-run", type=str, default="")
    parser.add_argument("--lpips-backbone", type=str, default="alex", choices=["alex", "vgg", "squeeze"]) 
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision on CUDA.")
    parser.add_argument("--lpips-interval", type=int, default=1, help="Compute LPIPS every N batches.")
    parser.add_argument("--semantic-interval", type=int, default=1, help="Run classifier-based metrics every N batches.")
    parser.add_argument("--fast-eval", action="store_true", help="Shorthand to relax expensive metrics for speed.")
    parser.add_argument("--no-benchmark", action="store_true", help="Disable cudnn.benchmark even on CUDA.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":  # pragma: no cover
    main()
