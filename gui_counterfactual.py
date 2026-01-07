import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import gradio as gr
from torchvision import transforms

from src.classifiers.integrated_gradients import integrated_gradients_batch
from src.synthesis.config import Config
from src.synthesis.dataset import SELECTED_ATTRIBUTES
from src.synthesis.generator import CounterfactualGenerator

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ATTR_OPTIONS = ["Brown_Hair", "Mouth_Slightly_Open", "Eyeglasses"]
ATTR_DISPLAY = {
    "Brown_Hair": "Brown Hair",
    "Mouth_Slightly_Open": "Mouth Slightly Open",
    "Eyeglasses": "Eyeglasses",
}

VQ_CHECKPOINT = Path("outputs/checkpoints_production/checkpoints/checkpoint_step_22500.pth")
CLASSIFIER_CHECKPOINT = Path("outputs/cnn_classfier/resnet18_cbam_epoch_10_128.pth")
DECODER_CHECKPOINT = Path("outputs/synth_network/stylegan_decoder/latest.pth")
MUTATOR_TEMPLATE = (
    "outputs/synth_network/CF_generator/ceGAN_counterfactual_{attr}_final/checkpoints/epoch_019.pth"
)
IMAGE_SIZE = 128
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

generator_cache: dict[str, CounterfactualGenerator] = {}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def load_image_from_pil(image: Image.Image, device: torch.device, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """Resize, normalize, and move image to device."""
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform(image.convert("RGB")).unsqueeze(0).to(device)


def compute_saliency(
    classifier: torch.nn.Module,
    image_tensor: torch.Tensor,
    attr_idx: int,
    target_class: int,
    ig_steps: int,
    mask_threshold: float,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Compute IG saliency map and multi-scale masks."""
    ig_attr = integrated_gradients_batch(
        model=classifier,
        input_batch=image_tensor,
        attribute_idx=attr_idx,
        target_classes=torch.tensor([int(target_class)], device=image_tensor.device),
        steps=ig_steps,
        device=image_tensor.device,
    )

    raw_saliency = torch.abs(ig_attr).sum(dim=1, keepdim=True)
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
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    mask_m = F.interpolate(
        binary_mask.unsqueeze(0).unsqueeze(0),
        size=(16, 16),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    mask_b = F.interpolate(
        binary_mask.unsqueeze(0).unsqueeze(0),
        size=(32, 32),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    return ig_map.detach(), [mask_t.detach(), mask_m.detach(), mask_b.detach()], norm_map.detach()


def to_numpy_image(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu().clamp(0, 1)[0]
    return (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def create_overlay(base_img: np.ndarray, saliency_map: torch.Tensor, alpha: float = 0.4) -> np.ndarray:
    import cv2

    heatmap = cv2.applyColorMap(
        (saliency_map.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = alpha * heatmap + (1 - alpha) * base_img.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def get_attr_idx(attr_name: str) -> int:
    return SELECTED_ATTRIBUTES.index(attr_name)


def build_config(active_attr: str) -> Config:
    cfg = Config()
    cfg.vqvae_checkpoint = str(VQ_CHECKPOINT)
    cfg.classifier_checkpoint = str(CLASSIFIER_CHECKPOINT)
    cfg.active_attributes = [active_attr]
    cfg.max_active_attributes_per_epoch = 1
    return cfg


def load_generator_for_attr(attr_name: str) -> CounterfactualGenerator:
    if attr_name in generator_cache:
        return generator_cache[attr_name]

    cfg = build_config(attr_name)
    gen = CounterfactualGenerator(cfg, vqvae_path=cfg.vqvae_checkpoint, classifier_path=cfg.classifier_checkpoint, device=DEVICE)

    if DECODER_CHECKPOINT.is_file():
        gen.load_decoder_checkpoint(DECODER_CHECKPOINT, map_location=DEVICE)
    else:
        raise FileNotFoundError(f"Decoder checkpoint missing: {DECODER_CHECKPOINT}")

    mutator_ckpt = Path(MUTATOR_TEMPLATE.format(attr=attr_name.lower()))
    if not mutator_ckpt.is_file():
        raise FileNotFoundError(f"Mutator checkpoint missing for {attr_name}: {mutator_ckpt}")

    ckpt = torch.load(mutator_ckpt, map_location=DEVICE)
    if "mutator_state" in ckpt:
        gen.mutator.load_state_dict(ckpt["mutator_state"], strict=True)
    else:
        gen.mutator.load_state_dict(ckpt, strict=False)

    decoder_state = ckpt.get("decoder_state") if isinstance(ckpt, dict) else None
    if decoder_state is not None:
        gen.decoder.load_state_dict(decoder_state, strict=False)

    gen.eval()
    gen.mutator.eval()

    generator_cache[attr_name] = gen
    return gen


def format_scores(probs_orig: torch.Tensor, probs_cf: torch.Tensor) -> pd.DataFrame:
    rows = []
    for attr in ATTR_OPTIONS:
        idx = get_attr_idx(attr)
        p_o = float(probs_orig[idx].item())
        p_c = float(probs_cf[idx].item())
        rows.append(
            {
                "Attribute": ATTR_DISPLAY.get(attr, attr),
                "Original": round(p_o, 3),
                "Counterfactual": round(p_c, 3),
                "Delta": round(p_c - p_o, 3),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Inference pipeline
# -----------------------------------------------------------------------------
def run_counterfactual(image: Image.Image, attribute_choice: str):
    if image is None:
        raise gr.Error("Vui lòng chọn ảnh đầu vào.")

    attr_name = attribute_choice
    attr_idx = get_attr_idx(attr_name)
    generator = load_generator_for_attr(attr_name)

    image_tensor = load_image_from_pil(image, DEVICE, image_size=IMAGE_SIZE)

    with torch.no_grad():
        logits = generator.classifier(image_tensor)
        probs = torch.sigmoid(logits)
        base_labels = (probs > 0.5).float()

    target_labels = base_labels.clone()
    target_labels[:, attr_idx] = 1 - target_labels[:, attr_idx]
    target_class = int(target_labels[0, attr_idx].item())

    ig_map, masks_ig, cam_soft = compute_saliency(
        generator.classifier,
        image_tensor,
        attr_idx=attr_idx,
        target_class=target_class,
        ig_steps=getattr(generator, "ig_steps", 16),
        mask_threshold=getattr(generator, "cam_threshold", 0.35),
    )

    ig_batch = ig_map.unsqueeze(0)
    masks_batch_ig = [m.unsqueeze(0) for m in masks_ig]

    with torch.no_grad():
        x_cf_ig, _, _ = generator(
            image_tensor,
            ig_batch,
            masks_batch_ig,
            target_labels,
            probs,
            attr_idx,
            hard=True,
        )

        probs_orig_all = torch.sigmoid(generator.classifier(image_tensor))[0]
        probs_cf_all = torch.sigmoid(generator.classifier(x_cf_ig))[0]

    orig_disp = to_numpy_image((image_tensor * 0.5) + 0.5)
    cf_disp = to_numpy_image((x_cf_ig * 0.5) + 0.5)
    overlay_disp = create_overlay(orig_disp, cam_soft)
    scores_df = format_scores(probs_orig_all, probs_cf_all)

    return orig_disp, overlay_disp, cf_disp, scores_df


# -----------------------------------------------------------------------------
# Gradio UI
# -----------------------------------------------------------------------------
description_md = """
### Counterfactual Image Demo
- Chọn attribute muốn lật (Brown Hair, Mouth Slightly Open, Eyeglasses).
- Tải lên ảnh khuôn mặt 128x128 (ảnh khác sẽ được resize).
- Xem ảnh gốc, saliency (IG), ảnh phản thực, và confidence score của classifier.
"""

with gr.Blocks(title="Counterfactual Generator") as demo:
    gr.Markdown(description_md)

    with gr.Row():
        image_in = gr.Image(label="Ảnh đầu vào", type="pil")
        attr_radio = gr.Radio(
            choices=ATTR_OPTIONS,
            value="Mouth_Slightly_Open",
            label="Attribute",
            info="Attribute sẽ được flip",
        )

    run_btn = gr.Button("Sinh ảnh phản thực", variant="primary")

    with gr.Row():
        orig_out = gr.Image(label="Ảnh gốc", type="numpy")
        sal_out = gr.Image(label="Saliency (IG)", type="numpy")
        cf_out = gr.Image(label="Ảnh phản thực", type="numpy")

    scores = gr.Dataframe(label="Confidence", interactive=False)

    run_btn.click(run_counterfactual, inputs=[image_in, attr_radio], outputs=[orig_out, sal_out, cf_out, scores])


if __name__ == "__main__":
    demo.launch()
