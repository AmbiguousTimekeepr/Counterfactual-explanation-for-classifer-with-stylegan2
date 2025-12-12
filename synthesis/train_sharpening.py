import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
import lpips
from torchvision.utils import save_image

from .config import Config
from .generator import CounterfactualGenerator
from .dataset import get_loader
from .discriminator_manager import PatchGANDiscriminator
from .utils_ema import EMA


def _prepare_visualization_samples(cfg, device, train_loader):
    """Get a small batch of images for epoch-end visualization."""
    try:
        sample_loader = get_loader(cfg, split="val", batch_size=4, shuffle=False)
        sample_imgs, _ = next(iter(sample_loader))
    except Exception:
        sample_imgs, _ = next(iter(train_loader))
    return sample_imgs[:4].to(device)


def _save_epoch_visualization(model, ema, sample_imgs, epoch, vis_dir, device):
    """Run decoder inference on held-out samples and save side-by-side grid."""
    vis_dir.mkdir(parents=True, exist_ok=True)
    ema_model = ema.model
    ema_model.eval()

    with torch.no_grad():
        z_list = model.vqvae.get_codes(sample_imgs)
        recon = ema_model(z_list)

    grid_imgs = torch.cat([sample_imgs, recon], dim=0)
    save_path = vis_dir / f"epoch_{epoch:03d}.png"
    save_image(grid_imgs, save_path, nrow=sample_imgs.size(0), normalize=True, value_range=(-1, 1))


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return loss_real + loss_fake


def hinge_g_loss(logits_fake):
    return -torch.mean(logits_fake)


def train_sharpening(checkpoint_path: str | None = None, cfg: Config | None = None):
    cfg = cfg or Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CounterfactualGenerator(
        cfg,
        vqvae_path=cfg.vqvae_checkpoint,
        classifier_path=cfg.classifier_checkpoint,
        device=device,
    )

    if checkpoint_path is None:
        checkpoint_path = getattr(cfg, "decoder_checkpoint_path", "outputs/stylegan_decoder/latest.pth")
    decoder_path = Path(checkpoint_path)
    if decoder_path.is_file():
        print(f"🔄 Loading pre-trained decoder for sharpening: {decoder_path}")
        model.decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    else:
        raise FileNotFoundError(
            f"Decoder checkpoint not found at '{decoder_path}'. Run decoder pre-training first."
        )

    discriminator = PatchGANDiscriminator().to(device)
    ema = EMA(model.decoder, decay=0.999)

    opt_g = optim.Adam(model.decoder.parameters(), lr=cfg.g_sharpening_lr, betas=cfg.sharpening_betas)
    opt_d = optim.Adam(discriminator.parameters(), lr=cfg.d_sharpening_lr, betas=cfg.sharpening_betas)

    scaler = GradScaler()
    loss_l1 = torch.nn.L1Loss()
    loss_lpips = lpips.LPIPS(net="vgg").to(device).eval()

    loader = get_loader(cfg, split="train", batch_size=8)
    sample_imgs = _prepare_visualization_samples(cfg, device, loader)

    checkpoint_root = getattr(cfg, "sharpening_checkpoint_dir", "") or "outputs/synth_network/stylegan_decoder_sharpened"
    vis_dir = Path(checkpoint_root) / "samples"

    print("🚀 Starting adversarial sharpening phase...")

    for epoch in range(cfg.sharpening_epochs):
        pbar = tqdm(loader, desc=f"Sharpening Epoch {epoch}")
        for imgs, _ in pbar:
            imgs = imgs.to(device)

            opt_d.zero_grad()
            with autocast():
                with torch.no_grad():
                    z_list = model.vqvae.get_codes(imgs)
                    fake_imgs = model.decoder(z_list)

                logits_real = discriminator(imgs)
                logits_fake = discriminator(fake_imgs.detach())
                d_loss = hinge_d_loss(logits_real, logits_fake)

            if d_loss.item() > 0.2:
                scaler.scale(d_loss).backward()
                scaler.step(opt_d)
                scaler.update()
            else:
                opt_d.zero_grad(set_to_none=True)

            opt_g.zero_grad()
            with autocast():
                fake_imgs_g = model.decoder(z_list)
                logits_fake_g = discriminator(fake_imgs_g)

                g_adv = hinge_g_loss(logits_fake_g)
                l1 = loss_l1(fake_imgs_g, imgs)
                perc = loss_lpips(fake_imgs_g, imgs).mean()

                loss_g = g_adv * 0.1 + l1 * 10.0 + perc * 1.0

            scaler.scale(loss_g).backward()
            scaler.step(opt_g)
            scaler.update()

            ema.update(model.decoder)

            pbar.set_postfix({
                "D": d_loss.item(),
                "G_Adv": g_adv.item(),
                "L1": l1.item(),
                "LPIPS": perc.item(),
            })

        if epoch % 2 == 0:
            sharp_dir = Path(checkpoint_root)
            sharp_dir.mkdir(parents=True, exist_ok=True)
            torch.save(ema.model.state_dict(), sharp_dir / f"sharp_epoch_{epoch}.pth")
            torch.save(ema.model.state_dict(), sharp_dir / "latest_sharp.pth")
            print(f"💾 Saved EMA decoder weights to {sharp_dir}")

        _save_epoch_visualization(model, ema, sample_imgs, epoch, vis_dir, device)


if __name__ == "__main__":
    train_sharpening()
