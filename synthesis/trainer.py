import torch
import os
from pathlib import Path
from torchvision.utils import save_image, make_grid

def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)

def save_samples(x_real, x_fake, step, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    x_real = denorm(x_real)
    x_fake = denorm(x_fake)
    grid = make_grid(torch.cat([x_real, x_fake], dim=0), nrow=x_real.size(0), padding=8, pad_value=1)
    save_image(grid, f"{save_dir}/step_{step:06d}_cf.png")
    # for i in range(x_real.size(0)):
    #     pair = torch.cat([x_real[i:i+1], x_fake[i:i+1]], dim=3)
    #     save_image(pair, os.path.join(save_dir, f"step_{step:06d}_sample_{i:02d}.png"))

def save_checkpoint(generator, discriminator, opt_g, opt_d, step, history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'step': step,
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'opt_g': opt_g.state_dict(),
        'opt_d': opt_d.state_dict(),
        'history': history
    }
    torch.save(checkpoint, os.path.join(save_dir, f"step_{step:06d}.pth"))
    torch.save(checkpoint, os.path.join(save_dir, "latest.pth"))

def train_counterfactual(
    generator, discriminator, opt_g, opt_d, loss_manager, dataloader,
    num_steps, device, sample_dir="outputs/CF_training_samples", ckpt_dir="outputs/CF_training_checkpoints", log_interval=500
):
    generator.train()
    discriminator.train()
    sample_dir = Path(sample_dir)
    ckpt_dir = Path(ckpt_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history = {'g_loss': [], 'd_loss': [], 'cf_loss': [], 'ret_loss': [], 'adv_loss': []}
    data_iter = iter(dataloader)

    for step in range(1, num_steps + 1):
        try:
            x_real, attrs_real = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x_real, attrs_real = next(data_iter)
        x_real = x_real.to(device)
        attrs_real = attrs_real.to(device)
        batch_size = x_real.size(0)
        z_noise = torch.randn(batch_size, 512, device=device)
        # Sample alpha_vec randomly between 0 and 1 for each batch and level
        alpha_vec = torch.rand(batch_size, 3, device=device)
        # Randomly flip up to 4 attributes per sample
        attrs_target = attrs_real.clone()
        for i in range(batch_size):
            k = torch.randint(1, 5, (1,)).item()
            idx = torch.randperm(attrs_real.size(1), device=device)[:k]
            attrs_target[i, idx] = 1.0 - attrs_target[i, idx]

        # Generator forward
        fake, gen_info = generator(x_real, z_noise, alpha_vec, attrs_target)
        # Classifier outputs
        with torch.no_grad():
            logits_real = generator.classifier(x_real)
            probs_real = torch.sigmoid(logits_real)
            logits_fake = generator.classifier(fake)
            probs_fake = torch.sigmoid(logits_fake)

        # Discriminator step
        opt_d.zero_grad()
        d_loss, d_dict = loss_manager.discriminator_step(discriminator, x_real, fake)
        d_loss.backward()
        opt_d.step()

        # Generator step
        opt_g.zero_grad()
        g_dict = loss_manager.generator_step(
            discriminator, generator, x_real, fake, attrs_target, alpha_vec, gen_info
        )
        g_loss = g_dict['total_loss']
        g_loss.backward()
        opt_g.step()

        # Logging
        history['g_loss'].append(g_loss.item())
        history['d_loss'].append(d_loss.item())
        history['cf_loss'].append(g_dict['cf'].item())
        history['ret_loss'].append(g_dict['retention'].item())
        history['adv_loss'].append(g_dict['adv'].item())

        if step % log_interval == 0 or step == 1:
            print(f"Step {step:5d} | D: {d_loss.item():.3f} | G: {g_loss.item():.3f} | "
                  f"CF: {g_dict['cf'].item():.3f} | Ret: {g_dict['retention'].item():.3f} | Adv: {g_dict['adv'].item():.3f}")
            save_samples(x_real[:8], fake[:8], step, sample_dir)
            save_checkpoint(generator, discriminator, opt_g, opt_d, step, history, ckpt_dir)

    # Final save
    save_samples(x_real[:8], fake[:8], num_steps, sample_dir)
    save_checkpoint(generator, discriminator, opt_g, opt_d, num_steps, history, ckpt_dir)
    print("Training complete. Final samples and checkpoint saved.")
    return history
