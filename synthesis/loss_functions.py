import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
import pytorch_ssim

LAMBDA_ADV = 1.0
LAMBDA_CF = 8.0
LAMBDA_RETENTION = 12.0
LAMBDA_MASK_SIM = 6.0
LAMBDA_LATENT_PROX = 4.0
LAMBDA_CONSISTENCY = 2.0
LAMBDA_ORTHO = 0.2
LAMBDA_VQ = 25e-5

class LossFunctions:
    def __init__(self, device='cuda'):
        self.device = device
        self.lpips_net = lpips.LPIPS(net='alex', version='0.1').to(device)
        self.ssim = pytorch_ssim.SSIM(window_size=11).to(device)

    # ===============================================================
    # === DISCRIMINATOR LOSSES ======================================
    # ===============================================================
    def r1_regularizer(self, discriminator, real_images, r1_gamma=10.0):
        real_images.requires_grad_(True)
        pred = discriminator(real_images)
        grads = torch.autograd.grad(pred.sum(), real_images, create_graph=True)[0]
        return (grads.norm(2, dim=[1,2,3]) ** 2).mean() * (r1_gamma / 2)

    def relative_pairing_discriminator_loss(self, D, real_imgs, fake_imgs, r1_gamma=10.0):
        real_pred = D(real_imgs)
        fake_pred = D(fake_imgs.detach())
        rp_loss = F.relu(1.0 - (real_pred - fake_pred)).mean()
        r1 = self.r1_regularizer(D, real_imgs, r1_gamma)
        return rp_loss + r1, {'rp_loss': rp_loss, 'r1': r1}

    def adversarial_rp_loss(self, D, real_imgs, fake_imgs):
        real_pred = D(real_imgs)
        fake_pred = D(fake_imgs)
        return F.relu(1.0 - (fake_pred - real_pred)).mean()

    # ===============================================================
    # === GENERATOR LOSSES ==========================================
    # ===============================================================

    def counterfactual_loss(self, pred_probs, target_attrs, margin=0.8, already_correct_weight=0.2):
        diff = (target_attrs.round() != pred_probs.round()).float()
        if diff.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        # BCE with margin
        target_for_loss = target_attrs * margin + (1 - target_attrs) * (1 - margin)
        loss = F.binary_cross_entropy(pred_probs, target_for_loss, reduction='none')
        loss = (loss * diff).sum() / (diff.sum() + 1e-8)

        # Small penalty if already correct (prevents collapse)
        correct_penalty = F.binary_cross_entropy(pred_probs, target_attrs, reduction='none')
        correct_penalty = (correct_penalty * (1 - diff)).mean() * already_correct_weight

        return loss + correct_penalty

    def retention_loss(self, fake, real, alpha_vec, spatial_mask=None, weight_outside=3.0):
        """Strong L1 retention outside edited regions"""
        alpha = alpha_vec.mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)  # [B,1,1,1]
        base_weight = 1.0 - alpha

        if spatial_mask is not None:
            mask = F.interpolate(spatial_mask, size=real.shape[2:], mode='bilinear', align_corners=False)
            weight = base_weight * (1 - mask) * weight_outside + base_weight * mask
        else:
            weight = base_weight

        return F.l1_loss(fake * weight, real * weight)

    def masked_similarity_loss(self, fake, real, spatial_mask=None, alpha_vec=None, outside_weight=5.0):
        if spatial_mask is None:
            return torch.tensor(0.0, device=self.device)

        mask = F.interpolate(spatial_mask, size=real.shape[2:], mode='bilinear', align_corners=False)
        outside = (1 - mask).clamp(min=0.0)

        if alpha_vec is not None:
            alpha = alpha_vec.mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
            outside = outside * (1 - alpha)

        return F.l1_loss(fake * outside, real * outside) * outside_weight

    def latent_proximity_loss(self, z_orig_list, z_edited_list, alpha_vec, spatial_mask=None, outside_weight=3.0):
        """Penalize latent deviation, stronger outside edited regions"""
        loss = 0.0
        for z_o, z_e, alpha_level in zip(z_orig_list, z_edited_list, alpha_vec.t()):
            diff = (z_e - z_o).abs()
            alpha_level = alpha_level.view(-1, 1, 1, 1)

            if spatial_mask is not None:
                mask = F.interpolate(spatial_mask, size=z_o.shape[2:], mode='bilinear', align_corners=False)
                weight = (1 - mask) * outside_weight + mask
                loss += (diff * weight * (1 - alpha_level) * 2.0 + diff * alpha_level).mean()
            else:
                loss += (diff * (1 - alpha_level) * outside_weight).mean()

        return loss / len(z_orig_list)

    def consistency_loss(self, fake_images, loss_type='lpips'):
        flipped = fake_images.flip(3)
        if loss_type == 'lpips':
            return self.lpips_net(fake_images, flipped).mean()
        elif loss_type == 'l2':
            return F.mse_loss(fake_images, flipped)
        elif loss_type == 'ssim':
            return 1.0 - self.ssim(fake_images, flipped)
        else:
            raise ValueError(f"Unknown consistency loss: {loss_type}")

    def orthogonality_penalty(self, latent_editor):
        D = latent_editor.directions  # [39, embed_dim]
        normed = F.normalize(D, dim=1)
        gram = torch.mm(normed, normed.t())
        off_diag = gram - torch.diag(torch.diag(gram))
        return off_diag.abs().mean()

    def generator_total_loss(
        self,
        discriminator,
        generator,
        real_images,
        fake_images,
        target_attrs,
        alpha_vec,
        gen_info,
        spatial_mask=None,
        lambda_adv=LAMBDA_ADV,
        lambda_cf=LAMBDA_CF,
        lambda_ret=LAMBDA_RETENTION,
        lambda_mask_sim=LAMBDA_MASK_SIM,
        lambda_latent_prox=LAMBDA_LATENT_PROX,
        lambda_consistency=LAMBDA_CONSISTENCY,
        lambda_ortho=LAMBDA_ORTHO,
        lambda_vq=LAMBDA_VQ,
    ):
        # 1. Adversarial
        g_adv = self.adversarial_rp_loss(discriminator, real_images, fake_images)

        # 2. Counterfactual (on generated image)
        with torch.no_grad():
            _, pred_probs = generator.classifier(fake_images), torch.sigmoid(generator.classifier(fake_images))
        g_cf = self.counterfactual_loss(pred_probs, target_attrs)

        # 3. Retention + Masked Similarity
        g_ret = self.retention_loss(fake_images, real_images, alpha_vec, spatial_mask)
        g_mask_sim = self.masked_similarity_loss(fake_images, real_images, spatial_mask, alpha_vec)

        # 4. Latent Proximity
        g_latent = self.latent_proximity_loss(
            gen_info['z_orig'], gen_info['z_edited'], alpha_vec, spatial_mask
        )

        # 5. Consistency
        g_cons = self.consistency_loss(fake_images, loss_type='lpips')

        # 6. VQ
        g_vq = gen_info.get('vq_loss', 0.0) * lambda_vq
        g_ortho = self.orthogonality_penalty(generator.mutator)
        total = (
            lambda_adv * g_adv +
            lambda_cf * g_cf +
            lambda_ret * g_ret +
            lambda_mask_sim * g_mask_sim +
            lambda_latent_prox * g_latent +
            lambda_consistency * g_cons +
            g_vq +
            lambda_ortho * g_ortho
        )
        return {
            'total_loss': total,
            'adv': g_adv,
            'cf': g_cf,
            'retention': g_ret,
            'masked_sim': g_mask_sim,
            'latent_prox': g_latent,
            'consistency': g_cons,
            'vq': g_vq,
            'ortho': g_ortho,
            'alpha_mean': alpha_vec.mean().item(),
        }

# ===============================================================
# === TRAINING MANAGER (clean wrapper)
# ===============================================================
class TrainingLossManager:
    def __init__(self, device='cuda'):
        self.loss_fn = LossFunctions(device)
        self.device = device

    def discriminator_step(self, D, real, fake, r1_gamma=10.0):
        return self.loss_fn.relative_pairing_discriminator_loss(D, real, fake, r1_gamma)

    def generator_step(self, D, G, real, fake, target_attrs, alpha_vec, gen_info, spatial_mask=None):
        return self.loss_fn.generator_total_loss(
            discriminator=D,
            generator=G,
            real_images=real,
            fake_images=fake,
            target_attrs=target_attrs,
            alpha_vec=alpha_vec,
            gen_info=gen_info,
            spatial_mask=spatial_mask,
        )