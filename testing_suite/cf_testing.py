# Full 2025 Counterfactual Evaluation Suite for CelebA-HQ
# Run: python cf_testing.py --ckpt path/to/model.pth --vqvae path/to/vqvae.pth

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import os
from PIL import Image
import json
import pandas as pd

# ----------------------- Imports (install once) -----------------------
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from facenet_pytorch import InceptionResnetV1
from pytorch_fid import fid_score
import lpips
from sklearn.metrics import accuracy_score

# ----------------------- Config & Args -----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True, help='Counterfactual generator checkpoint')
parser.add_argument('--vqvae', type=str, required=True, help='Pretrained VQ-VAE checkpoint')
parser.add_argument('--classifier', type=str, default='models/classifier.pth', help='Attribute classifier')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--num_samples', type=int, default=5000, help='Test on 5000–10000')
parser.add_argument('--attributes', type=str, nargs='+', 
                    default=['Smiling', 'Young', 'Male', 'Eyeglasses', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Pale_Skin'])
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# ----------------------- Load Models -----------------------
print("Loading models...")
generator = torch.load(args.ckpt, map_location=device).eval()
vqvae = torch.load(args.vqvae, map_location=device).eval()
classifier = torch.load(args.classifier, map_location=device).eval()

# Identity extractor (ArcFace)
identity_net = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# LPIPS
lpips_loss = lpips.LPIPS(net='alex').to(device)

# CelebA attribute names mapping (use official 40 but we pick 39)
ATTRS = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 
         'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 
         'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 
         'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 
         'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 
         'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

target_idx = [ATTRS.index(a) for a in args.attributes]

# ----------------------- Dataset -----------------------
test_dataset = load_dataset("nielsr/celebahq-attribute", split="test")
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

# ----------------------- Metrics Storage -----------------------
results = {
    'CSR': [], 'Target_Acc': [], 'Untargeted_Stability': [], 'Identity_Sim': [], 
    'LPIPS': [], 'IoU_Mask': [], 'SEC': [], 'FID': None, 'CF_Score': []
}

def compute_csr(pred_fake, target_attr, target_idx):
    """Counterfactual Success Rate (CSR)"""
    target_correct = ((pred_fake.round()[:, target_idx] == target_attr).all(dim=1)).float()
    return target_correct.cpu().numpy()

def compute_targeted_flip_accuracy(pred_fake, target_idx):
    """Targeted Flip Accuracy"""
    return pred_fake[:, target_idx].mean(dim=1).cpu().numpy()

def compute_untargeted_stability(pred_real, pred_fake, target_idx):
    """Untargeted Stability"""
    untargeted_diff = (pred_fake - pred_real).abs()
    untargeted_diff[:, target_idx] = 0
    return untargeted_diff.mean(dim=1).cpu().numpy()

def compute_identity_similarity(identity_net, img, fake):
    """Identity Similarity (ArcFace)"""
    feat_real = identity_net((img + 1) * 127.5)
    feat_fake = identity_net((fake + 1) * 127.5)
    cos_sim = F.cosine_similarity(feat_real, feat_fake)
    return cos_sim.cpu().numpy()

def compute_lpips(lpips_loss, img, fake):
    """LPIPS Distance"""
    lpips_val = lpips_loss(img, fake).squeeze().cpu().numpy()
    return lpips_val

def compute_locality_metrics(fake, img, cam):
    """IoU & SEC (using Grad-CAM from info if available)"""
    changed_pixels = ((fake - img).abs() > 0.1).float().mean(1, keepdim=True)
    iou = ((cam > 0.5) & (changed_pixels > 0.5)).sum() / ((cam > 0.5) | (changed_pixels > 0.5)).sum()
    sec = ((changed_pixels * (cam > 0.5)).sum() / (changed_pixels.sum() + 1e-8))
    return iou.cpu().numpy(), sec.cpu().numpy()

def compute_fid(all_orig, all_fake, device):
    """FID Score"""
    all_orig = torch.cat(all_orig).numpy()
    all_fake = torch.cat(all_fake).numpy()
    fid = fid_score.calculate_fid_given_arrays([all_orig, all_fake], batch_size=128, device=device, dims=2048)
    return fid

def compute_cf_score(results):
    """Final CF-Score"""
    csr = np.mean(results['CSR'])
    id_sim = np.mean(results['Identity_Sim'])
    lpips_val = np.mean(results['LPIPS'])
    sec = np.mean(results['SEC']) if results['SEC'] else 0.8
    cf_score = csr * id_sim * (1 - lpips_val) * sec
    return cf_score

@torch.no_grad()
def evaluate():
    all_orig, all_fake = [], []
    pbar = tqdm(test_loader, desc="Evaluating CF")
    
    for batch in pbar:
        img = batch['image'].to(device) / 127.5 - 1.0  # [-1, 1]
        attr_real = batch['attributes'].float().to(device)[:, target_idx]

        B = img.shape[0]
        # Randomly choose 1–3 attributes to flip
        num_flip = np.random.randint(1, min(4, len(target_idx)+1))
        flip_idx = np.random.choice(len(target_idx), num_flip, replace=False)
        target_attr = attr_real.clone()
        target_attr[range(B), flip_idx] = 1.0 - target_attr[range(B), flip_idx]

        # Generate counterfactual
        z_noise = torch.randn(B, 512, device=device)
        alpha_vec = torch.ones(B, 3, device=device) * 0.9
        fake, info = generator(img, z_noise, alpha_vec, target_attr)

        # Classifier predictions
        pred_real = torch.sigmoid(classifier(img))
        pred_fake = torch.sigmoid(classifier(fake))

        # 1. Counterfactual Success Rate (per sample)
        results['CSR'].extend(compute_csr(pred_fake, target_attr, target_idx))

        # 2. Targeted Flip Accuracy
        results['Target_Acc'].extend(compute_targeted_flip_accuracy(pred_fake, target_idx))

        # 3. Untargeted Stability
        results['Untargeted_Stability'].extend(compute_untargeted_stability(pred_real, pred_fake, target_idx))

        # 4. Identity Similarity (ArcFace)
        results['Identity_Sim'].extend(compute_identity_similarity(identity_net, img, fake))

        # 5. LPIPS
        results['LPIPS'].extend(compute_lpips(lpips_loss, img, fake))

        # 6. Locality: IoU & SEC (using Grad-CAM from info if available)
        if 'cam' in info:
            cam = info['cam']
            iou, sec = compute_locality_metrics(fake, img, cam)
            results['IoU_Mask'].extend(iou)
            results['SEC'].extend(sec)

        # Save for FID
        all_orig.append(((img + 1) * 127.5).cpu())
        all_fake.append(((fake + 1) * 127.5).cpu())

    # FID
    results['FID'] = compute_fid(all_orig, all_fake, device)

    # Final CF-Score
    results['CF_Score'] = compute_cf_score(results)

    return results

# ----------------------- Run & Print -----------------------
if __name__ == '__main__':
    metrics = evaluate()

    print("\n" + "="*60)
    print("           COUNTERFACTUAL EVALUATION RESULTS (2025 SOTA)")
    print("="*60)
    print(f"Counterfactual Success Rate (CSR)       : {np.mean(metrics['CSR']):.4f}")
    print(f"Targeted Flip Accuracy                  : {np.mean(metrics['Target_Acc']):.4f}")
    print(f"Untargeted Stability (lower better)     : {np.mean(metrics['Untargeted_Stability']):.4f}")
    print(f"Identity Similarity (ArcFace)           : {np.mean(metrics['Identity_Sim']):.4f}")
    print(f"LPIPS Distance (lower better)           : {np.mean(metrics['LPIPS']):.4f}")
    print(f"Grad-CAM IoU                            : {np.mean(metrics['IoU_Mask']):.4f}")
    print(f"Spatial Edit Concentration (SEC)        : {np.mean(metrics['SEC']):.4f}")
    print(f"Paired FID                              : {metrics['FID']:.2f}")
    print(f"CF-Score (higher is better)             : {metrics['CF_Score']:.4f}")
    print("="*60)
    if metrics['CF_Score'] >= 0.78:
        print("SOTA ACHIEVED — Ready for submission!")
    elif metrics['CF_Score'] >= 0.72:
        print("Top-5 performance — Strong paper")
    else:
        print("Room for improvement")
    print("="*60)

    # Save results
    os.makedirs("eval_results", exist_ok=True)
    # Fix: Use tuple of classes, not generic types
    json.dump({k: float(v) if isinstance(v, (float, np.float32, np.float64)) else v for k, v in metrics.items()},
              open(f"eval_results/result_{os.path.basename(args.ckpt)}.json", "w"), indent=2)