import torch
import torch.nn.functional as F

def integrated_gradients_manual(model, images, target_attrs, steps=50, spatial=False):
    # images: [B, C, H, W], target_attrs: [B, num_classes]
    device = images.device
    baseline = torch.zeros_like(images)
    alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1, 1, 1)
    images = images.detach()
    baseline = baseline.detach()
    diff = images - baseline

    attributions = []
    for i in range(images.shape[0]):
        img = images[i:i+1]
        tgt = target_attrs[i:i+1]
        interpolated = baseline[i:i+1] + alphas * diff[i:i+1]
        interpolated.requires_grad_(True)
        outputs = model(interpolated)
        # IG for each attribute
        grads = []
        for j in range(outputs.shape[1]):
            score = outputs[:, j].sum()
            grad = torch.autograd.grad(score, interpolated, retain_graph=True)[0]
            grads.append(grad)
        grads = torch.stack(grads, dim=1)  # [steps, num_classes, C, H, W]
        avg_grads = grads.mean(dim=0)      # [num_classes, C, H, W]
        attr = diff[i:i+1] * avg_grads     # [num_classes, C, H, W]
        if spatial:
            # Sum over channels, keep spatial
            attr_map = attr.abs().sum(dim=1)  # [num_classes, H, W]
            attr_map = attr_map.unsqueeze(0)  # [1, num_classes, H, W]
            attr_map = F.interpolate(attr_map, (img.shape[2], img.shape[3]), mode='bilinear')
            attr_map = attr_map.squeeze(0)    # [num_classes, H, W]
            attr_map = attr_map / (attr_map.max() + 1e-8)
            attributions.append(attr_map)
        else:
            # Mean over spatial dims and channels
            attr_score = attr.abs().mean(dim=[1,2,3])  # [num_classes]
            attributions.append(attr_score)
    if spatial:
        return torch.stack(attributions)  # [B, num_classes, H, W]
    else:
        return torch.stack(attributions)  # [B, num_classes]

def get_ig_safe(classifier, images, target_attrs, spatial=False):
    # classifier: ExplainableClassifier, images: [B, C, H, W], target_attrs: [B, num_classes]
    return integrated_gradients_manual(classifier, images, target_attrs, steps=50, spatial=spatial)