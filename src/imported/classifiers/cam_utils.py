import torch
import torch.nn.functional as F

def get_gradcam_mask(classifier, images, target_attrs, current_probs):
    classifier.eval()
    images = images.detach().clone().requires_grad_(True)

    logits = classifier(images)
    # Ensure target_attrs and current_probs have matching shape
    if target_attrs.shape[1] != logits.shape[1]:
        # If target_attrs has an extra column (e.g., 40 instead of 39), trim or select only the needed attributes
        target_attrs = target_attrs[:, :logits.shape[1]]
        current_probs = current_probs[:, :logits.shape[1]]

    flip = (target_attrs.round() != current_probs.round()).float()
    if not flip.any():
        return torch.zeros(images.shape[0], 1, images.shape[2], images.shape[3], device=images.device)

    score = (logits * flip).sum(dim=1)
    classifier.zero_grad()
    score.sum().backward()

    grads = classifier.gradients.get(classifier.target_layer_name)
    acts = classifier.activations.get(classifier.target_layer_name)
    if grads is None or acts is None:
        return torch.zeros(images.shape[0], 1, images.shape[2], images.shape[3], device=images.device)

    grad2 = grads ** 2
    grad3 = grad2 * grads
    eps = 1e-8
    denom = 2 * grad2 + (acts * grad3).sum(dim=(2, 3), keepdim=True)
    alphas = grad2 / (denom + eps)
    weights = (alphas * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, (images.shape[2], images.shape[3]), mode='bilinear', align_corners=False)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + eps)

    return cam.detach()