import torch
import torch.nn as nn
import torch.nn.functional as F

from .latent_mutator import LatentMutator
from .synthesis import StyleGAN3CounterfactualSynthesis
from .loss_functions import LossFunctions

from ..latent_space.model import HierarchicalVQVAE
from ..classifiers.ex_classifier import ExplainableClassifier, load_trained_model
from ..classifiers.cam_utils import get_gradcam_mask
from ..classifiers.ig_utils import get_ig_safe

class CounterfactualGenerator(nn.Module):
    def __init__(self, resolution=128, vqvae_ckpt=None, classifier_ckpt=None, classifier_model_name='mobilenet_v2', device='cuda'):
        super().__init__()
        # Load pretrained VQ-VAE
        self.vqvae = HierarchicalVQVAE().to(device)
        if vqvae_ckpt is not None:
            state = torch.load(vqvae_ckpt, map_location=device)
            # If checkpoint is a dict, extract model weights
            if isinstance(state, dict):
                if 'model_state_dict' in state:
                    state = state['model_state_dict']
                elif 'state_dict' in state:
                    state = state['state_dict']
            self.vqvae.load_state_dict(state, strict=False)
        self.vqvae.eval()

        self.mutator = LatentMutator()
        self.synthesis = StyleGAN3CounterfactualSynthesis(
            resolution=resolution
        )
        # Load pretrained classifier
        if classifier_ckpt is not None:
            self.classifier = load_trained_model(
                classifier_ckpt, classifier_model_name, device=device
            ).eval()
        else:
            self.classifier = ExplainableClassifier().eval()

    def forward(self, x_real, z_noise, alpha_vec, target_attrs):
        with torch.no_grad():
            z_list = self.vqvae.encode_to_list(x_real)

        # Get XAI guidance
        ig = get_ig_safe(self.classifier, x_real, target_attrs)
        current_probs = torch.sigmoid(self.classifier(x_real))
        cam = get_gradcam_mask(self.classifier, x_real, target_attrs, current_probs)

        # Mutation in latent space
        z_edited = self.mutator(
            z_list, alpha_vec, target_attrs,
            current_probs, ig, cam
        )

        fake = self.synthesis(z_noise, z_edited)

        return fake, {
            'z_orig': z_list,
            'z_edited': z_edited,
            'cam': cam
        }