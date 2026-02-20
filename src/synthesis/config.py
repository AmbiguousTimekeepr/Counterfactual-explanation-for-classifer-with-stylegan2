import argparse
from pathlib import Path
from ..unsupervised_latentspace.config import Config as VQVAEConfig
from .dataset import SELECTED_ATTRIBUTES


class Config(VQVAEConfig):
    """
    Counterfactual Generation Training Config
    Inherits VQ-VAE architecture params from unsupervised_latentspace.Config
    Adds synthesis-specific parameters
    """
    
    def __init__(self):
        # Inherit all VQ-VAE config (hidden_dim, embed_dim, num_embeddings, etc.)
        super().__init__()
        
        # ============================================================================
        # Synthesis-Specific Training Hyperparameters
        # ============================================================================
        self.learning_rate = 1e-4          # Override: lower LR for fine-tuning
        self.weight_decay = 1e-5
        self.batch_size = 4                # Smaller batch for editing
        self.num_epochs = 50
        self.num_workers = 4
        
        # ============================================================================
        # Model Paths (Pre-trained Checkpoints)
        # ============================================================================
        self.vqvae_checkpoint = "outputs/checkpoints_production/checkpoints/checkpoint_step_22500.pth"
        self.classifier_checkpoint = "outputs/cnn_classfier/best_model.pth"
        self.data_root = "Dataset/celeba_70percent_721"
        
        # ============================================================================
        # Classifier Configuration
        # ============================================================================
        self.num_attributes = len(SELECTED_ATTRIBUTES)
        self.num_classes = len(SELECTED_ATTRIBUTES)
        self.active_attributes = list(SELECTED_ATTRIBUTES)
        self.max_active_attributes_per_epoch = 6
        self.ig_steps = 16
        self.saliency_cache_size = 512
        self.cam_threshold = 0.35
        self.use_ig = True
        self.gradcamplusplus_use = False
        
        # ============================================================================
        # Synthesis-Specific Data Config
        # ============================================================================
        self.image_size = 128              # Already in VQVAEConfig, but confirm here
        self.use_eval_partition = False
        
        # ============================================================================
        # Loss Weights (Counterfactual-Specific)
        # ============================================================================
        self.synthesis_loss_weights = {
            'cf': 18.0,
            'retention': 15.0,
            'latent_prox': 5.0,
            'ortho': 0.5,
            'sparse': 1e-3
        }
        
        # ============================================================================
        # Learning Rate Schedule
        # ============================================================================
        self.lr_schedule = {
            10: 5e-5,
            20: 2.5e-5,
            30: 1e-5
        }
        
        # ============================================================================
        # Checkpointing & Logging
        # ============================================================================
        self.save_interval = 1
        self.vis_interval = 1
        
        # ============================================================================
        # Decoder Pretraining Settings
        # ============================================================================
        self.decoder_pretrain_epochs = 10
        self.decoder_lr = 1e-3
        self.decoder_betas = (0.0, 0.99)
        self.decoder_checkpoint_dir = "outputs/synth_network/stylegan_decoder"
        self.decoder_checkpoint_path = ""
        
        # ============================================================================
        # Decoder Sharpening Settings
        # ============================================================================   
        self.sharpening_epochs = 15
        self.g_sharpening_lr = 2e-4
        self.d_sharpening_lr = 1e-5
        self.sharpening_betas = (0.0, 0.99)
        self.sharpening_checkpoint_dir = "outputs/synth_network/stylegan_decoder_sharpened"
        self.sharpened_decoder_path = ""

        # =========================================================================
        # Counterfactual Post-processing & Regularization
        # =========================================================================
        # Blend in latent space using saliency masks before decoding (soft blending)
        self.use_decoder_blend = False
        self.blend_kernel_size = 3  # odd number; 1 = no smoothing

        # Reduce global leakage: keep decoder style vector (w) computed from original latents
        # instead of edited latents. This prevents local edits from changing global modulation.
        self.decoder_w_from_orig = True

        # Mutator step controls (reduce if you see leakage/blur)
        # step = sigmoid(step_logits) * mutator_step_scale + mutator_step_min
        # NOTE: Large positive bias_init pushes sigmoid() ~ 1.0 from epoch 1,
        # producing near-max steps immediately (often causes leak + blur).
        self.mutator_step_min = 0.05
        self.mutator_step_scale = 1.5
        self.mutator_mid_mult = 1.2
        self.mutator_step_bias_init = -2.0

        # Optional: directly scale edit intensity using mean saliency inside the mask.
        # Recommended safe range: 0.05–0.30. Start small to avoid artifacts, especially
        # when masks are broad (mask=1/global edit).
        self.ig_step_gain = 0.10

        # Attribution alignment (optional, disabled by default)
        self.align_weight = 0.0
        self.align_interval = 400  # steps between alignment computations
        
        
    @classmethod
    def from_args(cls):
        """Load config from command-line arguments"""
        parser = argparse.ArgumentParser(description='Counterfactual Generation Config')
        parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
        parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
        parser.add_argument('--vqvae_path', type=str, default='outputs/checkpoints_production/checkpoint_step_22500.pth')
        parser.add_argument('--classifier_path', type=str, default='outputs/cnn_classifier/best_model.pth')
        parser.add_argument('--data_root', type=str, default='Dataset/celeba_70percent_721/train')
        
        args = parser.parse_args()
        
        cfg = cls()
        cfg.learning_rate = args.lr
        cfg.batch_size = args.batch_size
        cfg.num_epochs = args.num_epochs
        cfg.vqvae_checkpoint = args.vqvae_path
        cfg.classifier_checkpoint = args.classifier_path
        cfg.data_root = args.data_root
        
        return cfg
    
    def __repr__(self):
        items = []
        for k, v in self.__dict__.items():
            items.append(f"{k}: {v}")
        return "Config(\n  " + "\n  ".join(items) + "\n)"
