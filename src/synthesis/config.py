import argparse
from pathlib import Path
from ..latentspace.config import Config as VQVAEConfig
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
        # Decoder Pretraining (runs first)
        # ============================================================================
        self.decoder_pretrain_epochs = 10
        self.decoder_lr = 1e-3
        self.decoder_betas = (0.0, 0.99)
        self.decoder_checkpoint_dir = "outputs/synth_network/stylegan_decoder"
        self.decoder_checkpoint_path = ""

        # ============================================================================
        # Decoder Sharpening (GAN finetune on recon fidelity)
        # ============================================================================
        self.sharpening_epochs = 15
        self.g_sharpening_lr = 2e-4
        self.d_sharpening_lr = 1e-5
        self.sharpening_betas = (0.0, 0.99)
        self.sharpening_checkpoint_dir = "outputs/synth_network/stylegan_decoder_sharpened"
        self.sharpened_decoder_path = ""
        self.sharpening_adv_weight = 1.0
        self.sharpening_l1_weight = 1.0
        self.sharpening_lpips_weight = 1.0
        self.sharpening_fm_weight = 1.0
        self.sharpening_r1_gamma = 10.0
        self.sharpening_batch_size = 8
        self.sharpening_max_batches = 0   # 0 = full epoch; set small for faster runs
        self.sharpening_sample_size = 4   # samples saved each epoch

        # ============================================================================
        # Mutation / Counterfactual Training (comprehensive phase)
        # ============================================================================
        self.learning_rate = 1e-4          # Override: lower LR for fine-tuning
        self.weight_decay = 1e-5
        self.batch_size = 4                # Smaller batch for editing
        self.num_epochs = 50
        self.accumulation_steps = 8        # Effective batch = 32
        self.num_workers = 4
        self.adv_weight = 0.5
        self.adv_r1_gamma = 0.0            # R1 regularization for main D (0 to disable)
        self.grad_clip = 1.0               # Generator/mutator clipping
        self.d_grad_clip = 1.0             # Discriminator clipping
        self.use_amp = False               # Force full float32 for stability

        # ============================================================================
        # Model Paths (Pre-trained Checkpoints)
        # ============================================================================
        self.vqvae_checkpoint = "outputs/latent_space/hrvqvae/checkpoints/best_gan.pth"
        self.classifier_checkpoint = "outputs/classifier/checkpoints/resnet18_cbam_128_05_3/best_model.pth"
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
        
        # ============================================================================
        # Synthesis-Specific Data Config
        # ============================================================================
        self.image_size = 128              # Already in VQVAEConfig, but confirm here
        self.use_eval_partition = False
        
        # ============================================================================
        # Loss Weights (Counterfactual-Specific)
        # ============================================================================
        self.synthesis_loss_weights = {
            'cf': 10.0,
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
