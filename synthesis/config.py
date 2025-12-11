import argparse
from pathlib import Path
from ..unsupervised_latentspace.config import Config as VQVAEConfig


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
        self.accumulation_steps = 8        # Effective batch = 32
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
        self.num_attributes = 40
        self.num_classes = 40
        
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
            'ortho': 0.3,
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
        self.decoder_pretrain_epochs = 0
        self.decoder_lr = 1e-3
        self.decoder_betas = (0.0, 0.99)
        self.decoder_checkpoint_dir = "outputs/stylegan_decoder"
        self.decoder_checkpoint_path = ""
        
    @classmethod
    def from_args(cls):
        """Load config from command-line arguments"""
        parser = argparse.ArgumentParser(description='Counterfactual Generation Config')
        parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
        parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
        parser.add_argument('--vqvae_path', type=str, default='outputs/checkpoints_production/best_model.pth')
        parser.add_argument('--classifier_path', type=str, default='outputs/trained_classifiers(0512)/best_model.pth')
        parser.add_argument('--data_root', type=str, default='Dataset/celeba_70percent_721')
        
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
