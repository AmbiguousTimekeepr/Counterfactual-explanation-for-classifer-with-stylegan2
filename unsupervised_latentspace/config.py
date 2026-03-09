class Config:
    def __init__(self):
        # --- Hardware & Paths ---
        self.device = "cuda"
        self.data_path = "./data/celeba" 
        self.save_dir = "./checkpoints_production"
        
        # --- Data Params ---
        self.image_size = 128
        # RTX 2060S Optimization: Lower batch size + Gradient Accumulation
        self.batch_size = 8       # Physical batch size
        self.accumulation_steps = 4 # Effective batch size = 8 * 4 = 32
        self.num_workers = 4
        self.pin_memory = True
        
        # --- Architecture (VQ-VAE-2) ---
        self.levels = ['top', 'mid', 'bottom']
        self.in_channels = 3
        self.hidden_dim = 128     
        self.num_res_blocks = 2   
        
        # --- Quantizer (EMA) ---
        self.embed_dim = 64
        self.num_embeddings = 512 
        self.commitment_cost = 0.25
        self.decay = 0.99         
        self.epsilon = 1e-5
        
        # --- Training (Step-based) ---
        self.learning_rate = 3e-4 # Base LR
        self.min_learning_rate = 1e-5
        self.total_steps = 100000  # Total training steps
        self.log_interval = 50     # Log every N steps
        self.save_interval = 5000  # Save checkpoint every N steps
        self.use_amp = True       
        
        # --- Sampling ---
        self.sample_interval = 2500  # Sample every N steps (half of save_interval)
        self.num_samples = 8         # Number of images to sample
        
        # --- Adversarial Training (VQ-GAN) ---
        self.disc_start_step = 1000  # Start discriminator after N steps warmup
        self.disc_weight = 0.5       # Weight for GAN loss
        
        # Loss Weights
        self.weights = {
            'recon': 1.0,
            'vq': 1.0,
            'perceptual': 1.0,     # Increased for photorealism
            'disc': 0.8            # Generator adversarial loss weight
        }

cfg = Config()