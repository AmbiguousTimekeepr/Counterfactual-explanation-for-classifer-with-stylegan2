import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizerEMA(nn.Module):
    """
    Exponential Moving Average (EMA) Vector Quantizer.
    This version creates stable codebooks essential for your 'Discrete Latent Mutator'.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Initialize embeddings and EMA buffers
        self.register_buffer('embedding', torch.randn(num_embeddings, embedding_dim))
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', torch.randn(num_embeddings, embedding_dim))

    def forward(self, inputs):
        # inputs: [B, C, H, W] -> [B, H, W, C]
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # ✅ FIX 1: Force float32 for distance calculation (prevent overflow in AMP)
        flat_input_fp32 = flat_input.float()
        embedding_fp32 = self.embedding.float()
        
        # Calculate distances: (x-e)^2 = x^2 + e^2 - 2xe
        distances = (torch.sum(flat_input_fp32**2, dim=1, keepdim=True) 
                    + torch.sum(embedding_fp32**2, dim=1)
                    - 2 * torch.matmul(flat_input_fp32, embedding_fp32.t()))
            
        # Encoding Indices
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize (use original precision)
        quantized = torch.matmul(encodings, self.embedding).view(input_shape)
        
        # Training Updates (EMA)
        if self.training:
            # Update cluster size
            encodings_sum = encodings.sum(0)
            self.ema_cluster_size.data.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
            
            # Laplace smoothing to prevent division by zero
            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n
            )
            
            # Update embeddings
            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            self.embedding.data.copy_(self.ema_w / cluster_size.unsqueeze(1))
            
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator (Gradient Copy)
        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        # ✅ Return one-hot encodings [B*H*W, num_embeddings] for perplexity calculation
        # This avoids conversion overhead in trainer.py
        return quantized, loss, encodings