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

        init = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(init, mean=0.0, std=0.1)

        self.register_buffer('embedding', init.clone())
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', init.clone())

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

    def revive_dead_codes(self, encoder_outputs_flat, threshold=5.0):
        """
        Revive dead codes by resetting them to current encoder outputs.
        Called periodically during training.
        """
        with torch.no_grad():
            if encoder_outputs_flat is None:
                return

            counts = self.ema_cluster_size
            dead_mask = counts < threshold

            if not dead_mask.any():
                return

            num_dead = int(dead_mask.sum().item())
            encoder_outputs_flat = encoder_outputs_flat.to(self.embedding.device)
            encoder_outputs_flat = encoder_outputs_flat.to(self.ema_w.dtype)

            if encoder_outputs_flat.size(0) >= num_dead and num_dead > 0:
                indices = torch.randint(0, encoder_outputs_flat.size(0), (num_dead,), device=self.embedding.device)
                new_embeddings = encoder_outputs_flat[indices]
            else:
                live_mask = ~dead_mask
                if live_mask.any():
                    live_emb = self.embedding[live_mask]
                    idx = torch.randint(0, live_emb.size(0), (num_dead,), device=self.embedding.device)
                    new_embeddings = live_emb[idx]
                    new_embeddings = new_embeddings + torch.randn_like(new_embeddings) * 0.05
                else:
                    new_embeddings = torch.randn(num_dead, self.embedding_dim, device=self.embedding.device, dtype=self.ema_w.dtype)

            self.ema_w[dead_mask] = new_embeddings
            self.ema_cluster_size[dead_mask] = threshold

            updated_cluster_size = self.ema_cluster_size + self.epsilon
            normalized = self.ema_w / updated_cluster_size.unsqueeze(1)
            self.embedding.copy_(normalized)