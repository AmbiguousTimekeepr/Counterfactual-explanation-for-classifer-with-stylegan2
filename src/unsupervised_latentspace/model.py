import torch
import torch.nn as nn
from collections import OrderedDict
from .decoder import HVQDecoder, ResBlock
from .quantizer import VectorQuantizerEMA

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_res_blocks, stride):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, hidden_dim, 4, stride, 1),
            nn.ReLU()
        ]
        for _ in range(num_res_blocks):
            layers.append(ResBlock(hidden_dim, hidden_dim))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class HVQVAE_3Level(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hd = cfg.hidden_dim
        ed = cfg.embed_dim
        self.in_channels = cfg.in_channels
        
        # --- ENCODERS ---
        self.enc_b = nn.Sequential(
            nn.Conv2d(cfg.in_channels, hd // 2, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(hd // 2, hd, 4, 2, 1), nn.ReLU(),
            *[ResBlock(hd, hd) for _ in range(cfg.num_res_blocks)]
        )
        self.enc_m = Encoder(hd, hd, cfg.num_res_blocks, stride=2)
        self.enc_t = Encoder(hd, hd, cfg.num_res_blocks, stride=2)
        
        # --- QUANTIZERS ---
        self.quant_conv_t = nn.Conv2d(hd, ed, 1)
        self.quant_conv_m = nn.Conv2d(hd, ed, 1)
        self.quant_conv_b = nn.Conv2d(hd, ed, 1)
        
        self.quant_t = VectorQuantizerEMA(
            num_embeddings=cfg.num_embeddings['top'],
            embedding_dim=ed,
            commitment_cost=cfg.commitment_cost['top'],
            decay=cfg.ema_decay['top'],
            epsilon=cfg.epsilon,
        )
        self.quant_m = VectorQuantizerEMA(
            num_embeddings=cfg.num_embeddings['mid'],
            embedding_dim=ed,
            commitment_cost=cfg.commitment_cost['mid'],
            decay=cfg.ema_decay['mid'],
            epsilon=cfg.epsilon,
        )
        self.quant_b = VectorQuantizerEMA(
            num_embeddings=cfg.num_embeddings['bottom'],
            embedding_dim=ed,
            commitment_cost=cfg.commitment_cost['bottom'],
            decay=cfg.ema_decay['bottom'],
            epsilon=cfg.epsilon,
        )
        
        # --- DECODER ---
        self.decoder = HVQDecoder(cfg)
        self.dec_t = self.decoder.dec_t
        self.dec_m = self.decoder.dec_m
        self.dec_b = self.decoder.dec_b
        
        # ✅ FIX: Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to prevent NaN/Inf on first step"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, dropout_rate=0.0):
        # ✅ FIX: Clamp input to [-1, 1]
        x = torch.clamp(x, -1.0, 1.0)
        
        # 1. Encode
        feat_b = self.enc_b(x)       # 32x32
        feat_m = self.enc_m(feat_b)  # 16x16
        feat_t = self.enc_t(feat_m)  # 8x8
        
        # 2. Quantize
        q_t, loss_t, ids_t = self.quant_t(self.quant_conv_t(feat_t))
        q_m, loss_m, ids_m = self.quant_m(self.quant_conv_m(feat_m))
        q_b, loss_b, ids_b = self.quant_b(self.quant_conv_b(feat_b))
        
        # ✅ NEW: CASCADED DROPOUT LOGIC (Enforce Hierarchy)
        # Implements 25-25-50 split to force semantic hierarchy learning
        if self.training and dropout_rate > 0.0:
            r = torch.rand(1).item()
            
            # Scenario A: TRAIN TOP ONLY (25%) -> Forces Top to learn Structure
            if r < 0.25:
                q_m = torch.zeros_like(q_m)
                q_b = torch.zeros_like(q_b)
                
            # Scenario B: TRAIN TOP + MID (25%) -> Forces Mid to learn Attributes
            elif r < 0.50:
                q_b = torch.zeros_like(q_b)
                
            # Scenario C: TRAIN FULL (50%) -> Learns Texture and Fine Details
            else:
                pass
        
        # ✅ FIX: Clamp quantized values to prevent NaN
        q_t = torch.clamp(q_t, -10.0, 10.0)
        q_m = torch.clamp(q_m, -10.0, 10.0)
        q_b = torch.clamp(q_b, -10.0, 10.0)
        
        # 3. Decode (Pass NATIVE resolutions)
        recon = self.decode_codes(q_t, q_m, q_b)
        
        # ✅ FIX: Ensure output is in [-1, 1]
        recon = torch.clamp(recon, -1.0, 1.0)
        
        return recon, loss_t + loss_m + loss_b, (ids_t, ids_m, ids_b)

    def decode_codes(self, q_t, q_m, q_b):
        """
        Decode from quantized codes at their NATIVE resolutions:
        q_t: 8x8
        q_m: 16x16
        q_b: 32x32
        """
        # 1. Decode Top (8x8 -> 16x16)
        # dec_t uses ConvTranspose2d(stride=2), so 8x8 becomes 16x16
        return self.decoder(q_t, q_m, q_b)

    def get_codes(self, x):
        """Helper to get quantized vectors for partial recon check"""
        feat_b = self.enc_b(x)
        feat_m = self.enc_m(feat_b)
        feat_t = self.enc_t(feat_m)
        
        q_t, _, _ = self.quant_t(self.quant_conv_t(feat_t))
        q_m, _, _ = self.quant_m(self.quant_conv_m(feat_m))
        q_b, _, _ = self.quant_b(self.quant_conv_b(feat_b))
        return q_t, q_m, q_b

    def load_state_dict(self, state_dict, strict: bool = True):
        remapped = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("dec_") and not key.startswith("decoder."):
                new_key = f"decoder.{key}"
                remapped[new_key] = value
            else:
                remapped[key] = value
        return super().load_state_dict(remapped, strict=strict)