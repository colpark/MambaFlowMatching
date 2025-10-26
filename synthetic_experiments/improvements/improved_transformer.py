"""
Improved Transformer Flow with Modular Technique Integration

Based on train_transformer_v2.py but with pluggable improvements
"""
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
synthetic_dir = os.path.dirname(script_dir)
repo_root = os.path.dirname(synthetic_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict

from synthetic_experiments.improvements.techniques import (
    MultiScaleFourierFeatures,
    AdaptiveLayerNorm,
    DropPath,
    get_drop_path_schedule,
)


class ImprovedTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with optional improvements:
    - Adaptive LayerNorm (technique 5)
    - Stochastic Depth (technique 10)
    """
    def __init__(
        self,
        d_model,
        num_heads=8,
        dim_feedforward=512,
        dropout=0.1,
        use_adaln=False,
        time_embed_dim=128,
        drop_path_prob=0.0,
    ):
        super().__init__()

        self.use_adaln = use_adaln

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # Normalization
        if use_adaln:
            self.norm1 = AdaptiveLayerNorm(d_model, time_embed_dim)
            self.norm2 = AdaptiveLayerNorm(d_model, time_embed_dim)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        # Stochastic depth
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()

    def forward(self, x, time_embed=None):
        """
        Args:
            x: (B, N, d_model)
            time_embed: (B, time_embed_dim) - required if use_adaln=True
        """
        # Self-attention with residual
        if self.use_adaln:
            assert time_embed is not None
            x_norm = self.norm1(x, time_embed)
        else:
            x_norm = self.norm1(x)

        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path(attn_out)

        # Feed-forward with residual
        if self.use_adaln:
            x_norm = self.norm2(x, time_embed)
        else:
            x_norm = self.norm2(x)

        ffn_out = self.ffn(x_norm)
        x = x + self.drop_path(ffn_out)

        return x


class ImprovedTransformerFlow(nn.Module):
    """
    Improved Transformer Flow Matching with Pluggable Techniques

    Supports:
    - Technique 2: Multi-scale positional encoding
    - Technique 5: Adaptive LayerNorm
    - Technique 6: Self-conditioning (handled in training)
    - Technique 10: Stochastic depth
    """
    def __init__(
        self,
        d_model=128,
        num_layers=4,
        num_heads=8,
        dim_feedforward=512,
        num_frequencies=16,
        dropout=0.1,
        # Technique flags
        use_multiscale_pe=False,  # Technique 2
        num_scales=6,
        use_adaln=False,  # Technique 5
        use_stochastic_depth=False,  # Technique 10
        max_drop_prob=0.1,
        use_self_conditioning=False,  # Technique 6 (changes input dim)
    ):
        super().__init__()

        self.d_model = d_model
        self.use_multiscale_pe = use_multiscale_pe
        self.use_adaln = use_adaln
        self.use_self_conditioning = use_self_conditioning

        # Positional encoding (Technique 2: Multi-scale or standard)
        if use_multiscale_pe:
            self.coord_encoding = MultiScaleFourierFeatures(
                input_dim=2,
                num_frequencies=num_frequencies,
                num_scales=num_scales
            )
            coord_feat_dim = self.coord_encoding.output_dim
        else:
            # Standard Fourier features
            self.num_frequencies = num_frequencies
            self.register_buffer('B', torch.randn(2, num_frequencies) * 10.0)
            coord_feat_dim = 2 + 2 * num_frequencies

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )

        # Input projections
        # Sparse observation projection
        sparse_input_dim = coord_feat_dim + 1  # coords + value
        self.sparse_proj = nn.Linear(sparse_input_dim, d_model)

        # Query projection
        # If self-conditioning: concat [coord_feats, query_values, prev_prediction]
        query_value_dim = 2 if use_self_conditioning else 1  # current + prev
        query_input_dim = coord_feat_dim + query_value_dim
        self.query_proj = nn.Linear(query_input_dim, d_model)

        # Transformer layers (with optional improvements)
        if use_stochastic_depth:
            drop_path_probs = get_drop_path_schedule(num_layers, max_drop_prob)
        else:
            drop_path_probs = [0.0] * num_layers

        self.transformer_layers = nn.ModuleList([
            ImprovedTransformerEncoderLayer(
                d_model,
                num_heads,
                dim_feedforward,
                dropout,
                use_adaln=use_adaln,
                time_embed_dim=128,
                drop_path_prob=drop_path_probs[i],
            )
            for i in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, 1)

    def encode_coordinates(self, coords):
        """
        Encode coordinates with Fourier features

        Args:
            coords: (B, N, 2)
        Returns:
            features: (B, N, feat_dim)
        """
        if self.use_multiscale_pe:
            return self.coord_encoding(coords)
        else:
            # Standard Fourier features
            proj = 2 * np.pi * coords @ self.B  # (B, N, num_frequencies)
            return torch.cat([coords, torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(
        self,
        sparse_coords,
        sparse_values,
        query_coords,
        t,
        query_values=None,
        prev_prediction=None,  # For self-conditioning
    ):
        """
        Forward pass

        Args:
            sparse_coords: (B, N_sparse, 2)
            sparse_values: (B, N_sparse, 1)
            query_coords: (B, N_query, 2)
            t: (B,) or (B, 1) timestep
            query_values: (B, N_query, 1) optional, for training
            prev_prediction: (B, N_query, 1) optional, for self-conditioning
        """
        B = sparse_coords.shape[0]

        # Time embedding
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)
        time_embed = self.time_mlp(t)  # (B, 128)

        # Encode coordinates
        sparse_feats = self.encode_coordinates(sparse_coords)
        query_feats = self.encode_coordinates(query_coords)

        # Sparse observation tokens
        sparse_input = torch.cat([sparse_feats, sparse_values], dim=-1)
        sparse_tokens = self.sparse_proj(sparse_input)  # (B, N_sparse, d_model)

        # Query tokens
        if query_values is None:
            # Sampling mode: start from noise
            query_values = torch.randn(B, query_coords.shape[1], 1, device=query_coords.device)

        # Self-conditioning: concatenate previous prediction
        if self.use_self_conditioning and prev_prediction is not None:
            query_input = torch.cat([query_feats, query_values, prev_prediction], dim=-1)
        else:
            query_input = torch.cat([query_feats, query_values], dim=-1)

        query_tokens = self.query_proj(query_input)  # (B, N_query, d_model)

        # Combine sparse and query tokens
        tokens = torch.cat([sparse_tokens, query_tokens], dim=1)  # (B, N_sparse + N_query, d_model)

        # Transformer layers
        for layer in self.transformer_layers:
            if self.use_adaln:
                tokens = layer(tokens, time_embed)
            else:
                tokens = layer(tokens)

        # Extract query tokens
        N_sparse = sparse_coords.shape[1]
        query_tokens_out = tokens[:, N_sparse:, :]  # (B, N_query, d_model)

        # Predict velocity
        pred_v = self.output_proj(query_tokens_out)  # (B, N_query, 1)

        return pred_v


def build_model_from_techniques(
    technique_ids: List[int],
    d_model=128,
    num_layers=4,
    num_heads=8,
    dim_feedforward=512,
    num_frequencies=16,
    dropout=0.1,
) -> ImprovedTransformerFlow:
    """
    Build model with specified techniques enabled

    Args:
        technique_ids: List of technique IDs (1-10)
        other params: Architecture hyperparameters

    Returns:
        model: ImprovedTransformerFlow with techniques enabled
    """
    # Parse technique flags
    use_multiscale_pe = 2 in technique_ids
    use_adaln = 5 in technique_ids
    use_self_conditioning = 6 in technique_ids
    use_stochastic_depth = 10 in technique_ids

    # Build model
    model = ImprovedTransformerFlow(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        num_frequencies=num_frequencies,
        dropout=dropout,
        use_multiscale_pe=use_multiscale_pe,
        num_scales=6,
        use_adaln=use_adaln,
        use_stochastic_depth=use_stochastic_depth,
        max_drop_prob=0.1,
        use_self_conditioning=use_self_conditioning,
    )

    return model


# ============================================================================
# Sampling (same as V2)
# ============================================================================

@torch.no_grad()
def sample_heun(model, sparse_coords, sparse_values, query_coords, num_steps=50, device='cpu'):
    """
    Sample using Heun's method (2nd order ODE solver)
    """
    model.eval()

    B = query_coords.shape[0]
    N = query_coords.shape[1]

    # Start from noise
    z_t = torch.randn(B, N, 1, device=device)

    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((B,), i * dt, device=device)

        # First prediction
        v1 = model(sparse_coords, sparse_values, query_coords, t, query_values=z_t)

        # Euler step
        z_next = z_t + v1 * dt

        # Second prediction
        t_next = torch.full((B,), (i + 1) * dt, device=device)
        v2 = model(sparse_coords, sparse_values, query_coords, t_next, query_values=z_next)

        # Heun's method: average of two slopes
        z_t = z_t + 0.5 * (v1 + v2) * dt

    return z_t
