"""
Perceiver IO architecture for neural field applications

Flexible architecture that handles sparse inputs/outputs naturally through cross-attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FourierFeatures(nn.Module):
    """
    Fourier feature positional encoding

    Reference: "Fourier Features Let Networks Learn High Frequency Functions
                in Low Dimensional Domains" (Tancik et al., 2020)
    """
    def __init__(self, coord_dim=2, num_freqs=256, scale=10.0, learnable=False):
        """
        Args:
            coord_dim: Dimension of coordinates (2 for images: x,y)
            num_freqs: Number of frequency bands
            scale: Std of Gaussian sampling for frequency matrix
            learnable: If True, make frequency matrix learnable
        """
        super().__init__()
        self.coord_dim = coord_dim
        self.num_freqs = num_freqs
        self.output_dim = 2 * num_freqs  # sin + cos for each frequency

        # Sample frequency matrix B from Gaussian
        B = torch.randn(num_freqs, coord_dim) * scale

        if learnable:
            self.register_parameter('B', nn.Parameter(B))
        else:
            self.register_buffer('B', B)

    def forward(self, coords):
        """
        Args:
            coords: (B, N, coord_dim) coordinates in [0, 1]

        Returns:
            features: (B, N, 2*num_freqs) Fourier features
        """
        # coords: (B, N, coord_dim)
        # B: (num_freqs, coord_dim)

        # Compute 2π * B @ coords^T
        coords_proj = 2 * math.pi * torch.matmul(coords, self.B.T)  # (B, N, num_freqs)

        # Concatenate sin and cos
        features = torch.cat([
            torch.sin(coords_proj),
            torch.cos(coords_proj)
        ], dim=-1)  # (B, N, 2*num_freqs)

        return features


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention layer"""

    def __init__(self, query_dim, key_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.head_dim = query_dim // num_heads

        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(key_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (B, N_q, query_dim)
            key: (B, N_k, key_dim)
            value: (B, N_k, key_dim)
            mask: Optional attention mask

        Returns:
            output: (B, N_q, query_dim)
        """
        B, N_q, _ = query.shape
        N_k = key.shape[1]

        # Project and reshape for multi-head attention
        Q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, N_q, N_k)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Combine heads
        output = torch.matmul(attn, V)  # (B, H, N_q, head_dim)
        output = output.transpose(1, 2).contiguous().view(B, N_q, self.query_dim)
        output = self.out_proj(output)

        return output


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention layer"""

    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttention(dim, dim, num_heads, dropout)

    def forward(self, x, mask=None):
        return self.cross_attn(x, x, x, mask)


class PerceiverBlock(nn.Module):
    """Single Perceiver block with cross-attention + self-attention"""

    def __init__(self, latent_dim, input_dim, num_heads=8, mlp_ratio=4, dropout=0.0):
        super().__init__()

        # Cross-attention: latents attend to inputs
        self.cross_attn = MultiHeadCrossAttention(
            query_dim=latent_dim,
            key_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.cross_norm1 = nn.LayerNorm(latent_dim)
        self.cross_norm2 = nn.LayerNorm(latent_dim)

        # Self-attention on latents
        self.self_attn = MultiHeadSelfAttention(latent_dim, num_heads, dropout)
        self.self_norm1 = nn.LayerNorm(latent_dim)
        self.self_norm2 = nn.LayerNorm(latent_dim)

        # MLP
        mlp_hidden_dim = int(latent_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, latent_dim),
            nn.Dropout(dropout)
        )

    def forward(self, latents, inputs):
        """
        Args:
            latents: (B, N_latent, latent_dim)
            inputs: (B, N_input, input_dim)

        Returns:
            latents: (B, N_latent, latent_dim) updated latents
        """
        # Cross-attention
        latents = latents + self.cross_attn(
            self.cross_norm1(latents),
            inputs,
            inputs
        )

        # Self-attention
        latents = latents + self.self_attn(self.self_norm1(latents))

        # MLP
        latents = latents + self.mlp(self.self_norm2(latents))

        return latents


class PerceiverIO(nn.Module):
    """
    Perceiver IO for sparse pixel prediction

    Architecture:
    1. Encode inputs: Sparse pixels + Fourier features → Input embeddings
    2. Cross-attention: Latents attend to input embeddings
    3. Self-attention: Process latents
    4. Decode: Query positions attend to latents → Output predictions
    """

    def __init__(
        self,
        input_channels=3,           # RGB
        output_channels=3,          # RGB
        coord_dim=2,                # (x, y)
        num_latents=512,            # Number of latent vectors
        latent_dim=512,             # Latent dimension
        num_fourier_feats=256,      # Fourier feature dimension
        fourier_scale=10.0,         # Fourier feature scale
        num_blocks=6,               # Number of Perceiver blocks
        num_heads=8,                # Attention heads
        mlp_ratio=4,                # MLP expansion ratio
        dropout=0.1,
        learnable_fourier=False
    ):
        super().__init__()

        self.num_latents = num_latents
        self.latent_dim = latent_dim

        # Fourier features for positional encoding
        self.fourier_features = FourierFeatures(
            coord_dim=coord_dim,
            num_freqs=num_fourier_feats,
            scale=fourier_scale,
            learnable=learnable_fourier
        )

        fourier_dim = 2 * num_fourier_feats

        # Input embedding: Fourier features + pixel values
        self.input_embed = nn.Sequential(
            nn.Linear(fourier_dim + input_channels, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )

        # Learnable latent array
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        nn.init.normal_(self.latents, std=0.02)

        # Perceiver blocks
        self.blocks = nn.ModuleList([
            PerceiverBlock(
                latent_dim=latent_dim,
                input_dim=latent_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])

        # Query embedding: Fourier features for output positions
        self.query_embed = nn.Sequential(
            nn.Linear(fourier_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )

        # Decoder cross-attention: queries attend to latents
        self.decoder = MultiHeadCrossAttention(
            query_dim=latent_dim,
            key_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Output projection
        self.output_proj = nn.Linear(latent_dim, output_channels)

    def forward(self, input_coords, input_values, query_coords):
        """
        Args:
            input_coords: (B, N_in, 2) input pixel coordinates in [0, 1]
            input_values: (B, N_in, 3) input pixel RGB values
            query_coords: (B, N_out, 2) query pixel coordinates in [0, 1]

        Returns:
            output_values: (B, N_out, 3) predicted RGB values at query positions
        """
        B = input_coords.shape[0]

        # 1. Encode input pixels
        input_pos = self.fourier_features(input_coords)  # (B, N_in, fourier_dim)
        input_feats = torch.cat([input_pos, input_values], dim=-1)  # (B, N_in, fourier_dim + 3)
        input_embed = self.input_embed(input_feats)  # (B, N_in, latent_dim)

        # 2. Initialize latents (broadcast to batch)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B, num_latents, latent_dim)

        # 3. Process with Perceiver blocks
        for block in self.blocks:
            latents = block(latents, input_embed)

        # 4. Encode query positions
        query_pos = self.fourier_features(query_coords)  # (B, N_out, fourier_dim)
        query_embed = self.query_embed(query_pos)  # (B, N_out, latent_dim)

        # 5. Decode: queries attend to latents
        output_feats = self.decoder(query_embed, latents, latents)  # (B, N_out, latent_dim)

        # 6. Project to output values
        output_values = self.output_proj(output_feats)  # (B, N_out, 3)

        return output_values


# Test code
if __name__ == "__main__":
    # Test Fourier features
    print("Testing Fourier Features...")
    ff = FourierFeatures(coord_dim=2, num_freqs=64, scale=10.0)
    coords = torch.rand(4, 100, 2)  # Batch of 4, 100 coordinates
    feats = ff(coords)
    print(f"  Input: {coords.shape} → Output: {feats.shape}")

    # Test Perceiver IO
    print("\nTesting Perceiver IO...")
    model = PerceiverIO(
        input_channels=3,
        output_channels=3,
        num_latents=256,
        latent_dim=256,
        num_fourier_feats=128,
        num_blocks=4,
        num_heads=8
    )

    # Sparse input: 20% of 32x32 = 204 pixels
    input_coords = torch.rand(4, 204, 2)
    input_values = torch.rand(4, 204, 3)

    # Query: another 20% = 204 pixels
    query_coords = torch.rand(4, 204, 2)

    output = model(input_coords, input_values, query_coords)
    print(f"  Input: {input_coords.shape} (coords) + {input_values.shape} (values)")
    print(f"  Query: {query_coords.shape}")
    print(f"  Output: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
