"""
MAMBA Diffusion V4 - Transformer Encoder Baseline

Replaces MAMBA state space model with standard Transformer encoder to:
- Compare linear-complexity MAMBA vs quadratic-complexity Transformer
- Evaluate if attention's global context helps for sparse neural fields
- Benchmark against state-of-the-art attention mechanisms

Architecture:
- Standard Transformer encoder (multi-head self-attention)
- Same overall structure as V1 (6 layers, cross-attention decoder)
- Positional encoding for sequence information
- Efficient implementation with Flash Attention if available

Expected results:
- Better or comparable quality to MAMBA (global attention)
- Quadratic complexity O(N²) vs MAMBA's linear O(N)
- Benchmark to validate MAMBA's efficiency claims
"""
import sys
import os

# Add repository root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
v4_dir = os.path.dirname(script_dir)  # v4/
repo_root = os.path.dirname(v4_dir)   # MambaFlowMatching/
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
from datetime import datetime

from core.neural_fields.perceiver import FourierFeatures
from core.sparse.cifar10_sparse import SparseCIFAR10Dataset
from core.sparse.metrics import MetricsTracker

# Import sampling functions from V1
sys.path.insert(0, os.path.join(repo_root, 'v1', 'training'))
from train_mamba_standalone import (
    SinusoidalTimeEmbedding,
    conditional_flow, target_velocity, heun_sample, sde_sample, ddim_sample
)


# ============================================================================
# Transformer Components
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequences

    Adds position information to token embeddings using sin/cos functions
    at different frequencies (standard Transformer approach)
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (B, N, d_model) input tokens
        Returns:
            x: (B, N, d_model) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    Standard Transformer encoder layer

    Architecture:
    1. Multi-head self-attention
    2. Add & Norm (residual connection)
    3. Feed-forward network (MLP)
    4. Add & Norm (residual connection)
    """
    def __init__(self, d_model, num_heads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()

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

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            x: (B, N, d_model) input sequence
            src_mask: Optional attention mask
            src_key_padding_mask: Optional padding mask

        Returns:
            x: (B, N, d_model) output sequence
        """
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        x = residual + self.dropout(attn_out)

        # Feed-forward with residual
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        return x


class TransformerEncoder(nn.Module):
    """
    Stack of Transformer encoder layers

    Replaces MAMBA blocks with standard Transformer encoder
    O(N²) complexity for self-attention vs O(N) for MAMBA
    """
    def __init__(self, d_model, num_layers=6, num_heads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, src_key_padding_mask=None):
        """
        Args:
            x: (B, N, d_model) input sequence

        Returns:
            x: (B, N, d_model) encoded sequence
        """
        for layer in self.layers:
            x = layer(x, mask, src_key_padding_mask)

        return self.norm(x)


# ============================================================================
# Transformer Diffusion Model
# ============================================================================

class TransformerDiffusion(nn.Module):
    """
    V4: Transformer-based diffusion model for sparse neural fields

    Architecture comparison with V1:
    - MAMBA (V1): Linear complexity O(N), sequential processing
    - Transformer (V4): Quadratic complexity O(N²), parallel with global attention

    Same structure as V1:
    - Fourier features for coordinates
    - 6 encoder layers
    - Single cross-attention layer
    - Same decoder

    Key differences:
    - Transformer encoder instead of MAMBA blocks
    - Positional encoding for sequence position
    - Multi-head self-attention for global context
    """
    def __init__(
        self,
        num_fourier_feats=256,
        d_model=512,
        num_layers=6,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads

        print(f"\nInitializing Transformer Diffusion V4:")
        print(f"  d_model: {d_model}")
        print(f"  num_layers: {num_layers}")
        print(f"  num_heads: {num_heads}")
        print(f"  dim_feedforward: {dim_feedforward}")

        # Fourier features (same as V1)
        self.fourier = FourierFeatures(coord_dim=2, num_freqs=num_fourier_feats, scale=10.0)
        feat_dim = num_fourier_feats * 2

        # Project inputs and queries
        self.input_proj = nn.Linear(feat_dim + 3, d_model)
        self.query_proj = nn.Linear(feat_dim + 3, d_model)

        # Time embedding (same as V1)
        self.time_embed = SinusoidalTimeEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # NEW: Positional encoding for sequence position
        self.pos_encoder = PositionalEncoding(d_model, max_len=5000, dropout=dropout)

        # NEW: Transformer encoder (replaces MAMBA blocks)
        self.transformer_encoder = TransformerEncoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # Cross-attention (same as V1)
        self.query_cross_attn = nn.MultiheadAttention(
            d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Output decoder (same as V1)
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3)
        )

    def forward(self, noisy_values, query_coords, t, input_coords, input_values):
        """
        Forward pass - same interface as V1 for compatibility

        Args:
            noisy_values: (B, N_out, 3)
            query_coords: (B, N_out, 2)
            t: (B,) timestep
            input_coords: (B, N_in, 2)
            input_values: (B, N_in, 3)
        """
        B = query_coords.shape[0]
        N_in = input_coords.shape[1]
        N_out = query_coords.shape[1]

        # Time embedding
        t_emb = self.time_mlp(self.time_embed(t))  # (B, d_model)

        # Fourier features
        input_feats = self.fourier(input_coords)  # (B, N_in, feat_dim)
        query_feats = self.fourier(query_coords)  # (B, N_out, feat_dim)

        # Encode inputs and queries
        input_tokens = self.input_proj(
            torch.cat([input_feats, input_values], dim=-1)
        )  # (B, N_in, d_model)

        query_tokens = self.query_proj(
            torch.cat([query_feats, noisy_values], dim=-1)
        )  # (B, N_out, d_model)

        # Add time embedding
        input_tokens = input_tokens + t_emb.unsqueeze(1)
        query_tokens = query_tokens + t_emb.unsqueeze(1)

        # Concatenate inputs and queries as sequence
        seq = torch.cat([input_tokens, query_tokens], dim=1)  # (B, N_in+N_out, d_model)

        # NEW: Add positional encoding
        seq = self.pos_encoder(seq)

        # NEW: Process through Transformer encoder (replaces MAMBA)
        seq = self.transformer_encoder(seq)

        # Split back into input and query sequences
        input_seq = seq[:, :N_in, :]  # (B, N_in, d_model)
        query_seq = seq[:, N_in:, :]  # (B, N_out, d_model)

        # Cross-attention: queries attend to processed inputs
        output, _ = self.query_cross_attn(query_seq, input_seq, input_seq)

        # Decode to RGB
        return self.decoder(output)


# ============================================================================
# Training Loop (Same as V1)
# ============================================================================

def train_flow_matching(
    model, train_loader, test_loader, epochs=1000, lr=1e-4, device='cuda',
    visualize_every=50, eval_every=10, save_every=10, save_dir='checkpoints_transformer_v4'
):
    """
    Train with flow matching (same as V1)
    """
    os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    losses = []

    # Track best model
    best_val_loss = float('inf')

    # Create full coordinate grid for visualization
    y, x = torch.meshgrid(
        torch.linspace(0, 1, 32),
        torch.linspace(0, 1, 32),
        indexing='ij'
    )
    full_coords = torch.stack([x.flatten(), y.flatten()], dim=-1).to(device)

    # Get viz batch
    viz_batch = next(iter(test_loader))
    viz_input_coords = viz_batch['input_coords'][:4].to(device)
    viz_input_values = viz_batch['input_values'][:4].to(device)
    viz_target_values = viz_batch['output_values'][:4].to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            input_coords = batch['input_coords'].to(device)
            input_values = batch['input_values'].to(device)
            output_coords = batch['output_coords'].to(device)
            output_values = batch['output_values'].to(device)

            # Flow matching
            t = torch.rand(input_coords.shape[0], device=device)
            x_0 = torch.randn_like(output_values)
            x_1 = output_values
            x_t = conditional_flow(x_0, x_1, t.view(-1, 1, 1))
            v_target = target_velocity(x_0, x_1)

            # Predict velocity
            v_pred = model(x_t, output_coords, t, input_coords, input_values)

            # Loss
            loss = F.mse_loss(v_pred, v_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        scheduler.step()

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}", flush=True)

        # Evaluation
        if (epoch + 1) % eval_every == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    input_coords = batch['input_coords'].to(device)
                    input_values = batch['input_values'].to(device)
                    output_coords = batch['output_coords'].to(device)
                    output_values = batch['output_values'].to(device)

                    t = torch.rand(input_coords.shape[0], device=device)
                    x_0 = torch.randn_like(output_values)
                    x_1 = output_values
                    x_t = conditional_flow(x_0, x_1, t.view(-1, 1, 1))
                    v_target = target_velocity(x_0, x_1)

                    v_pred = model(x_t, output_coords, t, input_coords, input_values)
                    loss = F.mse_loss(v_pred, v_target)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(test_loader)
            print(f"  Validation Loss: {avg_val_loss:.6f}", flush=True)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, os.path.join(save_dir, 'transformer_v4_best.pth'))
                print(f"  ✓ Saved best model (val_loss: {best_val_loss:.6f})", flush=True)

        # Periodic checkpoint
        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(save_dir, f'transformer_v4_epoch_{epoch+1:04d}.pth'))

        # Visualize
        if (epoch + 1) % visualize_every == 0:
            model.eval()
            with torch.no_grad():
                full_coords_batch = full_coords.unsqueeze(0).repeat(4, 1, 1)
                samples = heun_sample(
                    model, full_coords_batch, viz_input_coords, viz_input_values,
                    num_steps=50, device=device
                )

                fig, axes = plt.subplots(4, 3, figsize=(9, 12))
                for i in range(4):
                    # Input
                    input_img = torch.zeros(32, 32, 3)
                    for j in range(viz_input_coords.shape[1]):
                        coord = viz_input_coords[i, j]
                        y_idx = int(coord[1].item() * 31)
                        x_idx = int(coord[0].item() * 31)
                        input_img[y_idx, x_idx] = viz_input_values[i, j].cpu()
                    axes[i, 0].imshow(input_img.numpy())
                    axes[i, 0].set_title('Input (20%)')
                    axes[i, 0].axis('off')

                    # Generated
                    gen_img = samples[i].reshape(32, 32, 3).cpu().numpy()
                    axes[i, 1].imshow(np.clip(gen_img, 0, 1))
                    axes[i, 1].set_title('Generated')
                    axes[i, 1].axis('off')

                    # Ground truth
                    gt_img = viz_target_values[i].reshape(32, 32, 3).cpu().numpy()
                    axes[i, 2].imshow(np.clip(gt_img, 0, 1))
                    axes[i, 2].set_title('Ground Truth')
                    axes[i, 2].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'samples_epoch_{epoch+1:04d}.png'))
                plt.close()

    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, os.path.join(save_dir, 'transformer_v4_latest.pth'))

    return losses


def main():
    parser = argparse.ArgumentParser(description='Train Transformer Diffusion V4')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_every', type=int, default=10, help='Save every N epochs')
    parser.add_argument('--eval_every', type=int, default=10, help='Evaluate every N epochs')
    parser.add_argument('--visualize_every', type=int, default=50, help='Visualize every N epochs')
    parser.add_argument('--save_dir', type=str, default='checkpoints_transformer_v4', help='Save directory')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='FFN dimension')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    args = parser.parse_args()

    # Device setup
    print("=" * 60)
    print("DEVICE SETUP")
    print("=" * 60)

    if args.device == 'cpu':
        device = torch.device('cpu')
        print("✓ Using CPU")
    elif args.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("⚠️  CUDA not available, falling back to CPU")
    else:  # auto
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("⚠️  Using CPU")
    print("=" * 60 + "\n")

    # Load dataset
    print("Loading CIFAR-10 dataset...")
    train_dataset = SparseCIFAR10Dataset(
        root='../data', train=True, input_ratio=0.2, output_ratio=0.2, download=True, seed=42
    )
    test_dataset = SparseCIFAR10Dataset(
        root='../data', train=False, input_ratio=0.2, output_ratio=0.2, download=True, seed=42
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}\n")

    # Initialize model
    print("=" * 60)
    print("TRANSFORMER DIFFUSION V4")
    print("=" * 60)
    print("Architecture comparison:")
    print("  V1 (MAMBA): Linear complexity O(N), sequential state propagation")
    print("  V4 (Transformer): Quadratic complexity O(N²), parallel global attention")
    print("")
    print("Expected trade-offs:")
    print("  ✓ Better quality from global attention")
    print("  ✗ Slower training (quadratic complexity)")
    print("  ✓ Standard architecture (well-understood)")
    print("=" * 60 + "\n")

    model = TransformerDiffusion(
        num_fourier_feats=256,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=0.1
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}\n")

    # Train
    print("Starting training...")
    losses = train_flow_matching(
        model, train_loader, test_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        visualize_every=args.visualize_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        save_dir=args.save_dir
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Checkpoints saved to: {args.save_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
