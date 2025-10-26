"""
Transformer V2 Flow Matching on Synthetic Sinusoidal Data

Uses standard Transformer encoder with multi-head attention instead of MAMBA.
Comparison architecture to test if attention helps with sparse reconstruction.
"""
import sys
import os

# Add repository root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
synthetic_dir = os.path.dirname(script_dir)
repo_root = os.path.dirname(synthetic_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import math

from synthetic_experiments.datasets import SinusoidalDataset


# ============================================================================
# Transformer Components
# ============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Time embedding for flow matching"""
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,)
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class TransformerEncoderLayer(nn.Module):
    """Standard Transformer encoder layer"""
    def __init__(self, d_model, num_heads=8, dim_feedforward=512, dropout=0.1):
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

    def forward(self, x):
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x)
        x = residual + self.dropout(attn_out)

        # Feed-forward with residual
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        return x


class TransformerFlow(nn.Module):
    """
    Transformer Flow Matching for Synthetic Data

    Architecture:
    - Fourier features for coordinates
    - Transformer encoder (self-attention)
    - Cross-attention decoder
    - MLP output head
    """
    def __init__(
        self,
        d_model=128,
        num_layers=4,
        num_heads=8,
        dim_feedforward=512,
        num_frequencies=16,
        dropout=0.1
    ):
        super().__init__()

        # Fourier features for coordinate encoding
        self.num_frequencies = num_frequencies
        self.register_buffer(
            'B',
            torch.randn(2, num_frequencies) * 10.0
        )

        # Projections
        coord_feat_dim = 2 * num_frequencies  # sin + cos
        self.query_proj = nn.Linear(coord_feat_dim + 1, d_model)
        self.input_proj = nn.Linear(coord_feat_dim + 1, d_model)

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(d_model)

        # Transformer encoder
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Cross-attention
        self.query_cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def fourier_features(self, coords):
        """Fourier feature encoding"""
        proj = coords @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, sparse_coords, sparse_values, query_coords, t, query_values=None):
        """
        Args:
            sparse_coords: (B, N_sparse, 2) observed coordinates
            sparse_values: (B, N_sparse, 1) observed values
            query_coords: (B, N_query, 2) query coordinates
            t: (B,) time in [0, 1]
            query_values: (B, N_query, 1) optional noisy query values

        Returns:
            predicted_values: (B, N_query, 1)
        """
        B = sparse_coords.shape[0]

        # Encode coordinates
        sparse_feats = self.fourier_features(sparse_coords)
        query_feats = self.fourier_features(query_coords)

        # Project with values
        input_tokens = self.input_proj(
            torch.cat([sparse_feats, sparse_values], dim=-1)
        )

        # Use provided query_values or generate noise
        if query_values is None:
            query_values = torch.randn(B, query_coords.shape[1], 1, device=query_coords.device)

        query_tokens = self.query_proj(
            torch.cat([query_feats, query_values], dim=-1)
        )

        # Time conditioning
        t_embed = self.time_embed(t)
        query_tokens = query_tokens + t_embed.unsqueeze(1)
        input_tokens = input_tokens + t_embed.unsqueeze(1)

        # Concatenate sequences
        seq = torch.cat([input_tokens, query_tokens], dim=1)

        # Process with Transformer
        for transformer_layer in self.transformer_layers:
            seq = transformer_layer(seq)

        # Split back
        N_sparse = sparse_coords.shape[1]
        input_seq = seq[:, :N_sparse, :]
        query_seq = seq[:, N_sparse:, :]

        # Cross-attention: queries attend to inputs
        attended, _ = self.query_cross_attn(query_seq, input_seq, input_seq)

        # Decode to values
        output = self.decoder(attended)

        return output


# ============================================================================
# Flow Matching Training (same as V1)
# ============================================================================

def conditional_flow(x0, x1, t):
    """Linear interpolation flow from x0 to x1"""
    t = t.view(-1, 1, 1)
    z_t = t * x1 + (1 - t) * x0
    v_t = x1 - x0
    return z_t, v_t


@torch.no_grad()
def visualize_reconstruction(model, train_coords, train_values, full_data, H, W, device, epoch, save_dir):
    """Visualize reconstruction on first 4 samples"""
    model.eval()

    # Create query coordinates for full field
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing='ij'
    )
    query_coords = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1).unsqueeze(0)

    fig, axes = plt.subplots(4, 3, figsize=(12, 12))

    for idx in range(min(4, len(full_data))):
        # Get sample
        train_coords_sample = train_coords[idx:idx+1].to(device)
        train_values_sample = train_values[idx:idx+1].to(device)
        query_coords_sample = query_coords.to(device)
        target = full_data[idx:idx+1].to(device)

        # Sample reconstruction
        pred = sample_heun(model, train_coords_sample, train_values_sample,
                          query_coords_sample, num_steps=50, device=device)
        pred = pred.reshape(1, 1, H, W).cpu()
        target = target.cpu()

        # Sparse observations
        train_coords_vis = train_coords_sample[0].cpu()

        # Plot: Ground Truth | Reconstruction | Error
        axes[idx, 0].imshow(target[0, 0], cmap='viridis', vmin=0, vmax=1)
        axes[idx, 0].set_title(f'Sample {idx+1}: Ground Truth')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(pred[0, 0], cmap='viridis', vmin=0, vmax=1)
        coords_pixel = ((train_coords_vis + 1) / 2 * (W - 1)).numpy()
        axes[idx, 1].scatter(coords_pixel[:, 0], coords_pixel[:, 1],
                            c='red', s=5, alpha=0.5, marker='x')
        axes[idx, 1].set_title(f'Reconstruction (5% obs)')
        axes[idx, 1].axis('off')

        error = torch.abs(pred - target)
        im = axes[idx, 2].imshow(error[0, 0], cmap='hot', vmin=0, vmax=0.3)
        axes[idx, 2].set_title(f'Error (MSE: {error.mean():.4f})')
        axes[idx, 2].axis('off')
        plt.colorbar(im, ax=axes[idx, 2], fraction=0.046)

    plt.suptitle(f'Transformer V2 - Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'reconstruction_epoch_{epoch:04d}.png'), dpi=100)
    plt.close()

    model.train()


def train_flow_matching(
    model,
    dataset,
    num_epochs=100,
    batch_size=32,
    lr=1e-3,
    train_sparsity=0.05,
    test_sparsity=0.05,
    device='cpu',
    save_dir='checkpoints',
    visualize_every=50
):
    """Train Transformer flow matching"""

    os.makedirs(save_dir, exist_ok=True)
    vis_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Get train/test split
    train_coords, train_values, test_coords, test_values, full_data = dataset.get_train_test_split(
        train_sparsity=train_sparsity,
        test_sparsity=test_sparsity,
        strategy='random'
    )

    # Training dataset
    num_samples = len(full_data)
    indices = list(range(num_samples))
    H, W = dataset.resolution, dataset.resolution

    losses = []
    best_loss = float('inf')

    print(f"\n🚀 Training Transformer V2 Flow Matching")
    print(f"   Dataset: {dataset.complexity}, Samples: {num_samples}")
    print(f"   Train: {train_sparsity*100:.0f}% ({train_coords.shape[1]} pixels)")
    print(f"   Test: {test_sparsity*100:.0f}% ({test_coords.shape[1]} pixels, disjoint)")
    print(f"   Resolution: {H}x{W}")
    print(f"   Epochs: {num_epochs}, Batch size: {batch_size}\n")
    sys.stdout.flush()

    for epoch in range(num_epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        num_batches = 0

        # Mini-batch training
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            if len(batch_indices) < batch_size:
                continue

            # Get batch
            train_coords_batch = train_coords[batch_indices].to(device)
            train_values_batch = train_values[batch_indices].to(device)
            test_coords_batch = test_coords[batch_indices].to(device)
            test_values_batch = test_values[batch_indices].to(device)

            # Sample timestep
            t = torch.rand(len(batch_indices), device=device)

            # Flow matching
            noise = torch.randn_like(test_values_batch)
            z_t, v_t = conditional_flow(noise, test_values_batch, t)

            # Predict velocity
            pred_v = model(train_coords_batch, train_values_batch, test_coords_batch, t, query_values=z_t)

            # Loss
            loss = F.mse_loss(pred_v, v_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f}")
            sys.stdout.flush()

        # Visualize
        if visualize_every > 0 and (epoch + 1) % visualize_every == 0:
            print(f"   📊 Generating visualization...")
            sys.stdout.flush()
            visualize_reconstruction(
                model, train_coords, train_values, full_data,
                H, W, device, epoch + 1, vis_dir
            )

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(save_dir, 'best_model.pth'))

    print(f"\n✅ Training complete! Best loss: {best_loss:.6f}")
    sys.stdout.flush()

    return losses


@torch.no_grad()
def sample_heun(model, sparse_coords, sparse_values, query_coords, num_steps=50, device='cpu'):
    """Sample using Heun's ODE solver"""
    B = sparse_coords.shape[0]

    # Start from noise
    x = torch.randn(B, query_coords.shape[1], 1, device=device)

    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.ones(B, device=device) * (i / num_steps)

        # Predict velocity
        v = model(sparse_coords, sparse_values, query_coords, t, query_values=x)

        # Heun's method (2nd order)
        x_temp = x + dt * v
        t_next = t + dt

        v_next = model(sparse_coords, sparse_values, query_coords, t_next, query_values=x_temp)
        x = x + dt * 0.5 * (v + v_next)

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--complexity', type=str, default='simple',
                       choices=['simple', 'multi_frequency', 'radial', 'interference',
                               'modulated', 'composite'])
    parser.add_argument('--resolution', type=int, default=32)
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--noise_level', type=float, default=0.0)
    parser.add_argument('--train_sparsity', type=float, default=0.05)
    parser.add_argument('--test_sparsity', type=float, default=0.05)

    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dim_feedforward', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--visualize_every', type=int, default=50)

    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--save_dir', type=str, default='synthetic_experiments/baselines/checkpoints_v2')

    args = parser.parse_args()

    # Print startup
    print("=" * 70)
    print("Transformer V2 Synthetic Training - Starting")
    print("=" * 70)
    sys.stdout.flush()

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    sys.stdout.flush()

    # Create dataset
    print(f"\n📊 Creating dataset...")
    sys.stdout.flush()
    dataset = SinusoidalDataset(
        resolution=args.resolution,
        num_samples=args.num_samples,
        complexity=args.complexity,
        noise_level=args.noise_level
    )
    print(f"✅ Dataset created")
    sys.stdout.flush()

    # Create model
    print(f"\n🏗️  Creating Transformer model...")
    sys.stdout.flush()
    model = TransformerFlow(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    )

    print(f"📦 Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    sys.stdout.flush()

    # Train
    losses = train_flow_matching(
        model,
        dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_sparsity=args.train_sparsity,
        test_sparsity=args.test_sparsity,
        device=device,
        save_dir=args.save_dir,
        visualize_every=args.visualize_every
    )

    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Transformer V2 Training Loss - {args.complexity}')
    plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, 'training_curve.png'), dpi=150)
    plt.close()

    print(f"💾 Saved training curve to {args.save_dir}/training_curve.png")
    sys.stdout.flush()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: Training failed with exception:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
