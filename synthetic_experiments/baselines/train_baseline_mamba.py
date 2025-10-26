"""
Baseline MAMBA Flow Matching on Synthetic Sinusoidal Data

Simple baseline for comparing improvement methods
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
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from datetime import datetime

from synthetic_experiments.datasets import SinusoidalDataset


# ============================================================================
# Simple MAMBA Components (Lightweight for synthetic data)
# ============================================================================

class SimpleSSM(nn.Module):
    """Simplified SSM for synthetic experiments"""
    def __init__(self, d_model=128, d_state=8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.A_log = nn.Parameter(torch.randn(d_state) * 0.1 - 1.0)
        self.B = nn.Linear(d_model, d_state, bias=False)
        self.C = nn.Linear(d_state, d_model, bias=False)
        self.D = nn.Parameter(torch.randn(d_model) * 0.01)

    def forward(self, x):
        B, N, D = x.shape

        # Discretization
        A = -torch.exp(self.A_log).clamp(min=1e-8, max=10.0)
        dt = 1.0 / N
        A_bar = torch.exp(dt * A)

        # Input projection
        Bu = self.B(x)  # (B, N, d_state)

        # Sequential state computation (simplified)
        h = torch.zeros(B, self.d_state, device=x.device)
        outputs = []

        for t in range(N):
            h = A_bar * h + Bu[:, t]
            y_t = self.C(h) + self.D * x[:, t]
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (B, N, D)


class SimpleMambaBlock(nn.Module):
    """Lightweight MAMBA block"""
    def __init__(self, d_model=128):
        super().__init__()
        self.ssm = SimpleSSM(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Gated SSM
        residual = x
        x = self.norm(x)
        gate = self.gate(x)
        x = gate * self.ssm(x)
        return residual + x


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


class BaselineMAMBAFlow(nn.Module):
    """
    Baseline MAMBA Flow Matching for Synthetic Data

    Simple architecture:
    - Fourier features for coordinates
    - MAMBA encoder
    - Cross-attention
    - MLP decoder
    """
    def __init__(
        self,
        d_model=128,
        num_layers=4,
        fourier_dim=64,
        num_frequencies=16
    ):
        super().__init__()

        # Fourier features for coordinate encoding
        self.num_frequencies = num_frequencies
        self.register_buffer(
            'B',
            torch.randn(2, num_frequencies) * 10.0  # Random Fourier features
        )

        # Projections
        coord_feat_dim = 2 * num_frequencies  # sin + cos
        self.query_proj = nn.Linear(coord_feat_dim + 1, d_model)  # +1 for value
        self.input_proj = nn.Linear(coord_feat_dim + 1, d_model)

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(d_model)

        # MAMBA encoder
        self.mamba_blocks = nn.ModuleList([
            SimpleMambaBlock(d_model) for _ in range(num_layers)
        ])

        # Cross-attention
        self.query_cross_attn = nn.MultiheadAttention(
            d_model, num_heads=4, batch_first=True
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)  # Output single value
        )

    def fourier_features(self, coords):
        """
        Fourier feature encoding for coordinates

        Args:
            coords: (B, N, 2) coordinates in [-1, 1]

        Returns:
            features: (B, N, 2*num_frequencies)
        """
        # coords @ B: (B, N, 2) @ (2, num_frequencies) = (B, N, num_frequencies)
        proj = coords @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, sparse_coords, sparse_values, query_coords, t):
        """
        Args:
            sparse_coords: (B, N_sparse, 2) observed coordinates
            sparse_values: (B, N_sparse, 1) observed values
            query_coords: (B, N_query, 2) query coordinates
            t: (B,) time in [0, 1]

        Returns:
            predicted_values: (B, N_query, 1)
        """
        B = sparse_coords.shape[0]

        # Encode coordinates with Fourier features
        sparse_feats = self.fourier_features(sparse_coords)  # (B, N_sparse, 2*freq)
        query_feats = self.fourier_features(query_coords)    # (B, N_query, 2*freq)

        # Project with values
        input_tokens = self.input_proj(
            torch.cat([sparse_feats, sparse_values], dim=-1)
        )  # (B, N_sparse, D)

        # For flow matching, add noise to query values
        # At t=0: pure noise, at t=1: clean values
        noise = torch.randn(B, query_coords.shape[1], 1, device=query_coords.device)
        noisy_values = noise  # In flow matching, we start from noise

        query_tokens = self.query_proj(
            torch.cat([query_feats, noisy_values], dim=-1)
        )  # (B, N_query, D)

        # Time conditioning
        t_embed = self.time_embed(t)  # (B, D)
        query_tokens = query_tokens + t_embed.unsqueeze(1)
        input_tokens = input_tokens + t_embed.unsqueeze(1)

        # Concatenate sequences
        seq = torch.cat([input_tokens, query_tokens], dim=1)  # (B, N_sparse + N_query, D)

        # Process with MAMBA
        for mamba in self.mamba_blocks:
            seq = mamba(seq)

        # Split back
        N_sparse = sparse_coords.shape[1]
        input_seq = seq[:, :N_sparse, :]
        query_seq = seq[:, N_sparse:, :]

        # Cross-attention: queries attend to inputs
        attended, _ = self.query_cross_attn(query_seq, input_seq, input_seq)

        # Decode to values
        output = self.decoder(attended)  # (B, N_query, 1)

        return output


# ============================================================================
# Flow Matching Training
# ============================================================================

def conditional_flow(x0, x1, t):
    """
    Linear interpolation flow from x0 to x1

    Args:
        x0: noise (B, N, 1)
        x1: target values (B, N, 1)
        t: time (B,)

    Returns:
        z_t: interpolated state
        v_t: target velocity
    """
    t = t.view(-1, 1, 1)
    z_t = t * x1 + (1 - t) * x0
    v_t = x1 - x0  # Constant velocity
    return z_t, v_t


def train_flow_matching(
    model,
    dataset,
    num_epochs=100,
    batch_size=32,
    lr=1e-3,
    sparsity=0.2,
    device='cpu',
    save_dir='checkpoints'
):
    """Train baseline MAMBA flow matching"""

    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Get sparse observations once
    coords_sparse, values_sparse, full_data = dataset.get_sparse_observations(
        sparsity=sparsity,
        strategy='random'
    )

    # Create query coordinates (full grid)
    H, W = dataset.resolution, dataset.resolution
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing='ij'
    )
    query_coords = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)  # (H*W, 2)
    query_coords = query_coords.unsqueeze(0).expand(batch_size, -1, -1)  # (B, H*W, 2)

    # Training dataset
    num_samples = len(full_data)
    indices = list(range(num_samples))

    losses = []
    best_loss = float('inf')

    print(f"\n🚀 Training Baseline MAMBA Flow Matching")
    print(f"   Dataset: {dataset.complexity}, Samples: {num_samples}")
    print(f"   Sparsity: {sparsity*100:.0f}%, Resolution: {H}x{W}")
    print(f"   Epochs: {num_epochs}, Batch size: {batch_size}\n")

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
            coords_batch = coords_sparse[batch_indices].to(device)
            values_batch = values_sparse[batch_indices].to(device)
            target_batch = full_data[batch_indices].to(device)  # (B, 1, H, W)
            target_batch = target_batch.reshape(len(batch_indices), -1, 1)  # (B, H*W, 1)

            # Sample timestep
            t = torch.rand(len(batch_indices), device=device)

            # Flow matching
            noise = torch.randn_like(target_batch)
            z_t, v_t = conditional_flow(noise, target_batch, t)

            # Predict velocity
            pred_v = model(coords_batch, values_batch, query_coords[:len(batch_indices)].to(device), t)

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

        # Predict velocity at current point
        v = model(sparse_coords, sparse_values, query_coords, t)

        # Heun's method (2nd order)
        x_temp = x + dt * v
        t_next = t + dt

        v_next = model(sparse_coords, sparse_values, query_coords, t_next)
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
    parser.add_argument('--sparsity', type=float, default=0.2)

    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--save_dir', type=str, default='synthetic_experiments/baselines/checkpoints')

    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")

    # Create dataset
    print(f"\n📊 Creating dataset...")
    dataset = SinusoidalDataset(
        resolution=args.resolution,
        num_samples=args.num_samples,
        complexity=args.complexity,
        noise_level=args.noise_level
    )

    # Create model
    print(f"🏗️  Creating model...")
    model = BaselineMAMBAFlow(
        d_model=args.d_model,
        num_layers=args.num_layers
    )

    print(f"📦 Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Train
    losses = train_flow_matching(
        model,
        dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sparsity=args.sparsity,
        device=device,
        save_dir=args.save_dir
    )

    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss - {args.complexity}')
    plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, 'training_curve.png'), dpi=150)
    plt.close()

    print(f"💾 Saved training curve to {args.save_dir}/training_curve.png")


if __name__ == '__main__':
    main()
