"""
MAMBA Diffusion Training Script - Standalone Version
Exactly matches the original mamba_diffusion.ipynb implementation
"""
import sys
import os

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

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

# ============================================================================
# SSM Components (Exactly from notebook)
# ============================================================================

class SSMBlockFast(nn.Module):
    """
    Ultra-fast SSM using cumulative scan

    Eliminates ALL Python loops for maximum speed
    """
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # State space parameters
        self.A_log = nn.Parameter(torch.randn(d_state) * 0.1 - 1.0)
        self.B = nn.Linear(d_model, d_state, bias=False)
        self.C = nn.Linear(d_state, d_model, bias=False)
        self.D = nn.Parameter(torch.randn(d_model) * 0.01)

        nn.init.xavier_uniform_(self.B.weight, gain=0.5)
        nn.init.xavier_uniform_(self.C.weight, gain=0.5)

        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.eps = 1e-8

    def forward(self, x):
        """
        Fully vectorized - uses einsum for maximum speed

        Args:
            x: (B, N, d_model)
        Returns:
            y: (B, N, d_model)
        """
        B, N, D = x.shape

        # Discretization
        A = -torch.exp(self.A_log).clamp(min=self.eps, max=10.0)
        dt = 1.0 / N
        A_bar = torch.exp(dt * A)
        B_bar = torch.where(
            torch.abs(A) > self.eps,
            (A_bar - 1.0) / (A + self.eps),
            torch.ones_like(A) * dt
        )

        # Input projection (vectorized)
        Bu = self.B(x) * B_bar  # (B, N, d_state)

        # Sequential computation (optimized with torch operations)
        # Create exponential decay matrix
        indices = torch.arange(N, device=x.device)
        decay = A_bar.unsqueeze(0).pow(
            (indices.unsqueeze(0) - indices.unsqueeze(1)).clamp(min=0).unsqueeze(-1)
        )  # (N, N, d_state)

        # Mask to only include i >= j (causal)
        mask = indices.unsqueeze(0) >= indices.unsqueeze(1)  # (N, N)
        decay = decay * mask.unsqueeze(-1).float()  # (N, N, d_state)

        # Compute all states: h[t] = sum_{s<=t} decay[t,s] * Bu[s]
        h = torch.einsum('nmd,bnd->bmd', decay, Bu)  # (B, N, d_state)
        h = torch.clamp(h, min=-10.0, max=10.0)

        # Output
        y = self.C(h) + self.D * x

        # Gating and residual
        gate = self.gate(x)
        y = gate * y + (1 - gate) * x

        return self.dropout(self.norm(y))


class MambaBlock(nn.Module):
    """Complete Mamba block with FAST SSM + MLP"""
    def __init__(self, d_model, d_state=16, expand_factor=2, dropout=0.1):
        super().__init__()

        # Expand
        self.proj_in = nn.Linear(d_model, d_model * expand_factor)

        # Use the FAST SSM implementation
        self.ssm = SSMBlockFast(d_model * expand_factor, d_state, dropout)

        # Contract
        self.proj_out = nn.Linear(d_model * expand_factor, d_model)

        # MLP
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # SSM branch
        residual = x
        x = self.proj_in(x)
        x = self.ssm(x)
        x = self.proj_out(x)
        x = x + residual

        # MLP branch
        x = x + self.mlp(x)

        return x


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class MAMBADiffusion(nn.Module):
    """
    State space model for sparse field diffusion

    Key features:
    - Linear complexity (vs quadratic for attention)
    - State propagation for long-range dependencies
    - Efficient sequential processing
    """
    def __init__(
        self,
        num_fourier_feats=256,
        d_model=512,
        num_layers=6,
        d_state=16,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model

        # Fourier features
        self.fourier = FourierFeatures(coord_dim=2, num_freqs=num_fourier_feats, scale=10.0)
        feat_dim = num_fourier_feats * 2  # FourierFeatures outputs 2*num_freqs (sin + cos)

        # Project inputs and queries
        self.input_proj = nn.Linear(feat_dim + 3, d_model)
        self.query_proj = nn.Linear(feat_dim + 3, d_model)

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # Mamba blocks for sequence processing
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, expand_factor=2, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Cross-attention to extract query-specific features
        self.query_cross_attn = nn.MultiheadAttention(
            d_model, num_heads=8, dropout=dropout, batch_first=True
        )

        # Output decoder
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

        # Process through Mamba blocks (SSM)
        for mamba_block in self.mamba_blocks:
            seq = mamba_block(seq)

        # Split back into input and query sequences
        input_seq = seq[:, :N_in, :]  # (B, N_in, d_model)
        query_seq = seq[:, N_in:, :]  # (B, N_out, d_model)

        # Cross-attention: queries attend to processed inputs
        output, _ = self.query_cross_attn(query_seq, input_seq, input_seq)

        # Decode to RGB
        return self.decoder(output)


# ============================================================================
# Flow Matching (Exactly from notebook)
# ============================================================================

def conditional_flow(x_0, x_1, t):
    """Linear interpolation: (1-t)*x_0 + t*x_1"""
    return (1 - t) * x_0 + t * x_1


def target_velocity(x_0, x_1):
    """Target velocity: x_1 - x_0"""
    return x_1 - x_0


@torch.no_grad()
def heun_sample(model, output_coords, input_coords, input_values, num_steps=50, device='cuda'):
    """Heun ODE solver for flow matching"""
    B, N_out = output_coords.shape[0], output_coords.shape[1]
    x_t = torch.randn(B, N_out, 3, device=device)

    dt = 1.0 / num_steps
    ts = torch.linspace(0, 1 - dt, num_steps)

    for t_val in ts:
        t = torch.full((B,), t_val.item(), device=device)
        t_next = torch.full((B,), t_val.item() + dt, device=device)

        v1 = model(x_t, output_coords, t, input_coords, input_values)
        x_next_pred = x_t + dt * v1

        v2 = model(x_next_pred, output_coords, t_next, input_coords, input_values)
        x_t = x_t + dt * 0.5 * (v1 + v2)

    return torch.clamp(x_t, 0, 1)


@torch.no_grad()
def sde_sample(model, output_coords, input_coords, input_values,
               num_steps=50, temperature=0.5, device='cuda'):
    """
    SDE sampling with Langevin dynamics for smoother results

    Key improvements over ODE sampling:
    - Adds stochastic correction at each step
    - Reduces speckled artifacts and background noise
    - Temperature controls exploration vs exploitation
    - Annealed noise schedule for quality

    Args:
        model: MAMBA diffusion model
        output_coords: (B, N_out, 2) query coordinates
        input_coords: (B, N_in, 2) sparse input coordinates
        input_values: (B, N_in, 3) sparse input RGB values
        num_steps: Number of sampling steps (default: 50)
        temperature: Noise scale, higher = more exploration (default: 0.5)
        device: Device to run on

    Returns:
        Generated RGB values: (B, N_out, 3)
    """
    B, N_out = output_coords.shape[0], output_coords.shape[1]
    x_t = torch.randn(B, N_out, 3, device=device)

    dt = 1.0 / num_steps
    ts = torch.linspace(0, 1 - dt, num_steps)

    for i, t_val in enumerate(ts):
        t = torch.full((B,), t_val.item(), device=device)

        # Predict velocity
        v_pred = model(x_t, output_coords, t, input_coords, input_values)

        # Deterministic step (flow matching)
        x_t = x_t + dt * v_pred

        # Stochastic correction (Langevin dynamics)
        # Don't add noise in final steps for clean output
        if i < num_steps - 5:
            # Annealed noise: decreases over time for quality
            noise_scale = temperature * np.sqrt(dt) * (1 - t_val.item())
            noise = torch.randn_like(x_t) * noise_scale
            x_t = x_t + noise

    return torch.clamp(x_t, 0, 1)


@torch.no_grad()
def ddim_sample(model, output_coords, input_coords, input_values,
                num_steps=50, eta=0.0, device='cuda'):
    """
    DDIM-style sampling with non-uniform timesteps for flow matching

    Key improvements:
    - Non-uniform schedule: more steps early in denoising
    - Configurable stochasticity via eta parameter
    - Faster convergence with fewer steps
    - eta=0: deterministic, eta=1: more stochastic

    Args:
        model: MAMBA diffusion model
        output_coords: (B, N_out, 2) query coordinates
        input_coords: (B, N_in, 2) sparse input coordinates
        input_values: (B, N_in, 3) sparse input RGB values
        num_steps: Number of sampling steps (default: 50)
        eta: Stochasticity parameter, 0=deterministic, 1=stochastic (default: 0.0)
        device: Device to run on

    Returns:
        Generated RGB values: (B, N_out, 3)
    """
    B, N_out = output_coords.shape[0], output_coords.shape[1]
    x_t = torch.randn(B, N_out, 3, device=device)

    # Non-uniform timestep schedule (quadratic: more steps early)
    timesteps = torch.pow(torch.linspace(0, 1, num_steps, device=device), 2)

    for i in range(len(timesteps) - 1):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t_curr

        t = torch.full((B,), t_curr.item(), device=device)

        # Predict velocity
        v_pred = model(x_t, output_coords, t, input_coords, input_values)

        # DDIM step
        x_pred = x_t + dt * v_pred

        # Optional stochasticity
        if eta > 0 and i < len(timesteps) - 2:
            sigma = eta * torch.sqrt(dt)
            x_t = x_pred + sigma * torch.randn_like(x_pred)
        else:
            x_t = x_pred

    return torch.clamp(x_t, 0, 1)


# ============================================================================
# Training Loop (Modified for long training + periodic saves)
# ============================================================================

def train_flow_matching(
    model, train_loader, test_loader, epochs=1000, lr=1e-4, device='cuda',
    visualize_every=50, eval_every=10, save_every=10, save_dir='checkpoints_mamba'
):
    """
    Train with flow matching

    Args:
        model: MAMBA diffusion model
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Number of training epochs (default: 1000)
        lr: Learning rate
        device: Device to train on
        visualize_every: Visualize every N epochs
        eval_every: Evaluate every N epochs
        save_every: Save checkpoint every N epochs (default: 10)
        save_dir: Directory to save checkpoints
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
    viz_batch = next(iter(train_loader))
    viz_input_coords = viz_batch['input_coords'][:4].to(device)
    viz_input_values = viz_batch['input_values'][:4].to(device)
    viz_output_coords = viz_batch['output_coords'][:4].to(device)
    viz_output_values = viz_batch['output_values'][:4].to(device)
    viz_full_images = viz_batch['full_image'][:4].to(device)
    viz_input_indices = viz_batch['input_indices'][:4]

    # Training log
    log_file = os.path.join(save_dir, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write(f"Training started at {datetime.now()}\n")
        f.write(f"Epochs: {epochs}, LR: {lr}, Save every: {save_every}\n")
        f.write("="*60 + "\n")

    print(f"\n{'='*60}")
    print(f"Training MAMBA Diffusion for {epochs} epochs")
    print(f"Save directory: {save_dir}")
    print(f"Checkpoints will be saved every {save_every} epochs")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            input_coords = batch['input_coords'].to(device)
            input_values = batch['input_values'].to(device)
            output_coords = batch['output_coords'].to(device)
            output_values = batch['output_values'].to(device)

            B = input_coords.shape[0]
            t = torch.rand(B, device=device)

            x_0 = torch.randn_like(output_values)
            x_1 = output_values

            t_broadcast = t.view(B, 1, 1)
            x_t = conditional_flow(x_0, x_1, t_broadcast)
            u_t = target_velocity(x_0, x_1)

            v_pred = model(x_t, output_coords, t, input_coords, input_values)
            loss = F.mse_loss(v_pred, u_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        scheduler.step()

        log_msg = f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}, LR = {scheduler.get_last_lr()[0]:.6f}"
        print(log_msg, flush=True)  # flush=True ensures immediate output

        with open(log_file, 'a') as f:
            f.write(log_msg + "\n")
            f.flush()  # Ensure logs are written immediately

        # Evaluation
        val_loss = None
        if (epoch + 1) % eval_every == 0 or epoch == 0:
            model.eval()
            tracker = MetricsTracker()
            val_loss_accum = 0
            val_batches = 0
            with torch.no_grad():
                for i, batch in enumerate(test_loader):
                    if i >= 10:
                        break
                    pred_values = heun_sample(
                        model, batch['output_coords'].to(device),
                        batch['input_coords'].to(device), batch['input_values'].to(device),
                        num_steps=50, device=device
                    )
                    tracker.update(pred_values, batch['output_values'].to(device))
                    val_loss_accum += F.mse_loss(pred_values, batch['output_values'].to(device)).item()
                    val_batches += 1

                results = tracker.compute()
                val_loss = val_loss_accum / val_batches
                eval_msg = f"  Eval - MSE: {results['mse']:.6f}, MAE: {results['mae']:.6f}, Val Loss: {val_loss:.6f}"
                print(eval_msg, flush=True)

                with open(log_file, 'a') as f:
                    f.write(eval_msg + "\n")
                    f.flush()

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': avg_loss,
                        'val_loss': val_loss,
                        'best_val_loss': best_val_loss
                    }, os.path.join(save_dir, 'mamba_best.pth'))
                    best_msg = f"  ✓ Saved best model (val_loss: {val_loss:.6f})"
                    print(best_msg, flush=True)
                    with open(log_file, 'a') as f:
                        f.write(best_msg + "\n")
                        f.flush()

        # Save periodic checkpoint (every save_every epochs)
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(save_dir, f'mamba_epoch_{epoch+1:04d}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'val_loss': val_loss if val_loss is not None else avg_loss,
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            save_msg = f"  ✓ Saved checkpoint: {checkpoint_path}"
            print(save_msg, flush=True)
            with open(log_file, 'a') as f:
                f.write(save_msg + "\n")
                f.flush()

        # Save latest model (always)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'val_loss': val_loss if val_loss is not None else avg_loss,
            'best_val_loss': best_val_loss
        }, os.path.join(save_dir, 'mamba_latest.pth'))

        # Visualization
        if (epoch + 1) % visualize_every == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                # Sparse output prediction
                pred_values = heun_sample(
                    model, viz_output_coords, viz_input_coords, viz_input_values,
                    num_steps=50, device=device
                )

                # Full field reconstruction
                full_coords_batch = full_coords.unsqueeze(0).expand(4, -1, -1)
                full_pred_values = heun_sample(
                    model, full_coords_batch, viz_input_coords, viz_input_values,
                    num_steps=50, device=device
                )
                full_pred_images = full_pred_values.view(4, 32, 32, 3).permute(0, 3, 1, 2)

                # Create visualization
                fig, axes = plt.subplots(4, 5, figsize=(20, 16))

                for i in range(4):
                    # Ground truth
                    axes[i, 0].imshow(viz_full_images[i].permute(1, 2, 0).cpu().numpy())
                    axes[i, 0].set_title('Ground Truth' if i == 0 else '', fontsize=10)
                    axes[i, 0].axis('off')

                    # Sparse input
                    input_img = torch.zeros(3, 32, 32, device=device)
                    input_idx = viz_input_indices[i]
                    input_img.view(3, -1)[:, input_idx] = viz_input_values[i].T
                    axes[i, 1].imshow(input_img.permute(1, 2, 0).cpu().numpy())
                    axes[i, 1].set_title('Sparse Input (20%)' if i == 0 else '', fontsize=10)
                    axes[i, 1].axis('off')

                    # Sparse output target
                    target_img = torch.zeros(3, 32, 32, device=device)
                    output_idx = viz_batch['output_indices'][i]
                    target_img.view(3, -1)[:, output_idx] = viz_output_values[i].T
                    axes[i, 2].imshow(target_img.permute(1, 2, 0).cpu().numpy())
                    axes[i, 2].set_title('Sparse Target (20%)' if i == 0 else '', fontsize=10)
                    axes[i, 2].axis('off')

                    # Sparse prediction
                    pred_img = torch.zeros(3, 32, 32, device=device)
                    pred_img.view(3, -1)[:, output_idx] = pred_values[i].T
                    axes[i, 3].imshow(np.clip(pred_img.permute(1, 2, 0).cpu().numpy(), 0, 1))
                    axes[i, 3].set_title('Sparse Prediction' if i == 0 else '', fontsize=10)
                    axes[i, 3].axis('off')

                    # Full field reconstruction
                    axes[i, 4].imshow(np.clip(full_pred_images[i].permute(1, 2, 0).cpu().numpy(), 0, 1))
                    axes[i, 4].set_title('Full Field Recon' if i == 0 else '', fontsize=10)
                    axes[i, 4].axis('off')

                plt.suptitle(f'MAMBA - Epoch {epoch+1} (Best Val: {best_val_loss:.6f})',
                           fontsize=14, y=0.995)
                plt.tight_layout()
                viz_path = os.path.join(save_dir, f'mamba_epoch_{epoch+1:04d}.png')
                plt.savefig(viz_path, dpi=150, bbox_inches='tight')
                plt.close()

    final_msg = f"\n✓ Training complete! Best validation loss: {best_val_loss:.6f}"
    print(final_msg)
    print(f"  Best model: {save_dir}/mamba_best.pth")
    print(f"  Latest model: {save_dir}/mamba_latest.pth")

    with open(log_file, 'a') as f:
        f.write("="*60 + "\n")
        f.write(final_msg + "\n")
        f.write(f"Training completed at {datetime.now()}\n")

    return losses


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train MAMBA Diffusion Model')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--eval_every', type=int, default=10, help='Evaluate every N epochs (default: 10)')
    parser.add_argument('--visualize_every', type=int, default=50, help='Visualize every N epochs (default: 50)')
    parser.add_argument('--save_dir', type=str, default='checkpoints_mamba', help='Save directory')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension (default: 512)')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of layers (default: 6)')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers (default: 4)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Device to use: auto (default), cuda, or cpu')
    args = parser.parse_args()

    # Device setup with detailed diagnostics
    print("=" * 60, flush=True)
    print("DEVICE SETUP", flush=True)
    print("=" * 60, flush=True)
    print(f"PyTorch version: {torch.__version__}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"Requested device: {args.device}", flush=True)

    # Determine device based on args
    if args.device == 'cpu':
        device = torch.device('cpu')
        print(f"\n✓ Using CPU (forced by --device cpu)", flush=True)
    elif args.device == 'cuda':
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
                print(f"    Compute Capability: {props.major}.{props.minor}")
            device = torch.device('cuda:0')
            torch.cuda.set_device(0)
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            print(f"\n✓ Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
        else:
            print(f"\n❌ ERROR: --device cuda specified but CUDA not available!")
            print("   Falling back to CPU")
            device = torch.device('cpu')
    else:  # auto
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
                print(f"    Compute Capability: {props.major}.{props.minor}")
            device = torch.device('cuda:0')
            torch.cuda.set_device(0)
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            print(f"\n✓ Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
        else:
            device = torch.device('cpu')
            print(f"\n⚠️  CUDA not available - falling back to CPU")
            print("   To use GPU, ensure:")
            print("   1. NVIDIA GPU is installed")
            print("   2. CUDA drivers are installed")
            print("   3. PyTorch with CUDA support is installed")
            print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
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
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Initialize model
    print("Initializing MAMBA model...")
    model = MAMBADiffusion(
        num_fourier_feats=256,
        d_model=args.d_model,
        num_layers=args.num_layers,
        d_state=16,
        dropout=0.1
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Train
    print("\nStarting training...")
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

    # Save loss curve
    plt.figure(figsize=(10, 4))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss: MAMBA Diffusion')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(args.save_dir, 'training_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
