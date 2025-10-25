"""
MAMBA Diffusion V2 - Improved Architecture for Reduced Speckle Artifacts

Key improvements:
1. Bidirectional MAMBA (4 forward + 4 backward = 8 total layers) for full context
2. Lightweight Perceiver (2 iterations) with query self-attention for spatial coherence
3. Increased depth for better spatial coherence

Expected improvements:
- 70-80% reduction in background speckles
- +3-5 dB PSNR improvement
- Smoother, more coherent spatial fields
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

# Import original components
from train_mamba_standalone import (
    SSMBlockFast, MambaBlock, SinusoidalTimeEmbedding,
    conditional_flow, target_velocity, heun_sample, sde_sample, ddim_sample
)


# ============================================================================
# Bidirectional MAMBA
# ============================================================================

class BidirectionalMAMBA(nn.Module):
    """
    Bidirectional MAMBA with split forward/backward layers

    Architecture:
    - Forward layers: Process sequence left → right
    - Backward layers: Process sequence right → left
    - Combination: Merge bidirectional features

    Benefits:
    - Every position gets context from BOTH directions
    - Better spatial propagation for coherent fields
    - Same total layers as unidirectional, just split
    """
    def __init__(self, d_model, num_layers=8, d_state=16, dropout=0.1):
        super().__init__()

        # Split layers: half forward, half backward
        self.num_forward = num_layers // 2  # 4 layers
        self.num_backward = num_layers // 2  # 4 layers

        print(f"  Bidirectional MAMBA: {self.num_forward} forward + {self.num_backward} backward layers")

        # Forward MAMBA blocks (process left → right)
        self.forward_blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, expand_factor=2, dropout=dropout)
            for _ in range(self.num_forward)
        ])

        # Backward MAMBA blocks (process right → left)
        self.backward_blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, expand_factor=2, dropout=dropout)
            for _ in range(self.num_backward)
        ])

        # Combine forward and backward features
        self.combine = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Process sequence bidirectionally

        Args:
            x: (B, N, d_model) input sequence

        Returns:
            x: (B, N, d_model) bidirectional features
        """
        # Forward pass (left → right)
        x_forward = x
        for block in self.forward_blocks:
            x_forward = block(x_forward)

        # Backward pass (right → left)
        x_backward = torch.flip(x, dims=[1])  # Reverse sequence
        for block in self.backward_blocks:
            x_backward = block(x_backward)
        x_backward = torch.flip(x_backward, dims=[1])  # Reverse back to original order

        # Combine bidirectional features
        x_bi = torch.cat([x_forward, x_backward], dim=-1)  # (B, N, 2*d_model)
        x_combined = self.combine(x_bi)  # (B, N, d_model)

        return x_combined


# ============================================================================
# Lightweight Perceiver
# ============================================================================

class LightweightPerceiver(nn.Module):
    """
    Lightweight perceiver with query self-attention for spatial coherence

    Architecture:
    - 2 iterations of refinement (coarse → fine)
    - Each iteration:
      1. Cross-attention: Query → Input (gather information)
      2. Self-attention: Query → Query (spatial smoothing!)
      3. MLP: Refinement

    Key insight: Query self-attention allows neighboring pixels to influence
    each other, creating smooth, coherent spatial fields instead of isolated
    speckled predictions.
    """
    def __init__(self, d_model, num_heads=8, num_iterations=2, dropout=0.1):
        super().__init__()

        print(f"  Lightweight Perceiver: {num_iterations} iterations, {num_heads} heads")

        self.iterations = nn.ModuleList([
            nn.ModuleDict({
                # Cross-attention: Query gathers from Input
                'cross_attn': nn.MultiheadAttention(
                    d_model, num_heads, dropout=dropout, batch_first=True
                ),
                'cross_norm': nn.LayerNorm(d_model),

                # Self-attention: Query communicates with Query (spatial smoothing!)
                'self_attn': nn.MultiheadAttention(
                    d_model, num_heads, dropout=dropout, batch_first=True
                ),
                'self_norm': nn.LayerNorm(d_model),

                # MLP refinement (2x expansion instead of 4x for efficiency)
                'mlp': nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 2, d_model),
                    nn.Dropout(dropout)
                )
            }) for _ in range(num_iterations)
        ])

    def forward(self, query_tokens, input_tokens):
        """
        Iterative refinement with spatial smoothing

        Args:
            query_tokens: (B, N_out, d_model) query features
            input_tokens: (B, N_in, d_model) input features

        Returns:
            query_tokens: (B, N_out, d_model) refined features
        """
        for iteration in self.iterations:
            # Cross-attention: gather information from inputs
            residual = query_tokens
            attn_out, _ = iteration['cross_attn'](
                query_tokens, input_tokens, input_tokens
            )
            query_tokens = iteration['cross_norm'](residual + attn_out)

            # Self-attention: queries communicate for spatial coherence
            residual = query_tokens
            attn_out, _ = iteration['self_attn'](
                query_tokens, query_tokens, query_tokens
            )
            query_tokens = iteration['self_norm'](residual + attn_out)

            # MLP refinement
            query_tokens = query_tokens + iteration['mlp'](query_tokens)

        return query_tokens


# ============================================================================
# MAMBA Diffusion V2
# ============================================================================

class MAMBADiffusionV2(nn.Module):
    """
    MAMBA Diffusion V2 with bidirectional processing and spatial coherence

    Architecture improvements:
    1. Bidirectional MAMBA: 4 forward + 4 backward = 8 total layers
    2. Lightweight Perceiver: 2 iterations with query self-attention
    3. Same interface as V1 for easy comparison

    Key changes from V1:
    - self.mamba_blocks → self.bidirectional_mamba
    - self.query_cross_attn → self.perceiver

    Expected improvements:
    - 70-80% reduction in background speckles
    - +3-5 dB PSNR improvement
    - Smoother spatial fields with better coherence
    """
    def __init__(
        self,
        num_fourier_feats=256,
        d_model=256,  # Changed default from 512 to 256
        num_layers=8,  # Changed default from 6 to 8 (4 forward + 4 backward)
        d_state=16,
        dropout=0.1,
        perceiver_iterations=2,
        perceiver_heads=8
    ):
        super().__init__()
        self.d_model = d_model

        print(f"\nInitializing MAMBA Diffusion V2:")
        print(f"  d_model: {d_model}")
        print(f"  num_layers: {num_layers} (total MAMBA layers)")
        print(f"  d_state: {d_state}")

        # Fourier features (unchanged from V1)
        self.fourier = FourierFeatures(coord_dim=2, num_freqs=num_fourier_feats, scale=10.0)
        feat_dim = num_fourier_feats * 2

        # Project inputs and queries (unchanged from V1)
        self.input_proj = nn.Linear(feat_dim + 3, d_model)
        self.query_proj = nn.Linear(feat_dim + 3, d_model)

        # Time embedding (unchanged from V1)
        self.time_embed = SinusoidalTimeEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # NEW: Bidirectional MAMBA (replaces unidirectional mamba_blocks)
        self.bidirectional_mamba = BidirectionalMAMBA(
            d_model, num_layers, d_state, dropout
        )

        # NEW: Lightweight Perceiver (replaces single cross-attention)
        self.perceiver = LightweightPerceiver(
            d_model, perceiver_heads, perceiver_iterations, dropout
        )

        # Output decoder (unchanged from V1)
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

        # NEW: Bidirectional MAMBA processing
        seq = self.bidirectional_mamba(seq)

        # Split back into input and query sequences
        input_seq = seq[:, :N_in, :]  # (B, N_in, d_model)
        query_seq = seq[:, N_in:, :]  # (B, N_out, d_model)

        # NEW: Lightweight Perceiver with query self-attention
        output = self.perceiver(query_seq, input_seq)

        # Decode to RGB
        return self.decoder(output)


# ============================================================================
# Training Loop (Same as V1 with model name changes)
# ============================================================================

def train_flow_matching(
    model, train_loader, test_loader, epochs=1000, lr=1e-4, device='cuda',
    visualize_every=50, eval_every=10, save_every=10, save_dir='checkpoints_mamba_v2'
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
        f.write(f"Training V2 started at {datetime.now()}\n")
        f.write(f"Epochs: {epochs}, LR: {lr}, Save every: {save_every}\n")
        f.write("="*60 + "\n")

    print(f"\n{'='*60}")
    print(f"Training MAMBA Diffusion V2 for {epochs} epochs")
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
        print(log_msg, flush=True)

        with open(log_file, 'a') as f:
            f.write(log_msg + "\n")
            f.flush()

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
                    }, os.path.join(save_dir, 'mamba_v2_best.pth'))
                    best_msg = f"  ✓ Saved best model (val_loss: {val_loss:.6f})"
                    print(best_msg, flush=True)
                    with open(log_file, 'a') as f:
                        f.write(best_msg + "\n")
                        f.flush()

        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(save_dir, f'mamba_v2_epoch_{epoch+1:04d}.pth')
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

        # Save latest model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'val_loss': val_loss if val_loss is not None else avg_loss,
            'best_val_loss': best_val_loss
        }, os.path.join(save_dir, 'mamba_v2_latest.pth'))

        # Visualization (same as V1)
        if (epoch + 1) % visualize_every == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                pred_values = heun_sample(
                    model, viz_output_coords, viz_input_coords, viz_input_values,
                    num_steps=50, device=device
                )

                full_coords_batch = full_coords.unsqueeze(0).expand(4, -1, -1)
                full_pred_values = heun_sample(
                    model, full_coords_batch, viz_input_coords, viz_input_values,
                    num_steps=50, device=device
                )
                full_pred_images = full_pred_values.view(4, 32, 32, 3).permute(0, 3, 1, 2)

                fig, axes = plt.subplots(4, 5, figsize=(20, 16))

                for i in range(4):
                    axes[i, 0].imshow(viz_full_images[i].permute(1, 2, 0).cpu().numpy())
                    axes[i, 0].set_title('Ground Truth' if i == 0 else '', fontsize=10)
                    axes[i, 0].axis('off')

                    input_img = torch.zeros(3, 32, 32, device=device)
                    input_idx = viz_input_indices[i]
                    input_img.view(3, -1)[:, input_idx] = viz_input_values[i].T
                    axes[i, 1].imshow(input_img.permute(1, 2, 0).cpu().numpy())
                    axes[i, 1].set_title('Sparse Input (20%)' if i == 0 else '', fontsize=10)
                    axes[i, 1].axis('off')

                    target_img = torch.zeros(3, 32, 32, device=device)
                    output_idx = viz_batch['output_indices'][i]
                    target_img.view(3, -1)[:, output_idx] = viz_output_values[i].T
                    axes[i, 2].imshow(target_img.permute(1, 2, 0).cpu().numpy())
                    axes[i, 2].set_title('Sparse Target (20%)' if i == 0 else '', fontsize=10)
                    axes[i, 2].axis('off')

                    pred_img = torch.zeros(3, 32, 32, device=device)
                    pred_img.view(3, -1)[:, output_idx] = pred_values[i].T
                    axes[i, 3].imshow(np.clip(pred_img.permute(1, 2, 0).cpu().numpy(), 0, 1))
                    axes[i, 3].set_title('Sparse Prediction' if i == 0 else '', fontsize=10)
                    axes[i, 3].axis('off')

                    axes[i, 4].imshow(np.clip(full_pred_images[i].permute(1, 2, 0).cpu().numpy(), 0, 1))
                    axes[i, 4].set_title('Full Field Recon' if i == 0 else '', fontsize=10)
                    axes[i, 4].axis('off')

                plt.suptitle(f'MAMBA V2 - Epoch {epoch+1} (Best Val: {best_val_loss:.6f})',
                           fontsize=14, y=0.995)
                plt.tight_layout()
                viz_path = os.path.join(save_dir, f'mamba_v2_epoch_{epoch+1:04d}.png')
                plt.savefig(viz_path, dpi=150, bbox_inches='tight')
                plt.close()

    final_msg = f"\n✓ Training complete! Best validation loss: {best_val_loss:.6f}"
    print(final_msg)
    print(f"  Best model: {save_dir}/mamba_v2_best.pth")
    print(f"  Latest model: {save_dir}/mamba_v2_latest.pth")

    with open(log_file, 'a') as f:
        f.write("="*60 + "\n")
        f.write(final_msg + "\n")
        f.write(f"Training completed at {datetime.now()}\n")

    return losses


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train MAMBA Diffusion V2')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--eval_every', type=int, default=10, help='Evaluate every N epochs (default: 10)')
    parser.add_argument('--visualize_every', type=int, default=50, help='Visualize every N epochs (default: 50)')
    parser.add_argument('--save_dir', type=str, default='checkpoints_mamba_v2', help='Save directory')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension (default: 256)')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of MAMBA layers (default: 4)')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers (default: 4)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Device to use: auto (default), cuda, or cpu')
    args = parser.parse_args()

    # Device setup
    print("=" * 60)
    print("DEVICE SETUP")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Requested device: {args.device}")

    if args.device == 'cpu':
        device = torch.device('cpu')
        print(f"\n✓ Using CPU (forced by --device cpu)")
    elif args.device == 'cuda':
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            device = torch.device('cuda:0')
            torch.cuda.set_device(0)
            torch.backends.cudnn.benchmark = True
            print(f"\n✓ Using GPU: {torch.cuda.get_device_name(0)}")
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
            device = torch.device('cuda:0')
            torch.cuda.set_device(0)
            torch.backends.cudnn.benchmark = True
            print(f"\n✓ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print(f"\n⚠️  CUDA not available - falling back to CPU")
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

    # Initialize model V2
    print("Initializing MAMBA V2 model...")
    model = MAMBADiffusionV2(
        num_fourier_feats=256,
        d_model=args.d_model,
        num_layers=args.num_layers,
        d_state=16,
        dropout=0.1,
        perceiver_iterations=2,
        perceiver_heads=8
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
    plt.title('Training Loss: MAMBA Diffusion V2')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(args.save_dir, 'training_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
