"""
Multi-Scale Evaluation Script for Local Implicit and MAMBA Models

This script evaluates trained models at multiple resolutions to test scale invariance.

Usage:
    python evaluate_multiscale.py --model local_implicit --checkpoint model.pth
    python evaluate_multiscale.py --model mamba --checkpoint model.pth
"""

import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os

from core.sparse.cifar10_sparse import SparseCIFAR10Dataset
from core.sparse.metrics import MetricsTracker

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#=============================================================================
# Multi-Scale Utilities
#=============================================================================

def create_multi_scale_grids(device='cuda'):
    """Create coordinate grids at different resolutions"""
    grids = {}

    for size in [32, 64, 96, 128]:
        y, x = torch.meshgrid(
            torch.linspace(0, 1, size),
            torch.linspace(0, 1, size),
            indexing='ij'
        )
        grids[size] = torch.stack([x.flatten(), y.flatten()], dim=-1).to(device)

    print(f"Created grids for resolutions: {list(grids.keys())}")
    return grids


@torch.no_grad()
def heun_sample(model, output_coords, input_coords, input_values, num_steps=100, device='cuda'):
    """Heun ODE solver for sampling"""
    B, N_out = output_coords.shape[0], output_coords.shape[1]
    x_t = torch.randn(B, N_out, 3, device=device)

    dt = 1.0 / num_steps
    ts = torch.linspace(0, 1 - dt, num_steps)

    for t_val in tqdm(ts, desc="Sampling", leave=False):
        t = torch.full((B,), t_val.item(), device=device)
        t_next = torch.full((B,), t_val.item() + dt, device=device)

        v1 = model(x_t, output_coords, t, input_coords, input_values)
        x_next_pred = x_t + dt * v1

        v2 = model(x_next_pred, output_coords, t_next, input_coords, input_values)
        x_t = x_t + dt * 0.5 * (v1 + v2)

    return torch.clamp(x_t, 0, 1)


@torch.no_grad()
def multi_scale_reconstruction(model, input_coords, input_values, grids, num_steps=100, device='cuda'):
    """
    Reconstruct at multiple scales

    Args:
        model: Trained diffusion model
        input_coords: (B, N_in, 2) sparse input coordinates
        input_values: (B, N_in, 3) sparse RGB values
        grids: Dict of {size: coordinates}
        num_steps: ODE solver steps

    Returns:
        Dict of {size: reconstructed_images}
    """
    model.eval()
    B = input_coords.shape[0]

    reconstructions = {}

    for size, coords in grids.items():
        print(f"\nReconstructing at {size}x{size} ({size**2} pixels)...")

        # Expand coords for batch
        coords_batch = coords.unsqueeze(0).expand(B, -1, -1)

        # Sample
        pred_values = heun_sample(
            model, coords_batch, input_coords, input_values,
            num_steps=num_steps, device=device
        )

        # Reshape to image
        pred_images = pred_values.view(B, size, size, 3).permute(0, 3, 1, 2)
        reconstructions[size] = pred_images

        print(f"  Completed: {pred_images.shape}")

    return reconstructions


#=============================================================================
# Visualization Functions
#=============================================================================

def visualize_multi_scale(ground_truth, sparse_input_img, multi_scale_results, sample_idx=0, save_path=None):
    """Visualize multi-scale reconstructions"""

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 1: Inputs and native resolution
    axes[0, 0].imshow(ground_truth.permute(1, 2, 0).cpu().numpy())
    axes[0, 0].set_title('Ground Truth\n(32x32)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(sparse_input_img.permute(1, 2, 0).cpu().numpy())
    axes[0, 1].set_title('Sparse Input\n(20% = 204 pixels)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    img_32 = multi_scale_results[32][sample_idx].permute(1, 2, 0).cpu().numpy()
    axes[0, 2].imshow(np.clip(img_32, 0, 1))
    axes[0, 2].set_title('Reconstructed\n32x32 (Native)', fontsize=12, fontweight='bold', color='green')
    axes[0, 2].axis('off')

    # Bilinear upsampling for comparison
    img_32_bi_64 = F.interpolate(
        multi_scale_results[32][sample_idx:sample_idx+1],
        size=64, mode='bilinear', align_corners=False
    )[0].permute(1, 2, 0).cpu().numpy()
    axes[0, 3].imshow(np.clip(img_32_bi_64, 0, 1))
    axes[0, 3].set_title('32→64 Bilinear\n(Traditional)', fontsize=12, fontweight='bold')
    axes[0, 3].axis('off')

    # Row 2: Upsampled versions
    if 64 in multi_scale_results:
        img_64 = multi_scale_results[64][sample_idx].permute(1, 2, 0).cpu().numpy()
        axes[1, 0].imshow(np.clip(img_64, 0, 1))
        axes[1, 0].set_title('Continuous Field\n64x64 (2x)', fontsize=12, fontweight='bold', color='green')
        axes[1, 0].axis('off')

    if 96 in multi_scale_results:
        img_96 = multi_scale_results[96][sample_idx].permute(1, 2, 0).cpu().numpy()
        axes[1, 1].imshow(np.clip(img_96, 0, 1))
        axes[1, 1].set_title('Continuous Field\n96x96 (3x)', fontsize=12, fontweight='bold', color='green')
        axes[1, 1].axis('off')

    if 128 in multi_scale_results:
        img_128 = multi_scale_results[128][sample_idx].permute(1, 2, 0).cpu().numpy()
        axes[1, 2].imshow(np.clip(img_128, 0, 1))
        axes[1, 2].set_title('Continuous Field\n128x128 (4x)', fontsize=12, fontweight='bold', color='green')
        axes[1, 2].axis('off')

    # Bilinear 96 for comparison
    img_32_bi_96 = F.interpolate(
        multi_scale_results[32][sample_idx:sample_idx+1],
        size=96, mode='bilinear', align_corners=False
    )[0].permute(1, 2, 0).cpu().numpy()
    axes[1, 3].imshow(np.clip(img_32_bi_96, 0, 1))
    axes[1, 3].set_title('32→96 Bilinear\n(Traditional)', fontsize=12, fontweight='bold')
    axes[1, 3].axis('off')

    plt.suptitle('Scale-Invariant Continuous Field Reconstruction',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def create_comparison_grid(multi_scale_results, sample_idx=0, save_path=None):
    """Create detailed comparison showing zoom-ins"""

    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.2)

    # Full images
    ax1 = fig.add_subplot(gs[0, 0])
    img_32 = multi_scale_results[32][sample_idx].permute(1, 2, 0).cpu().numpy()
    ax1.imshow(np.clip(img_32, 0, 1))
    ax1.set_title('32x32 Full', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    if 64 in multi_scale_results:
        img_64 = multi_scale_results[64][sample_idx].permute(1, 2, 0).cpu().numpy()
        ax2.imshow(np.clip(img_64, 0, 1))
        ax2.set_title('64x64 Full', fontsize=14, fontweight='bold')
        ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    if 96 in multi_scale_results:
        img_96 = multi_scale_results[96][sample_idx].permute(1, 2, 0).cpu().numpy()
        ax3.imshow(np.clip(img_96, 0, 1))
        ax3.set_title('96x96 Full', fontsize=14, fontweight='bold')
        ax3.axis('off')

    ax4 = fig.add_subplot(gs[0, 3])
    if 128 in multi_scale_results:
        img_128 = multi_scale_results[128][sample_idx].permute(1, 2, 0).cpu().numpy()
        ax4.imshow(np.clip(img_128, 0, 1))
        ax4.set_title('128x128 Full', fontsize=14, fontweight='bold')
        ax4.axis('off')

    # Zoom-in region (center crop)
    zoom_sizes = {32: (8, 24), 64: (16, 48), 96: (24, 72), 128: (32, 96)}

    for col, size in enumerate([32, 64, 96, 128]):
        if size not in multi_scale_results:
            continue

        y1, y2 = zoom_sizes[size]
        x1, x2 = zoom_sizes[size]

        ax_zoom = fig.add_subplot(gs[1, col])
        img = multi_scale_results[size][sample_idx].permute(1, 2, 0).cpu().numpy()
        ax_zoom.imshow(np.clip(img[y1:y2, x1:x2], 0, 1))
        ax_zoom.set_title(f'{size}x{size} Zoom', fontsize=12)
        ax_zoom.axis('off')

        # Show upsampled version from 32x32
        ax_up = fig.add_subplot(gs[2, col])
        img_32_up = F.interpolate(
            multi_scale_results[32][sample_idx:sample_idx+1],
            size=size, mode='bilinear', align_corners=False
        )[0].permute(1, 2, 0).cpu().numpy()
        ax_up.imshow(np.clip(img_32_up[y1:y2, x1:x2], 0, 1))
        ax_up.set_title(f'32→{size} Bilinear', fontsize=12, color='red')
        ax_up.axis('off')

    plt.suptitle('Continuous Field vs Traditional Upsampling (Zoom Comparison)',
                 fontsize=18, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


#=============================================================================
# Evaluation Metrics
#=============================================================================

@torch.no_grad()
def evaluate_full_field(model, test_loader, grid_size=32, num_batches=100, num_steps=100, device='cuda'):
    """Evaluate full field reconstruction at native resolution"""

    y, x = torch.meshgrid(
        torch.linspace(0, 1, grid_size),
        torch.linspace(0, 1, grid_size),
        indexing='ij'
    )
    full_grid = torch.stack([x.flatten(), y.flatten()], dim=-1).to(device)

    model.eval()
    tracker = MetricsTracker()

    print(f"\nEvaluating full field reconstruction at {grid_size}x{grid_size}...")

    for i, batch in enumerate(tqdm(test_loader, desc="Full field eval")):
        if i >= num_batches:
            break

        B = batch['input_coords'].shape[0]
        grid_batch = full_grid.unsqueeze(0).expand(B, -1, -1)

        pred_values = heun_sample(
            model, grid_batch,
            batch['input_coords'].to(device),
            batch['input_values'].to(device),
            num_steps=num_steps, device=device
        )

        pred_images = pred_values.view(B, grid_size, grid_size, 3).permute(0, 3, 1, 2)
        tracker.update(None, None, pred_images, batch['full_image'].to(device))

    results = tracker.compute()
    results_std = tracker.compute_std()

    return results, results_std


def print_evaluation_results(results, results_std, title="Evaluation Results"):
    """Pretty print evaluation metrics"""
    print("\n" + "="*70)
    print(f"{title:^70}")
    print("="*70)
    print(f"  PSNR: {results['psnr']:.2f} ± {results_std['psnr_std']:.2f} dB")
    print(f"  SSIM: {results['ssim']:.4f} ± {results_std['ssim_std']:.4f}")
    print(f"  MSE:  {results['mse']:.6f} ± {results_std['mse_std']:.6f}")
    print(f"  MAE:  {results['mae']:.6f} ± {results_std['mae_std']:.6f}")
    print("="*70)


#=============================================================================
# Main Evaluation Script
#=============================================================================

def main():
    parser = argparse.ArgumentParser(description='Multi-scale evaluation')
    parser.add_argument('--model', type=str, required=True, choices=['local_implicit', 'mamba'],
                       help='Model type')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--num_steps', type=int, default=100,
                       help='ODE solver steps')
    parser.add_argument('--output_dir', type=str, default='./multiscale_results',
                       help='Output directory for visualizations')
    parser.add_argument('--scales', type=int, nargs='+', default=[32, 64, 96],
                       help='Resolutions to evaluate (e.g., 32 64 96 128)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading CIFAR-10 test dataset...")
    test_dataset = SparseCIFAR10Dataset(
        root='../data', train=False, input_ratio=0.2, output_ratio=0.2, download=True, seed=42
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load model
    print(f"Loading {args.model} model from {args.checkpoint}...")
    model = torch.load(args.checkpoint, map_location=device)
    model.eval()

    # Create multi-scale grids
    grids = {size: create_multi_scale_grids(device)[size] for size in args.scales}

    # Full field evaluation at native resolution
    print("\n" + "="*70)
    print("PHASE 1: Full Field Reconstruction at Native Resolution (32x32)")
    print("="*70)

    results_32, results_std_32 = evaluate_full_field(
        model, test_loader, grid_size=32, num_batches=100,
        num_steps=args.num_steps, device=device
    )
    print_evaluation_results(results_32, results_std_32,
                            title="Full Field Reconstruction (32x32)")

    # Multi-scale reconstruction
    print("\n" + "="*70)
    print("PHASE 2: Multi-Scale Continuous Field Reconstruction")
    print("="*70)

    test_batch = next(iter(test_loader))
    B_vis = min(4, test_batch['input_coords'].shape[0])

    multi_scale_results = multi_scale_reconstruction(
        model,
        test_batch['input_coords'][:B_vis].to(device),
        test_batch['input_values'][:B_vis].to(device),
        grids,
        num_steps=args.num_steps,
        device=device
    )

    # Visualizations
    print("\n" + "="*70)
    print("PHASE 3: Generating Visualizations")
    print("="*70)

    for i in range(B_vis):
        # Create sparse input visualization
        sparse_img = torch.zeros(3, 32, 32)
        input_idx = test_batch['input_indices'][i]
        sparse_img.view(3, -1)[:, input_idx] = test_batch['input_values'][i].T

        # Multi-scale visualization
        save_path = os.path.join(args.output_dir, f'{args.model}_multiscale_sample_{i}.png')
        visualize_multi_scale(
            test_batch['full_image'][i],
            sparse_img,
            multi_scale_results,
            sample_idx=i,
            save_path=save_path
        )
        plt.close()

        # Detailed comparison
        save_path = os.path.join(args.output_dir, f'{args.model}_comparison_sample_{i}.png')
        create_comparison_grid(
            multi_scale_results,
            sample_idx=i,
            save_path=save_path
        )
        plt.close()

    # Final summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"Model: {args.model}")
    print(f"Scales evaluated: {args.scales}")
    print(f"\nKey findings:")
    print(f"  - Full field PSNR: {results_32['psnr']:.2f} dB")
    print(f"  - Full field SSIM: {results_32['ssim']:.4f}")
    print(f"  - Successfully reconstructed at: {list(grids.keys())}")
    print("\nCheck the visualization images to assess scale invariance!")


if __name__ == '__main__':
    main()
