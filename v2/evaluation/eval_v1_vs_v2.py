"""
Architecture Comparison: MAMBA V1 vs V2

Compares the original MAMBA (V1) with the improved architecture (V2):
- V1: Unidirectional MAMBA + single cross-attention
- V2: Bidirectional MAMBA + lightweight perceiver with query self-attention

Tests both architectures on the same data and computes metrics to quantify
the improvement in speckle reduction and spatial coherence.
"""
import sys
import os
import argparse
from datetime import datetime

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from train_mamba_standalone import MAMBADiffusion, heun_sample
from train_mamba_v2 import MAMBADiffusionV2
from core.sparse.cifar10_sparse import SparseCIFAR10Dataset


# ============================================================================
# Helper Functions
# ============================================================================

def create_coordinate_grid(resolution, device):
    """Create normalized coordinate grid"""
    y, x = torch.meshgrid(
        torch.linspace(0, 1, resolution),
        torch.linspace(0, 1, resolution),
        indexing='ij'
    )
    coords = torch.stack([x.flatten(), y.flatten()], dim=-1).to(device)
    return coords


def compute_metrics(pred, target):
    """Compute image quality metrics"""
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)

    mse = np.mean((pred - target) ** 2)
    mae = np.mean(np.abs(pred - target))
    psnr_val = psnr(target, pred, data_range=1.0)
    ssim_val = ssim(target, pred, data_range=1.0, channel_axis=2)

    return {
        'mse': mse,
        'mae': mae,
        'psnr': psnr_val,
        'ssim': ssim_val
    }


# ============================================================================
# V1 vs V2 Comparison
# ============================================================================

@torch.no_grad()
def compare_v1_v2(
    model_v1,
    model_v2,
    test_loader,
    num_samples=20,
    num_steps=50,
    device='cuda',
    save_dir='eval_v1_vs_v2'
):
    """
    Compare V1 and V2 architectures

    Args:
        model_v1: Original MAMBA diffusion
        model_v2: Improved MAMBA diffusion V2
        test_loader: Test data loader
        num_samples: Number of samples to evaluate
        num_steps: Sampling steps
        device: Device
        save_dir: Save directory

    Returns:
        results: Dict with metrics for both models
    """
    os.makedirs(save_dir, exist_ok=True)
    model_v1.eval()
    model_v2.eval()

    # Store results
    results = {
        'v1': {'mse': [], 'mae': [], 'psnr': [], 'ssim': []},
        'v2': {'mse': [], 'mae': [], 'psnr': [], 'ssim': []}
    }

    # Store images for visualization
    sample_images = {
        'input_coords': [],
        'input_values': [],
        'ground_truth': [],
        'v1_predictions': [],
        'v2_predictions': []
    }

    print(f"\n{'='*70}")
    print("ARCHITECTURE COMPARISON: V1 vs V2")
    print(f"{'='*70}")
    print(f"Evaluating {num_samples} samples at 32x32 resolution")
    print(f"Using {num_steps} sampling steps")
    print(f"{'='*70}\n")

    # Evaluation loop
    sample_count = 0
    for batch in test_loader:
        if sample_count >= num_samples:
            break

        batch_size = batch['input_coords'].shape[0]
        for i in range(batch_size):
            if sample_count >= num_samples:
                break

            # Get sparse input
            input_coords = batch['input_coords'][i:i+1].to(device)
            input_values = batch['input_values'][i:i+1].to(device)
            output_coords = batch['output_coords'][i:i+1].to(device)
            output_values = batch['output_values'][i:i+1].to(device)
            ground_truth = batch['full_image'][i].to(device)

            # Store for visualization
            if sample_count < 4:
                sample_images['input_coords'].append(input_coords.cpu())
                sample_images['input_values'].append(input_values.cpu())
                sample_images['ground_truth'].append(ground_truth.cpu())

            print(f"Sample {sample_count + 1}/{num_samples}:")

            # V1 prediction
            v1_pred = heun_sample(
                model_v1, output_coords, input_coords, input_values,
                num_steps=num_steps, device=device
            )

            # V2 prediction
            v2_pred = heun_sample(
                model_v2, output_coords, input_coords, input_values,
                num_steps=num_steps, device=device
            )

            # Reshape to images
            v1_img = torch.zeros(3, 32, 32, device=device)
            v2_img = torch.zeros(3, 32, 32, device=device)
            output_idx = batch['output_indices'][i]

            v1_img.view(3, -1)[:, output_idx] = v1_pred[0].T
            v2_img.view(3, -1)[:, output_idx] = v2_pred[0].T

            # Convert to numpy for metrics
            v1_np = v1_img.cpu().permute(1, 2, 0).numpy().clip(0, 1)
            v2_np = v2_img.cpu().permute(1, 2, 0).numpy().clip(0, 1)
            gt_np = ground_truth.cpu().permute(1, 2, 0).numpy().clip(0, 1)

            # Compute metrics
            v1_metrics = compute_metrics(v1_np, gt_np)
            v2_metrics = compute_metrics(v2_np, gt_np)

            # Store metrics
            for key in ['mse', 'mae', 'psnr', 'ssim']:
                results['v1'][key].append(v1_metrics[key])
                results['v2'][key].append(v2_metrics[key])

            # Store images for visualization
            if sample_count < 4:
                sample_images['v1_predictions'].append(v1_img.cpu())
                sample_images['v2_predictions'].append(v2_img.cpu())

            # Print comparison
            improvement_psnr = v2_metrics['psnr'] - v1_metrics['psnr']
            improvement_ssim = v2_metrics['ssim'] - v1_metrics['ssim']

            print(f"  V1: PSNR={v1_metrics['psnr']:.2f}dB, SSIM={v1_metrics['ssim']:.4f}")
            print(f"  V2: PSNR={v2_metrics['psnr']:.2f}dB, SSIM={v2_metrics['ssim']:.4f}")
            print(f"  Δ:  PSNR={improvement_psnr:+.2f}dB, SSIM={improvement_ssim:+.4f}")
            print()

            sample_count += 1

    # Compute average metrics and improvements
    print(f"\n{'='*70}")
    print("AVERAGE METRICS AND IMPROVEMENTS")
    print(f"{'='*70}\n")

    summary = {'v1': {}, 'v2': {}, 'improvement': {}}

    for key in ['mse', 'mae', 'psnr', 'ssim']:
        v1_mean = np.mean(results['v1'][key])
        v1_std = np.std(results['v1'][key])
        v2_mean = np.mean(results['v2'][key])
        v2_std = np.std(results['v2'][key])

        summary['v1'][key] = {'mean': v1_mean, 'std': v1_std}
        summary['v2'][key] = {'mean': v2_mean, 'std': v2_std}

        if key in ['psnr', 'ssim']:
            improvement = v2_mean - v1_mean
            improvement_pct = (improvement / v1_mean) * 100 if v1_mean != 0 else 0
        else:  # mse, mae (lower is better)
            improvement = v1_mean - v2_mean
            improvement_pct = (improvement / v1_mean) * 100 if v1_mean != 0 else 0

        summary['improvement'][key] = {
            'absolute': improvement,
            'percent': improvement_pct
        }

        if key == 'psnr':
            print(f"{key.upper()}:")
            print(f"  V1: {v1_mean:.2f} ± {v1_std:.2f} dB")
            print(f"  V2: {v2_mean:.2f} ± {v2_std:.2f} dB")
            print(f"  Improvement: {improvement:+.2f} dB ({improvement_pct:+.1f}%)")
        elif key in ['mse', 'mae']:
            print(f"{key.upper()}:")
            print(f"  V1: {v1_mean:.6f} ± {v1_std:.6f}")
            print(f"  V2: {v2_mean:.6f} ± {v2_std:.6f}")
            print(f"  Reduction: {improvement:.6f} ({improvement_pct:+.1f}%)")
        else:
            print(f"{key.upper()}:")
            print(f"  V1: {v1_mean:.4f} ± {v1_std:.4f}")
            print(f"  V2: {v2_mean:.4f} ± {v2_std:.4f}")
            print(f"  Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")
        print()

    print(f"{'='*70}\n")

    # Save results
    save_results(results, summary, save_dir)

    # Create visualizations
    visualize_comparison(sample_images, save_dir)
    visualize_improvement_chart(summary, save_dir)

    return results, summary


def save_results(results, summary, save_dir):
    """Save detailed results"""
    results_file = os.path.join(save_dir, 'comparison_results.txt')

    with open(results_file, 'w') as f:
        f.write(f"V1 vs V2 Comparison Results\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("="*70 + "\n\n")

        f.write("AVERAGE METRICS\n")
        f.write("-"*70 + "\n\n")

        for key in ['mse', 'mae', 'psnr', 'ssim']:
            v1_mean = summary['v1'][key]['mean']
            v1_std = summary['v1'][key]['std']
            v2_mean = summary['v2'][key]['mean']
            v2_std = summary['v2'][key]['std']
            improvement = summary['improvement'][key]['absolute']
            improvement_pct = summary['improvement'][key]['percent']

            f.write(f"{key.upper()}:\n")
            if key == 'psnr':
                f.write(f"  V1: {v1_mean:.2f} ± {v1_std:.2f} dB\n")
                f.write(f"  V2: {v2_mean:.2f} ± {v2_std:.2f} dB\n")
                f.write(f"  Improvement: {improvement:+.2f} dB ({improvement_pct:+.1f}%)\n")
            elif key in ['mse', 'mae']:
                f.write(f"  V1: {v1_mean:.6f} ± {v1_std:.6f}\n")
                f.write(f"  V2: {v2_mean:.6f} ± {v2_std:.6f}\n")
                f.write(f"  Reduction: {improvement:.6f} ({improvement_pct:+.1f}%)\n")
            else:
                f.write(f"  V1: {v1_mean:.4f} ± {v1_std:.4f}\n")
                f.write(f"  V2: {v2_mean:.4f} ± {v2_std:.4f}\n")
                f.write(f"  Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)\n")
            f.write("\n")

    print(f"✓ Results saved to {results_file}")


def visualize_comparison(sample_images, save_dir):
    """Create visual comparison"""
    num_samples = len(sample_images['ground_truth'])
    if num_samples == 0:
        return

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Column 0: Sparse input
        input_img = torch.zeros(3, 32, 32)
        input_idx = (sample_images['input_coords'][i][0] * 31).long()
        input_idx_flat = input_idx[:, 1] * 32 + input_idx[:, 0]
        input_img.view(3, -1)[:, input_idx_flat] = sample_images['input_values'][i][0].T

        axes[i, 0].imshow(input_img.permute(1, 2, 0).numpy())
        axes[i, 0].set_title('Input (20%)' if i == 0 else '', fontsize=12)
        axes[i, 0].axis('off')

        # Column 1: Ground truth
        gt = sample_images['ground_truth'][i].permute(1, 2, 0).numpy()
        axes[i, 1].imshow(gt)
        axes[i, 1].set_title('Ground Truth' if i == 0 else '', fontsize=12)
        axes[i, 1].axis('off')

        # Column 2: V1 prediction
        v1_pred = sample_images['v1_predictions'][i].permute(1, 2, 0).numpy().clip(0, 1)
        axes[i, 2].imshow(v1_pred)
        axes[i, 2].set_title('V1 (Original)' if i == 0 else '', fontsize=12)
        axes[i, 2].axis('off')

        # Column 3: V2 prediction
        v2_pred = sample_images['v2_predictions'][i].permute(1, 2, 0).numpy().clip(0, 1)
        axes[i, 3].imshow(v2_pred)
        axes[i, 3].set_title('V2 (Improved)' if i == 0 else '', fontsize=12)
        axes[i, 3].axis('off')

    plt.suptitle('Architecture Comparison: V1 vs V2', fontsize=14, y=0.998)
    plt.tight_layout()

    viz_path = os.path.join(save_dir, 'v1_vs_v2_comparison.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualization saved to {viz_path}")


def visualize_improvement_chart(summary, save_dir):
    """Create improvement chart"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metrics = ['psnr', 'ssim']
    labels = ['PSNR (dB)', 'SSIM']
    colors = ['#2E86AB', '#A23B72']

    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[idx]

        v1_mean = summary['v1'][metric]['mean']
        v2_mean = summary['v2'][metric]['mean']
        improvement = summary['improvement'][metric]['absolute']
        improvement_pct = summary['improvement'][metric]['percent']

        x = [0, 1]
        values = [v1_mean, v2_mean]

        bars = ax.bar(x, values, width=0.5, color=[colors[0], colors[1]], alpha=0.8)

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Add improvement annotation
        ax.annotate('', xy=(1, v2_mean), xytext=(1, v1_mean),
                   arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        ax.text(1.15, (v1_mean + v2_mean)/2,
               f'+{improvement:.3f}\n({improvement_pct:+.1f}%)',
               fontsize=10, color='green', fontweight='bold', va='center')

        ax.set_xticks(x)
        ax.set_xticklabels(['V1\n(Original)', 'V2\n(Improved)'], fontsize=11)
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.set_title(f'{label} Improvement', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('V2 Architecture Improvements', fontsize=14, fontweight='bold')
    plt.tight_layout()

    chart_path = os.path.join(save_dir, 'improvement_chart.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Improvement chart saved to {chart_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compare V1 vs V2 Architectures')
    parser.add_argument('--v1_checkpoint', type=str, required=True,
                        help='Path to V1 checkpoint')
    parser.add_argument('--v2_checkpoint', type=str, required=True,
                        help='Path to V2 checkpoint')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of test samples (default: 20)')
    parser.add_argument('--num_steps', type=int, default=50,
                        help='Number of sampling steps (default: 50)')
    parser.add_argument('--save_dir', type=str, default='eval_v1_vs_v2',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Device: auto/cuda/cpu')
    parser.add_argument('--d_model', type=int, default=256,
                        help='Model dimension (default: 256)')
    parser.add_argument('--num_layers_v1', type=int, default=6,
                        help='V1 number of layers (default: 6)')
    parser.add_argument('--num_layers_v2', type=int, default=4,
                        help='V2 number of layers (default: 4)')

    args = parser.parse_args()

    # Device setup
    print("=" * 70)
    print("DEVICE SETUP")
    print("=" * 70)

    if args.device == 'cpu':
        device = torch.device('cpu')
        print("✓ Using CPU")
    elif args.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("❌ CUDA not available, falling back to CPU")
            device = torch.device('cpu')
    else:  # auto
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("⚠️  CUDA not available, using CPU")

    print("=" * 70 + "\n")

    # Load test dataset
    print("Loading CIFAR-10 test dataset...")
    test_dataset = SparseCIFAR10Dataset(
        root='../data',
        train=False,
        input_ratio=0.2,
        output_ratio=0.2,
        download=True,
        seed=42
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Test dataset: {len(test_dataset)} samples\n")

    # Initialize V1 model
    print("Initializing V1 model...")
    model_v1 = MAMBADiffusion(
        num_fourier_feats=256,
        d_model=args.d_model,
        num_layers=args.num_layers_v1,
        d_state=16,
        dropout=0.1
    ).to(device)

    # Initialize V2 model
    print("Initializing V2 model...")
    model_v2 = MAMBADiffusionV2(
        num_fourier_feats=256,
        d_model=args.d_model,
        num_layers=args.num_layers_v2,
        d_state=16,
        dropout=0.1,
        perceiver_iterations=2,
        perceiver_heads=8
    ).to(device)

    # Load checkpoints
    print(f"\nLoading V1 checkpoint: {args.v1_checkpoint}")
    v1_checkpoint = torch.load(args.v1_checkpoint, map_location=device)
    model_v1.load_state_dict(v1_checkpoint['model_state_dict'])

    print(f"Loading V2 checkpoint: {args.v2_checkpoint}")
    v2_checkpoint = torch.load(args.v2_checkpoint, map_location=device)
    model_v2.load_state_dict(v2_checkpoint['model_state_dict'])

    print("\n✓ Models loaded successfully\n")

    # Run comparison
    results, summary = compare_v1_v2(
        model_v1=model_v1,
        model_v2=model_v2,
        test_loader=test_loader,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        device=device,
        save_dir=args.save_dir
    )

    print("✓ Comparison complete!")
    print(f"  Results saved to: {args.save_dir}/")


if __name__ == '__main__':
    main()
