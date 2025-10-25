"""
SDE Multi-Scale Super-Resolution Evaluation

Evaluates MAMBA diffusion with SDE sampling for smoother, less speckled outputs.
Tests at multiple resolutions: 32x32 (reconstruction), 64x64, 96x96

Key improvements over ODE sampling:
- Stochastic corrections reduce speckled backgrounds
- Temperature control balances quality vs smoothness
- Comparison between ODE, SDE, and DDIM samplers
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

from train_mamba_standalone import MAMBADiffusion, heun_sample, sde_sample, ddim_sample
from core.sparse.cifar10_sparse import SparseCIFAR10Dataset


# ============================================================================
# Helper Functions
# ============================================================================

def create_coordinate_grid(resolution, device):
    """Create normalized coordinate grid for given resolution"""
    y, x = torch.meshgrid(
        torch.linspace(0, 1, resolution),
        torch.linspace(0, 1, resolution),
        indexing='ij'
    )
    coords = torch.stack([x.flatten(), y.flatten()], dim=-1).to(device)
    return coords


def bicubic_upscale(image, target_size):
    """Bicubic upscaling baseline"""
    return F.interpolate(
        image.unsqueeze(0),
        size=(target_size, target_size),
        mode='bicubic',
        align_corners=False
    ).squeeze(0)


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
# SDE Multi-Scale Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_sde_multiscale(
    model,
    test_loader,
    resolutions=[32, 64, 96],
    samplers=['heun', 'sde', 'ddim'],
    num_samples=10,
    num_steps=50,
    temperature=0.5,
    eta=0.3,
    device='cuda',
    save_dir='eval_sde_multiscale'
):
    """
    Evaluate multiple sampling strategies across resolutions

    Args:
        model: Trained MAMBA model
        test_loader: Test data loader
        resolutions: List of target resolutions [32, 64, 96]
        samplers: List of samplers to compare ['heun', 'sde', 'ddim']
        num_samples: Number of test samples
        num_steps: Sampling steps
        temperature: SDE temperature (0.5 recommended)
        eta: DDIM stochasticity (0.0-1.0)
        device: Device
        save_dir: Save directory

    Returns:
        results: Nested dict {sampler: {resolution: metrics}}
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # Initialize results structure
    results = {}
    for sampler in samplers:
        results[sampler] = {
            res: {'mse': [], 'mae': [], 'psnr': [], 'ssim': []}
            for res in resolutions
        }

    # Store images for visualization
    sample_images = {
        'input_coords': [],
        'input_values': [],
        'ground_truth_32': [],
        'predictions': {
            sampler: {res: [] for res in resolutions}
            for sampler in samplers
        }
    }

    print(f"\n{'='*70}")
    print("SDE MULTI-SCALE SUPER-RESOLUTION EVALUATION")
    print(f"{'='*70}")
    print(f"Samplers: {samplers}")
    print(f"Resolutions: {resolutions}")
    print(f"Samples: {num_samples}, Steps: {num_steps}")
    print(f"SDE Temperature: {temperature}, DDIM eta: {eta}")
    print(f"{'='*70}\n")

    # Sampler functions
    sampler_fns = {
        'heun': lambda m, oc, ic, iv: heun_sample(m, oc, ic, iv, num_steps, device),
        'sde': lambda m, oc, ic, iv: sde_sample(m, oc, ic, iv, num_steps, temperature, device),
        'ddim': lambda m, oc, ic, iv: ddim_sample(m, oc, ic, iv, num_steps, eta, device)
    }

    # Evaluate samples
    sample_count = 0
    for batch in test_loader:
        if sample_count >= num_samples:
            break

        batch_size = batch['input_coords'].shape[0]
        for i in range(batch_size):
            if sample_count >= num_samples:
                break

            # Get sparse input (20% of 32x32)
            input_coords = batch['input_coords'][i:i+1].to(device)
            input_values = batch['input_values'][i:i+1].to(device)
            ground_truth_32 = batch['full_image'][i].to(device)

            # Store for visualization
            if sample_count < 4:
                sample_images['input_coords'].append(input_coords.cpu())
                sample_images['input_values'].append(input_values.cpu())
                sample_images['ground_truth_32'].append(ground_truth_32.cpu())

            print(f"Sample {sample_count + 1}/{num_samples}:")

            # Evaluate each sampler
            for sampler_name in samplers:
                print(f"  [{sampler_name.upper()}]")
                sampler_fn = sampler_fns[sampler_name]

                # Evaluate each resolution
                for res in resolutions:
                    # Create coordinate grid
                    target_coords = create_coordinate_grid(res, device)
                    target_coords_batch = target_coords.unsqueeze(0)

                    # Generate with current sampler
                    pred_values = sampler_fn(model, target_coords_batch, input_coords, input_values)

                    # Reshape to image
                    pred_image = pred_values.view(1, res, res, 3).permute(0, 3, 1, 2)[0]

                    # Upscale ground truth for comparison
                    gt_upscaled = bicubic_upscale(ground_truth_32, res)

                    # Compute metrics
                    pred_np = pred_image.cpu().permute(1, 2, 0).numpy().clip(0, 1)
                    gt_np = gt_upscaled.cpu().permute(1, 2, 0).numpy().clip(0, 1)
                    metrics = compute_metrics(pred_np, gt_np)

                    # Store metrics
                    for key in ['mse', 'mae', 'psnr', 'ssim']:
                        results[sampler_name][res][key].append(metrics[key])

                    # Store image for visualization
                    if sample_count < 4:
                        sample_images['predictions'][sampler_name][res].append(pred_image.cpu())

                    print(f"    {res}x{res}: PSNR={metrics['psnr']:.2f}dB, SSIM={metrics['ssim']:.4f}")

            sample_count += 1
            print()

    # Compute average metrics
    print(f"\n{'='*70}")
    print("AVERAGE METRICS COMPARISON")
    print(f"{'='*70}\n")

    summary = {}
    for sampler_name in samplers:
        print(f"{sampler_name.upper()} Sampler:")
        summary[sampler_name] = {}

        for res in resolutions:
            summary[sampler_name][res] = {}
            print(f"  {res}x{res}:")

            for key in ['mse', 'mae', 'psnr', 'ssim']:
                avg = np.mean(results[sampler_name][res][key])
                std = np.std(results[sampler_name][res][key])
                summary[sampler_name][res][key] = {'mean': avg, 'std': std}

                if key == 'psnr':
                    print(f"    {key.upper()}: {avg:.2f} ± {std:.2f} dB")
                elif key in ['mse', 'mae']:
                    print(f"    {key.upper()}: {avg:.6f} ± {std:.6f}")
                else:
                    print(f"    {key.upper()}: {avg:.4f} ± {std:.4f}")
        print()

    print(f"{'='*70}\n")

    # Save results
    save_results(results, summary, samplers, resolutions, save_dir)

    # Create visualizations
    visualize_multiscale(sample_images, samplers, resolutions, save_dir)
    visualize_comparison_chart(summary, samplers, resolutions, save_dir)

    return results, summary


def save_results(results, summary, samplers, resolutions, save_dir):
    """Save detailed results to text file"""
    results_file = os.path.join(save_dir, 'metrics_comparison.txt')

    with open(results_file, 'w') as f:
        f.write(f"SDE Multi-Scale Evaluation Results\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("="*70 + "\n\n")

        # Summary
        f.write("AVERAGE METRICS COMPARISON\n")
        f.write("-"*70 + "\n\n")

        for sampler_name in samplers:
            f.write(f"{sampler_name.upper()} Sampler:\n")
            for res in resolutions:
                f.write(f"  {res}x{res}:\n")
                for key in ['mse', 'mae', 'psnr', 'ssim']:
                    mean = summary[sampler_name][res][key]['mean']
                    std = summary[sampler_name][res][key]['std']
                    if key == 'psnr':
                        f.write(f"    {key.upper()}: {mean:.2f} ± {std:.2f} dB\n")
                    elif key in ['mse', 'mae']:
                        f.write(f"    {key.upper()}: {mean:.6f} ± {std:.6f}\n")
                    else:
                        f.write(f"    {key.upper()}: {mean:.4f} ± {std:.4f}\n")
            f.write("\n")

        # Best performer per resolution
        f.write("\n" + "="*70 + "\n")
        f.write("BEST PERFORMER PER RESOLUTION\n")
        f.write("-"*70 + "\n\n")

        for res in resolutions:
            f.write(f"{res}x{res}:\n")
            for metric in ['psnr', 'ssim']:
                best_sampler = max(samplers,
                    key=lambda s: summary[s][res][metric]['mean'])
                best_val = summary[best_sampler][res][metric]['mean']
                f.write(f"  Best {metric.upper()}: {best_sampler.upper()} ({best_val:.4f})\n")
            f.write("\n")

    print(f"✓ Results saved to {results_file}")


def visualize_multiscale(sample_images, samplers, resolutions, save_dir):
    """Create comprehensive multi-scale visualization"""

    num_samples = len(sample_images['ground_truth_32'])
    if num_samples == 0:
        return

    # Create figure: columns = [Input, GT] + [samplers × resolutions]
    num_cols = 2 + len(samplers) * len(resolutions)
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(3.5*num_cols, 3.5*num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        col_idx = 0

        # Column 0: Sparse input
        input_img = torch.zeros(3, 32, 32)
        input_idx = (sample_images['input_coords'][i][0] * 31).long()
        input_idx_flat = input_idx[:, 1] * 32 + input_idx[:, 0]
        input_img.view(3, -1)[:, input_idx_flat] = sample_images['input_values'][i][0].T

        axes[i, col_idx].imshow(input_img.permute(1, 2, 0).numpy())
        axes[i, col_idx].set_title('Input (20%)' if i == 0 else '', fontsize=10)
        axes[i, col_idx].axis('off')
        col_idx += 1

        # Column 1: Ground truth 32x32
        gt_32 = sample_images['ground_truth_32'][i].permute(1, 2, 0).numpy()
        axes[i, col_idx].imshow(gt_32)
        axes[i, col_idx].set_title('GT 32x32' if i == 0 else '', fontsize=10)
        axes[i, col_idx].axis('off')
        col_idx += 1

        # Remaining columns: Samplers × Resolutions
        for sampler_name in samplers:
            for res in resolutions:
                pred_img = sample_images['predictions'][sampler_name][res][i]
                pred_np = pred_img.permute(1, 2, 0).numpy().clip(0, 1)

                axes[i, col_idx].imshow(pred_np)
                title = f'{sampler_name.upper()}\n{res}x{res}' if i == 0 else ''
                axes[i, col_idx].set_title(title, fontsize=9)
                axes[i, col_idx].axis('off')
                col_idx += 1

    plt.suptitle('SDE Multi-Scale Super-Resolution Comparison', fontsize=14, y=0.998)
    plt.tight_layout()

    viz_path = os.path.join(save_dir, 'multiscale_comparison.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualization saved to {viz_path}")


def visualize_comparison_chart(summary, samplers, resolutions, save_dir):
    """Create bar chart comparing samplers across resolutions"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metrics = ['psnr', 'ssim', 'mse', 'mae']
    metric_labels = ['PSNR (dB) ↑', 'SSIM ↑', 'MSE ↓', 'MAE ↓']

    x = np.arange(len(resolutions))
    width = 0.25

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        for i, sampler in enumerate(samplers):
            means = [summary[sampler][res][metric]['mean'] for res in resolutions]
            stds = [summary[sampler][res][metric]['std'] for res in resolutions]

            ax.bar(x + i*width, means, width, yerr=stds,
                   label=sampler.upper(), alpha=0.8, capsize=5)

        ax.set_xlabel('Resolution', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label.split()[0]} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'{r}x{r}' for r in resolutions])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Sampler Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    chart_path = os.path.join(save_dir, 'performance_chart.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Performance chart saved to {chart_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='SDE Multi-Scale Super-Resolution Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--resolutions', type=int, nargs='+', default=[32, 64, 96],
                        help='Target resolutions (default: 32 64 96)')
    parser.add_argument('--samplers', type=str, nargs='+', default=['heun', 'sde', 'ddim'],
                        help='Samplers to compare (default: heun sde ddim)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of test samples (default: 10)')
    parser.add_argument('--num_steps', type=int, default=50,
                        help='Number of sampling steps (default: 50)')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='SDE temperature (default: 0.5)')
    parser.add_argument('--eta', type=float, default=0.3,
                        help='DDIM stochasticity (default: 0.3)')
    parser.add_argument('--save_dir', type=str, default='eval_sde_multiscale',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Device: auto/cuda/cpu')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension (default: 512)')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of layers (default: 6)')

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
    print(f"Model parameters: {num_params:,}\n")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found at {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if 'epoch' in checkpoint:
        print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"  Validation loss: {checkpoint['val_loss']:.6f}")
    print()

    # Run evaluation
    results, summary = evaluate_sde_multiscale(
        model=model,
        test_loader=test_loader,
        resolutions=args.resolutions,
        samplers=args.samplers,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        temperature=args.temperature,
        eta=args.eta,
        device=device,
        save_dir=args.save_dir
    )

    print("✓ Evaluation complete!")
    print(f"  Results saved to: {args.save_dir}/")


if __name__ == '__main__':
    main()
