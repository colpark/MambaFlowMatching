"""
Super-Resolution Evaluation Script for MAMBA Diffusion

Evaluates the model's zero-shot super-resolution capability by:
1. Loading trained model from checkpoint
2. Using 20% sparse data from 32x32 images as input
3. Generating outputs at 64x64, 96x96, 128x128, and 256x256 resolutions
4. Computing metrics (PSNR, SSIM, MSE, MAE) for each resolution
5. Visualizing results with side-by-side comparisons
"""
import sys
import os
import argparse
from datetime import datetime

# Add repository root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
v1_dir = os.path.dirname(script_dir)  # v1 or v2
    repo_root = os.path.dirname(v1_dir)   # MambaFlowMatching
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from train_mamba_standalone import MAMBADiffusion, heun_sample
from core.sparse.cifar10_sparse import SparseCIFAR10Dataset


# ============================================================================
# Super-Resolution Helper Functions
# ============================================================================

def create_coordinate_grid(resolution, device):
    """
    Create normalized coordinate grid for given resolution

    Args:
        resolution: int, output resolution (e.g., 64, 96, 128, 256)
        device: torch device

    Returns:
        coords: (resolution^2, 2) normalized coordinates in [0, 1]
    """
    y, x = torch.meshgrid(
        torch.linspace(0, 1, resolution),
        torch.linspace(0, 1, resolution),
        indexing='ij'
    )
    coords = torch.stack([x.flatten(), y.flatten()], dim=-1).to(device)
    return coords


def bicubic_upscale(image, target_size):
    """
    Bicubic upscaling baseline for comparison

    Args:
        image: (3, H, W) tensor
        target_size: int, target resolution

    Returns:
        upscaled: (3, target_size, target_size) tensor
    """
    return F.interpolate(
        image.unsqueeze(0),
        size=(target_size, target_size),
        mode='bicubic',
        align_corners=False
    ).squeeze(0)


def compute_metrics(pred, target):
    """
    Compute image quality metrics

    Args:
        pred: (H, W, 3) numpy array [0, 1]
        target: (H, W, 3) numpy array [0, 1]

    Returns:
        dict with metrics
    """
    # Ensure float64 for accurate computation
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)

    # MSE and MAE
    mse = np.mean((pred - target) ** 2)
    mae = np.mean(np.abs(pred - target))

    # PSNR (computed on [0, 1] range)
    psnr_val = psnr(target, pred, data_range=1.0)

    # SSIM (computed per channel then averaged)
    ssim_val = ssim(target, pred, data_range=1.0, channel_axis=2)

    return {
        'mse': mse,
        'mae': mae,
        'psnr': psnr_val,
        'ssim': ssim_val
    }


# ============================================================================
# Evaluation Functions
# ============================================================================

@torch.no_grad()
def evaluate_superresolution(
    model,
    test_loader,
    resolutions=[64, 96, 128, 256],
    num_samples=10,
    num_steps=50,
    device='cuda',
    save_dir='eval_superres'
):
    """
    Evaluate super-resolution capacity at multiple resolutions

    Args:
        model: Trained MAMBA diffusion model
        test_loader: Test data loader (32x32 images)
        resolutions: List of target resolutions to evaluate
        num_samples: Number of test samples to evaluate
        num_steps: Number of sampling steps
        device: Device to run on
        save_dir: Directory to save results

    Returns:
        results: Dict with metrics for each resolution
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # Store results for each resolution
    results = {res: {'mse': [], 'mae': [], 'psnr': [], 'ssim': []} for res in resolutions}

    # Store images for visualization
    sample_images = {
        'input_coords': [],
        'input_values': [],
        'ground_truth_32': [],
        'predictions': {res: [] for res in resolutions}
    }

    print(f"\n{'='*60}")
    print("SUPER-RESOLUTION EVALUATION")
    print(f"{'='*60}")
    print(f"Evaluating {num_samples} samples at resolutions: {resolutions}")
    print(f"Using {num_steps} sampling steps")
    print(f"{'='*60}\n")

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
            ground_truth_32 = batch['full_image'][i].to(device)  # (3, 32, 32)

            # Store for visualization
            if sample_count < 4:  # Only store first 4 for detailed viz
                sample_images['input_coords'].append(input_coords.cpu())
                sample_images['input_values'].append(input_values.cpu())
                sample_images['ground_truth_32'].append(ground_truth_32.cpu())

            print(f"Sample {sample_count + 1}/{num_samples}:")

            # Evaluate each resolution
            for res in resolutions:
                # Create coordinate grid for target resolution
                target_coords = create_coordinate_grid(res, device)
                target_coords_batch = target_coords.unsqueeze(0)  # (1, res^2, 2)

                # Generate super-resolved image
                pred_values = heun_sample(
                    model,
                    target_coords_batch,
                    input_coords,
                    input_values,
                    num_steps=num_steps,
                    device=device
                )  # (1, res^2, 3)

                # Reshape to image
                pred_image = pred_values.view(1, res, res, 3).permute(0, 3, 1, 2)[0]  # (3, res, res)

                # Upscale ground truth to target resolution for comparison
                gt_upscaled = bicubic_upscale(ground_truth_32, res)

                # Convert to numpy for metric computation
                pred_np = pred_image.cpu().permute(1, 2, 0).numpy().clip(0, 1)
                gt_np = gt_upscaled.cpu().permute(1, 2, 0).numpy().clip(0, 1)

                # Compute metrics
                metrics = compute_metrics(pred_np, gt_np)

                # Store metrics
                for key in ['mse', 'mae', 'psnr', 'ssim']:
                    results[res][key].append(metrics[key])

                # Store image for visualization
                if sample_count < 4:
                    sample_images['predictions'][res].append(pred_image.cpu())

                print(f"  {res}x{res}: PSNR={metrics['psnr']:.2f}dB, "
                      f"SSIM={metrics['ssim']:.4f}, MSE={metrics['mse']:.6f}")

            sample_count += 1
            print()

    # Compute average metrics
    print(f"\n{'='*60}")
    print("AVERAGE METRICS")
    print(f"{'='*60}")

    summary = {}
    for res in resolutions:
        summary[res] = {}
        print(f"\n{res}x{res} Resolution:")
        for key in ['mse', 'mae', 'psnr', 'ssim']:
            avg = np.mean(results[res][key])
            std = np.std(results[res][key])
            summary[res][key] = {'mean': avg, 'std': std}

            if key == 'psnr':
                print(f"  {key.upper()}: {avg:.2f} ± {std:.2f} dB")
            elif key in ['mse', 'mae']:
                print(f"  {key.upper()}: {avg:.6f} ± {std:.6f}")
            else:
                print(f"  {key.upper()}: {avg:.4f} ± {std:.4f}")

    print(f"\n{'='*60}\n")

    # Save detailed results
    save_results(results, summary, save_dir)

    # Create visualizations
    visualize_superresolution(sample_images, resolutions, save_dir)

    return results, summary


def save_results(results, summary, save_dir):
    """Save detailed results to text file"""
    results_file = os.path.join(save_dir, 'metrics.txt')

    with open(results_file, 'w') as f:
        f.write(f"Super-Resolution Evaluation Results\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("="*60 + "\n\n")

        f.write("AVERAGE METRICS\n")
        f.write("-"*60 + "\n")
        for res in sorted(summary.keys()):
            f.write(f"\n{res}x{res} Resolution:\n")
            for key in ['mse', 'mae', 'psnr', 'ssim']:
                mean = summary[res][key]['mean']
                std = summary[res][key]['std']
                if key == 'psnr':
                    f.write(f"  {key.upper()}: {mean:.2f} ± {std:.2f} dB\n")
                elif key in ['mse', 'mae']:
                    f.write(f"  {key.upper()}: {mean:.6f} ± {std:.6f}\n")
                else:
                    f.write(f"  {key.upper()}: {mean:.4f} ± {std:.4f}\n")

        f.write("\n" + "="*60 + "\n\n")
        f.write("PER-SAMPLE METRICS\n")
        f.write("-"*60 + "\n")
        for res in sorted(results.keys()):
            f.write(f"\n{res}x{res} Resolution:\n")
            for i in range(len(results[res]['psnr'])):
                f.write(f"  Sample {i+1}: ")
                f.write(f"PSNR={results[res]['psnr'][i]:.2f}dB, ")
                f.write(f"SSIM={results[res]['ssim'][i]:.4f}, ")
                f.write(f"MSE={results[res]['mse'][i]:.6f}\n")

    print(f"✓ Results saved to {results_file}")


def visualize_superresolution(sample_images, resolutions, save_dir):
    """Create comprehensive visualization of super-resolution results"""

    num_samples = len(sample_images['ground_truth_32'])
    if num_samples == 0:
        return

    # Create figure with all resolutions
    num_cols = 2 + len(resolutions)  # Input + GT + predictions at each resolution
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(4*num_cols, 4*num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Column 0: Sparse input visualization
        input_img = torch.zeros(3, 32, 32)
        input_idx = (sample_images['input_coords'][i][0] * 31).long()  # Convert to pixel indices
        input_idx_flat = input_idx[:, 1] * 32 + input_idx[:, 0]
        input_img.view(3, -1)[:, input_idx_flat] = sample_images['input_values'][i][0].T

        axes[i, 0].imshow(input_img.permute(1, 2, 0).numpy())
        axes[i, 0].set_title('Input (20% sparse)' if i == 0 else '', fontsize=12)
        axes[i, 0].axis('off')

        # Column 1: Ground truth 32x32
        gt_32 = sample_images['ground_truth_32'][i].permute(1, 2, 0).numpy()
        axes[i, 1].imshow(gt_32)
        axes[i, 1].set_title('GT 32x32' if i == 0 else '', fontsize=12)
        axes[i, 1].axis('off')

        # Remaining columns: Predictions at each resolution
        for col_idx, res in enumerate(resolutions):
            pred_img = sample_images['predictions'][res][i].permute(1, 2, 0).numpy()
            pred_img = np.clip(pred_img, 0, 1)

            axes[i, 2 + col_idx].imshow(pred_img)
            axes[i, 2 + col_idx].set_title(f'Pred {res}x{res}' if i == 0 else '', fontsize=12)
            axes[i, 2 + col_idx].axis('off')

    plt.suptitle('Zero-Shot Super-Resolution Results', fontsize=16, y=0.995)
    plt.tight_layout()

    viz_path = os.path.join(save_dir, 'superresolution_comparison.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualization saved to {viz_path}")

    # Create separate high-resolution individual images
    for res in resolutions:
        fig, axes = plt.subplots(1, min(num_samples, 4), figsize=(16, 4))
        if num_samples == 1:
            axes = [axes]

        for i in range(min(num_samples, 4)):
            pred_img = sample_images['predictions'][res][i].permute(1, 2, 0).numpy()
            pred_img = np.clip(pred_img, 0, 1)
            axes[i].imshow(pred_img)
            axes[i].set_title(f'Sample {i+1}', fontsize=12)
            axes[i].axis('off')

        plt.suptitle(f'Super-Resolution {res}x{res}', fontsize=14)
        plt.tight_layout()

        res_path = os.path.join(save_dir, f'superres_{res}x{res}.png')
        plt.savefig(res_path, dpi=200, bbox_inches='tight')
        plt.close()

    print(f"✓ Individual resolution images saved")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate Super-Resolution Capacity')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (e.g., checkpoints_mamba/mamba_best.pth)')
    parser.add_argument('--resolutions', type=int, nargs='+', default=[64, 96, 128, 256],
                        help='Target resolutions to evaluate (default: 64 96 128 256)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of test samples to evaluate (default: 10)')
    parser.add_argument('--num_steps', type=int, default=50,
                        help='Number of sampling steps (default: 50)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation (default: 1)')
    parser.add_argument('--save_dir', type=str, default='eval_superres',
                        help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Device to use: auto (default), cuda, or cpu')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension (must match checkpoint, default: 512)')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of layers (must match checkpoint, default: 6)')

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
            print("❌ CUDA not available, falling back to CPU")
            device = torch.device('cpu')
    else:  # auto
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("⚠️  CUDA not available, using CPU")

    print("=" * 60 + "\n")

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0  # Set to 0 for evaluation to avoid multiprocessing issues
    )
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
    results, summary = evaluate_superresolution(
        model=model,
        test_loader=test_loader,
        resolutions=args.resolutions,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        device=device,
        save_dir=args.save_dir
    )

    print("✓ Evaluation complete!")
    print(f"  Results saved to: {args.save_dir}/")


if __name__ == '__main__':
    main()
