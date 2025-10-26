"""
Comprehensive Comparison of Different Methods on Synthetic Data

Evaluates and compares:
1. Baseline MAMBA
2. Content-Aware Sampling
3. Wavelet-based
4. Hierarchical MAMBA
5. Latent Space
6. Other improvement methods
"""
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
synthetic_dir = os.path.dirname(script_dir)
repo_root = os.path.dirname(synthetic_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from typing import Dict, List
import argparse

from synthetic_experiments.datasets import SinusoidalDataset


class MetricsCalculator:
    """Calculate reconstruction metrics"""

    @staticmethod
    def mse(pred, target):
        return torch.mean((pred - target) ** 2).item()

    @staticmethod
    def mae(pred, target):
        return torch.mean(torch.abs(pred - target)).item()

    @staticmethod
    def psnr(pred, target, max_val=1.0):
        mse = torch.mean((pred - target) ** 2).item()
        if mse == 0:
            return float('inf')
        return 20 * np.log10(max_val / np.sqrt(mse))

    @staticmethod
    def relative_error(pred, target):
        return (torch.norm(pred - target) / torch.norm(target)).item()

    @staticmethod
    def correlation(pred, target):
        pred_flat = pred.flatten()
        target_flat = target.flatten()

        pred_mean = pred_flat.mean()
        target_mean = target_flat.mean()

        numerator = torch.sum((pred_flat - pred_mean) * (target_flat - target_mean))
        denominator = torch.sqrt(
            torch.sum((pred_flat - pred_mean) ** 2) *
            torch.sum((target_flat - target_mean) ** 2)
        )

        return (numerator / denominator).item()

    @staticmethod
    def frequency_error(pred, target):
        """Error in frequency domain (using FFT)"""
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)

        error = torch.mean(torch.abs(pred_fft - target_fft) ** 2).item()
        return error


def evaluate_model(model, dataset, sparsity=0.2, num_samples=100, device='cpu'):
    """
    Evaluate a model on dataset

    Returns metrics dictionary
    """
    model.eval()
    model = model.to(device)

    # Get sparse observations
    coords_sparse, values_sparse, full_data = dataset.get_sparse_observations(
        sparsity=sparsity,
        strategy='random'
    )

    # Limit to num_samples
    coords_sparse = coords_sparse[:num_samples]
    values_sparse = values_sparse[:num_samples]
    full_data = full_data[:num_samples]

    # Create query coordinates
    H, W = dataset.resolution, dataset.resolution
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing='ij'
    )
    query_coords = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)
    query_coords = query_coords.unsqueeze(0).expand(num_samples, -1, -1)

    # Evaluate
    metrics = {
        'mse': [],
        'mae': [],
        'psnr': [],
        'relative_error': [],
        'correlation': [],
        'frequency_error': []
    }

    calc = MetricsCalculator()

    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Evaluating"):
            coords_batch = coords_sparse[i:i+1].to(device)
            values_batch = values_sparse[i:i+1].to(device)
            query_batch = query_coords[i:i+1].to(device)
            target_batch = full_data[i:i+1].to(device)

            # Sample (assuming model has sample method)
            try:
                # Try Heun sampling
                from synthetic_experiments.baselines.train_baseline_mamba import sample_heun
                pred = sample_heun(model, coords_batch, values_batch, query_batch,
                                 num_steps=50, device=device)
            except:
                # Direct forward pass (for non-diffusion models)
                t = torch.ones(1, device=device)
                pred = model(coords_batch, values_batch, query_batch, t)

            # Reshape
            pred = pred.reshape(1, 1, H, W)

            # Calculate metrics
            metrics['mse'].append(calc.mse(pred, target_batch))
            metrics['mae'].append(calc.mae(pred, target_batch))
            metrics['psnr'].append(calc.psnr(pred, target_batch))
            metrics['relative_error'].append(calc.relative_error(pred, target_batch))
            metrics['correlation'].append(calc.correlation(pred, target_batch))
            metrics['frequency_error'].append(calc.frequency_error(pred[0, 0], target_batch[0, 0]))

    # Aggregate
    results = {}
    for key in metrics:
        values = metrics[key]
        results[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values)
        }

    return results


def compare_methods_on_complexity(
    complexity: str,
    method_checkpoints: Dict[str, str],
    num_samples: int = 100,
    sparsity: float = 0.2,
    device: str = 'cpu',
    save_dir: str = 'results'
):
    """
    Compare multiple methods on a specific complexity level

    Args:
        complexity: Dataset complexity level
        method_checkpoints: Dict mapping method names to checkpoint paths
        num_samples: Number of test samples
        sparsity: Observation sparsity
        device: Compute device
        save_dir: Results directory
    """
    print(f"\n{'='*70}")
    print(f"Comparing Methods on Complexity: {complexity}")
    print(f"{'='*70}\n")

    # Create dataset
    dataset = SinusoidalDataset(
        resolution=32,
        num_samples=num_samples,
        complexity=complexity,
        noise_level=0.0
    )

    results = {}

    # Evaluate each method
    for method_name, checkpoint_path in method_checkpoints.items():
        print(f"\n📊 Evaluating: {method_name}")
        print(f"   Checkpoint: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            print(f"   ⚠️  Checkpoint not found, skipping...")
            continue

        # Load model (assuming BaselineMAMBAFlow architecture)
        # TODO: Extend for other architectures
        from synthetic_experiments.baselines.train_baseline_mamba import BaselineMAMBAFlow

        model = BaselineMAMBAFlow(d_model=128, num_layers=4)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate
        method_results = evaluate_model(
            model, dataset, sparsity=sparsity,
            num_samples=num_samples, device=device
        )

        results[method_name] = method_results

        # Print summary
        print(f"   ✅ Results:")
        print(f"      PSNR: {method_results['psnr']['mean']:.2f} ± {method_results['psnr']['std']:.2f} dB")
        print(f"      MSE:  {method_results['mse']['mean']:.6f} ± {method_results['mse']['std']:.6f}")
        print(f"      Corr: {method_results['correlation']['mean']:.4f} ± {method_results['correlation']['std']:.4f}")

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, f'{complexity}_comparison.json')

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Saved results to: {results_path}")

    return results


def visualize_comparison(
    results_dict: Dict[str, Dict],
    save_path: str = 'comparison.png'
):
    """
    Visualize comparison across methods

    Args:
        results_dict: Dict of {complexity: {method: results}}
        save_path: Where to save visualization
    """
    complexities = list(results_dict.keys())
    methods = list(list(results_dict.values())[0].keys())

    metrics_to_plot = ['psnr', 'mse', 'correlation']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]

        x = np.arange(len(complexities))
        width = 0.8 / len(methods)

        for i, method in enumerate(methods):
            means = []
            stds = []

            for complexity in complexities:
                if method in results_dict[complexity]:
                    means.append(results_dict[complexity][method][metric]['mean'])
                    stds.append(results_dict[complexity][method][metric]['std'])
                else:
                    means.append(0)
                    stds.append(0)

            ax.bar(x + i * width, means, width, label=method, yerr=stds, capsize=5)

        ax.set_xlabel('Complexity')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} Comparison')
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(complexities, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"📊 Saved comparison plot to: {save_path}")


def create_comparison_table(results_dict: Dict[str, Dict], save_path: str = 'comparison_table.txt'):
    """Create markdown comparison table"""

    complexities = list(results_dict.keys())
    methods = list(list(results_dict.values())[0].keys())

    with open(save_path, 'w') as f:
        # PSNR Table
        f.write("# PSNR Comparison (dB)\n\n")
        f.write("| Complexity | " + " | ".join(methods) + " |\n")
        f.write("|" + "---|" * (len(methods) + 1) + "\n")

        for complexity in complexities:
            row = [complexity]
            for method in methods:
                if method in results_dict[complexity]:
                    mean = results_dict[complexity][method]['psnr']['mean']
                    std = results_dict[complexity][method]['psnr']['std']
                    row.append(f"{mean:.2f} ± {std:.2f}")
                else:
                    row.append("N/A")
            f.write("| " + " | ".join(row) + " |\n")

        f.write("\n\n")

        # MSE Table
        f.write("# MSE Comparison\n\n")
        f.write("| Complexity | " + " | ".join(methods) + " |\n")
        f.write("|" + "---|" * (len(methods) + 1) + "\n")

        for complexity in complexities:
            row = [complexity]
            for method in methods:
                if method in results_dict[complexity]:
                    mean = results_dict[complexity][method]['mse']['mean']
                    std = results_dict[complexity][method]['mse']['std']
                    row.append(f"{mean:.6f} ± {std:.6f}")
                else:
                    row.append("N/A")
            f.write("| " + " | ".join(row) + " |\n")

    print(f"📄 Saved comparison table to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--complexities', nargs='+',
                       default=['simple', 'multi_frequency', 'radial'],
                       help='Complexity levels to test')
    parser.add_argument('--methods_dir', type=str,
                       default='synthetic_experiments/methods',
                       help='Directory containing method checkpoints')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--sparsity', type=float, default=0.2)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--save_dir', type=str,
                       default='synthetic_experiments/results')

    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Find method checkpoints
    method_checkpoints = {
        'Baseline': 'synthetic_experiments/baselines/checkpoints/best_model.pth',
        # Add more methods here as they're implemented
    }

    all_results = {}

    # Compare on each complexity
    for complexity in args.complexities:
        results = compare_methods_on_complexity(
            complexity=complexity,
            method_checkpoints=method_checkpoints,
            num_samples=args.num_samples,
            sparsity=args.sparsity,
            device=device,
            save_dir=args.save_dir
        )
        all_results[complexity] = results

    # Create visualizations
    visualize_comparison(
        all_results,
        save_path=os.path.join(args.save_dir, 'comparison_plot.png')
    )

    create_comparison_table(
        all_results,
        save_path=os.path.join(args.save_dir, 'comparison_table.md')
    )

    print("\n✅ Comparison complete!")


if __name__ == '__main__':
    main()
