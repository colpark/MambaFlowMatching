#!/usr/bin/env python3
"""
Analyze and Visualize Improvement Experiment Results

Creates comprehensive visualizations and analysis of 100 version results
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import argparse


def load_results(results_path='results_improvements/all_results.json'):
    """Load all results"""
    with open(results_path, 'r') as f:
        return json.load(f)


def create_technique_heatmap(results, output_path='results_improvements/technique_heatmap.png'):
    """
    Create heatmap showing technique combinations and their PSNR
    """
    # Prepare data
    data = []
    for version, result in results.items():
        row = [int(version[1:])]  # version number
        techniques = result.get('techniques', [])
        psnr = result.get('final_psnr', 0)

        # One-hot encode techniques
        for i in range(1, 11):
            row.append(1 if i in techniques else 0)

        row.append(psnr)
        data.append(row)

    # Create DataFrame
    columns = ['version'] + [f'T{i}' for i in range(1, 11)] + ['PSNR']
    df = pd.DataFrame(data, columns=columns)

    # Sort by PSNR
    df = df.sort_values('PSNR', ascending=False)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12), gridspec_kw={'width_ratios': [3, 1]})

    # Heatmap of technique combinations
    technique_data = df[[f'T{i}' for i in range(1, 11)]].values
    im = ax1.imshow(technique_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')

    ax1.set_xlabel('Technique ID', fontsize=12)
    ax1.set_ylabel('Version (sorted by PSNR)', fontsize=12)
    ax1.set_title('Technique Combination Heatmap (Top to Bottom: Best to Worst)', fontsize=14)
    ax1.set_xticks(range(10))
    ax1.set_xticklabels([f'T{i}' for i in range(1, 11)])

    # Add colorbar
    plt.colorbar(im, ax=ax1, label='Technique Present')

    # PSNR bars
    psnr_values = df['PSNR'].values
    colors = plt.cm.viridis(np.linspace(0, 1, len(psnr_values)))
    ax2.barh(range(len(psnr_values)), psnr_values, color=colors)
    ax2.set_xlabel('PSNR (dB)', fontsize=12)
    ax2.set_ylabel('Version', fontsize=12)
    ax2.set_title('Performance', fontsize=14)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"📊 Saved technique heatmap to: {output_path}")


def create_improvement_distribution(results, baseline_psnr=35.0, output_path='results_improvements/improvement_dist.png'):
    """
    Show distribution of improvements over baseline
    """
    improvements = []
    for version, result in results.items():
        psnr = result.get('final_psnr', 0)
        improvement = psnr - baseline_psnr
        improvements.append(improvement)

    # Create histogram
    plt.figure(figsize=(10, 6))

    plt.hist(improvements, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Baseline')
    plt.axvline(np.mean(improvements), color='green', linestyle='--', linewidth=2, label=f'Mean: +{np.mean(improvements):.2f} dB')
    plt.axvline(np.median(improvements), color='orange', linestyle='--', linewidth=2, label=f'Median: +{np.median(improvements):.2f} dB')

    plt.xlabel('Improvement over Baseline (dB)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Distribution of PSNR Improvements (Baseline: {baseline_psnr:.1f} dB)', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"📊 Saved improvement distribution to: {output_path}")


def create_technique_impact_plot(results, baseline_psnr=35.0, output_path='results_improvements/technique_impact.png'):
    """
    Plot individual technique impacts
    """
    # Calculate impact for each technique
    technique_impacts = {i: [] for i in range(1, 11)}

    for version, result in results.items():
        techniques = result.get('techniques', [])
        psnr = result.get('final_psnr', 0)
        improvement = psnr - baseline_psnr

        # Single technique analysis
        if len(techniques) == 1:
            tid = techniques[0]
            technique_impacts[tid].append(improvement)

    # Compute statistics
    tech_stats = []
    for tid in range(1, 11):
        if technique_impacts[tid]:
            mean_impact = np.mean(technique_impacts[tid])
            std_impact = np.std(technique_impacts[tid])
            n_samples = len(technique_impacts[tid])
            tech_stats.append({
                'id': tid,
                'mean': mean_impact,
                'std': std_impact,
                'n': n_samples
            })

    # Sort by mean impact
    tech_stats.sort(key=lambda x: x['mean'], reverse=True)

    # Plot
    plt.figure(figsize=(12, 6))

    x = range(len(tech_stats))
    means = [t['mean'] for t in tech_stats]
    stds = [t['std'] for t in tech_stats]
    labels = [f"T{t['id']}\n(n={t['n']})" for t in tech_stats]
    colors = plt.cm.viridis(np.linspace(0, 1, len(tech_stats)))

    plt.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=colors, edgecolor='black')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)

    plt.xlabel('Technique', fontsize=12)
    plt.ylabel('PSNR Improvement over Baseline (dB)', fontsize=12)
    plt.title('Individual Technique Impact (Single Technique Versions Only)', fontsize=14)
    plt.xticks(x, labels)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"📊 Saved technique impact plot to: {output_path}")


def create_combination_size_analysis(results, baseline_psnr=35.0, output_path='results_improvements/combination_size.png'):
    """
    Analyze performance vs number of techniques combined
    """
    # Group by combination size
    size_data = {i: [] for i in range(1, 11)}

    for version, result in results.items():
        techniques = result.get('techniques', [])
        psnr = result.get('final_psnr', 0)
        improvement = psnr - baseline_psnr
        size = len(techniques)

        if size > 0:
            size_data[size].append(improvement)

    # Create box plot
    plt.figure(figsize=(12, 6))

    # Filter non-empty sizes
    sizes = [s for s in range(1, 11) if size_data[s]]
    data = [size_data[s] for s in sizes]

    bp = plt.boxplot(data, labels=sizes, patch_artist=True)

    # Color boxes
    colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.axhline(0, color='red', linestyle='--', linewidth=1, label='Baseline')

    plt.xlabel('Number of Techniques Combined', fontsize=12)
    plt.ylabel('PSNR Improvement over Baseline (dB)', fontsize=12)
    plt.title('Performance vs Combination Size', fontsize=14)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"📊 Saved combination size analysis to: {output_path}")


def create_top_combinations_table(results, top_n=20, output_path='results_improvements/top_combinations.md'):
    """
    Create markdown table of top combinations
    """
    # Sort by PSNR
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('final_psnr', 0), reverse=True)

    # Create table
    with open(output_path, 'w') as f:
        f.write(f"# Top {top_n} Combinations\n\n")
        f.write("| Rank | Version | PSNR (dB) | Improvement | Techniques | Time (min) |\n")
        f.write("|------|---------|-----------|-------------|------------|------------|\n")

        baseline_psnr = 35.0  # Assume baseline

        for i, (version, result) in enumerate(sorted_results[:top_n], 1):
            psnr = result.get('final_psnr', 0)
            improvement = psnr - baseline_psnr
            techniques = result.get('techniques', [])
            time_min = result.get('training_time', 0) / 60

            tech_str = ','.join(map(str, techniques))

            f.write(f"| {i} | {version} | {psnr:.2f} | +{improvement:.2f} | {tech_str} | {time_min:.1f} |\n")

    print(f"📄 Saved top combinations table to: {output_path}")


def print_summary(results, baseline_psnr=35.0):
    """Print summary statistics"""
    print(f"\n{'='*70}")
    print("Results Summary")
    print(f"{'='*70}\n")

    # Overall statistics
    total_versions = len(results)
    psnrs = [r.get('final_psnr', 0) for r in results.values()]
    improvements = [p - baseline_psnr for p in psnrs]

    print(f"Total versions completed: {total_versions}/100")
    print(f"Success rate: {total_versions/100*100:.1f}%\n")

    print(f"Performance Statistics:")
    print(f"  Baseline (V2): {baseline_psnr:.2f} dB")
    print(f"  Best: {max(psnrs):.2f} dB (+{max(improvements):.2f} dB)")
    print(f"  Mean: {np.mean(psnrs):.2f} dB (+{np.mean(improvements):.2f} dB)")
    print(f"  Median: {np.median(psnrs):.2f} dB (+{np.median(improvements):.2f} dB)")
    print(f"  Std: {np.std(psnrs):.2f} dB\n")

    # Top 5 versions
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('final_psnr', 0), reverse=True)

    print("🏆 Top 5 Versions:")
    for i, (version, result) in enumerate(sorted_results[:5], 1):
        psnr = result.get('final_psnr', 0)
        improvement = psnr - baseline_psnr
        techniques = result.get('techniques', [])
        print(f"  {i}. {version}: {psnr:.2f} dB (+{improvement:.2f} dB) | Techniques: {techniques}")

    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default='results_improvements/all_results.json')
    parser.add_argument('--baseline_psnr', type=float, default=35.0, help='V2 baseline PSNR')
    parser.add_argument('--output_dir', type=str, default='results_improvements')
    args = parser.parse_args()

    # Load results
    print(f"📂 Loading results from: {args.results}")
    results = load_results(args.results)

    # Print summary
    print_summary(results, args.baseline_psnr)

    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)

    # Generate visualizations
    print("📊 Generating visualizations...\n")

    create_technique_heatmap(results, output_path=f'{args.output_dir}/technique_heatmap.png')
    create_improvement_distribution(results, args.baseline_psnr, output_path=f'{args.output_dir}/improvement_dist.png')
    create_technique_impact_plot(results, args.baseline_psnr, output_path=f'{args.output_dir}/technique_impact.png')
    create_combination_size_analysis(results, args.baseline_psnr, output_path=f'{args.output_dir}/combination_size.png')
    create_top_combinations_table(results, top_n=20, output_path=f'{args.output_dir}/top_combinations.md')

    print(f"\n✅ Analysis complete! Visualizations saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
