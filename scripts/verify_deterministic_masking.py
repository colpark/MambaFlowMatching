"""
Verify that the sparse sampling is instance-specific and deterministic

This script verifies:
1. Same instance always gets same input/output masks
2. Input and output pixels are non-overlapping (disjoint sets)
3. Sampling pattern is consistent across multiple accesses
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from core.sparse.cifar10_sparse import SparseCIFAR10Dataset

def test_deterministic_sampling():
    """Test that sampling is deterministic with seed"""
    print("=" * 60)
    print("TEST 1: Deterministic Sampling")
    print("=" * 60)

    # Create two datasets with same seed
    dataset1 = SparseCIFAR10Dataset(
        root='../data', train=True, input_ratio=0.2, output_ratio=0.2,
        download=True, seed=42
    )
    dataset2 = SparseCIFAR10Dataset(
        root='../data', train=True, input_ratio=0.2, output_ratio=0.2,
        download=True, seed=42
    )

    # Check multiple instances
    for idx in [0, 100, 1000]:
        sample1 = dataset1[idx]
        sample2 = dataset2[idx]

        # Compare indices
        input_match = torch.equal(sample1['input_indices'], sample2['input_indices'])
        output_match = torch.equal(sample1['output_indices'], sample2['output_indices'])

        print(f"\nInstance {idx}:")
        print(f"  Input indices match: {input_match}")
        print(f"  Output indices match: {output_match}")

        if not (input_match and output_match):
            print("  ❌ FAIL: Indices don't match!")
            return False

    print("\n✅ PASS: All instances have deterministic sampling")
    return True


def test_non_overlapping():
    """Test that input and output pixels are non-overlapping"""
    print("\n" + "=" * 60)
    print("TEST 2: Non-Overlapping Input/Output")
    print("=" * 60)

    dataset = SparseCIFAR10Dataset(
        root='../data', train=True, input_ratio=0.2, output_ratio=0.2,
        download=True, seed=42
    )

    # Check multiple instances
    for idx in [0, 100, 1000, 5000]:
        sample = dataset[idx]

        input_set = set(sample['input_indices'].tolist())
        output_set = set(sample['output_indices'].tolist())

        # Check for overlap
        overlap = input_set & output_set

        print(f"\nInstance {idx}:")
        print(f"  Input pixels: {len(input_set)}")
        print(f"  Output pixels: {len(output_set)}")
        print(f"  Overlap: {len(overlap)} pixels")

        if len(overlap) > 0:
            print(f"  ❌ FAIL: Found {len(overlap)} overlapping pixels!")
            print(f"  Overlapping indices: {list(overlap)[:10]}...")
            return False

    print("\n✅ PASS: All instances have non-overlapping input/output")
    return True


def test_consistency():
    """Test that same instance returns same masks on multiple accesses"""
    print("\n" + "=" * 60)
    print("TEST 3: Consistency Across Multiple Accesses")
    print("=" * 60)

    dataset = SparseCIFAR10Dataset(
        root='../data', train=True, input_ratio=0.2, output_ratio=0.2,
        download=True, seed=42
    )

    # Access same instance multiple times
    idx = 42
    samples = [dataset[idx] for _ in range(5)]

    print(f"\nInstance {idx} accessed 5 times:")

    # Compare all against first
    for i in range(1, len(samples)):
        input_match = torch.equal(samples[0]['input_indices'], samples[i]['input_indices'])
        output_match = torch.equal(samples[0]['output_indices'], samples[i]['output_indices'])

        print(f"  Access {i+1} vs Access 1:")
        print(f"    Input indices match: {input_match}")
        print(f"    Output indices match: {output_match}")

        if not (input_match and output_match):
            print("  ❌ FAIL: Indices changed across accesses!")
            return False

    print("\n✅ PASS: Instance returns consistent masks across accesses")
    return True


def test_coverage():
    """Test that we use exactly 40% of pixels (20% input + 20% output)"""
    print("\n" + "=" * 60)
    print("TEST 4: Pixel Coverage")
    print("=" * 60)

    dataset = SparseCIFAR10Dataset(
        root='../data', train=True, input_ratio=0.2, output_ratio=0.2,
        download=True, seed=42
    )

    total_pixels = 32 * 32  # CIFAR-10

    for idx in [0, 100, 1000]:
        sample = dataset[idx]

        n_input = len(sample['input_indices'])
        n_output = len(sample['output_indices'])
        coverage = (n_input + n_output) / total_pixels

        print(f"\nInstance {idx}:")
        print(f"  Total pixels: {total_pixels}")
        print(f"  Input pixels: {n_input} ({n_input/total_pixels*100:.1f}%)")
        print(f"  Output pixels: {n_output} ({n_output/total_pixels*100:.1f}%)")
        print(f"  Total coverage: {n_input + n_output} ({coverage*100:.1f}%)")

        # Check if close to 40% (within 1 pixel tolerance)
        expected_coverage = 0.4
        if abs(coverage - expected_coverage) > 0.01:
            print(f"  ❌ FAIL: Coverage {coverage*100:.1f}% not close to {expected_coverage*100}%")
            return False

    print("\n✅ PASS: All instances have ~40% coverage")
    return True


def visualize_sampling():
    """Create visualization to show sampling pattern"""
    print("\n" + "=" * 60)
    print("VISUALIZATION: Sampling Patterns")
    print("=" * 60)

    import matplotlib.pyplot as plt
    import numpy as np

    dataset = SparseCIFAR10Dataset(
        root='../data', train=True, input_ratio=0.2, output_ratio=0.2,
        download=True, seed=42
    )

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i in range(2):
        idx = i * 1000  # Sample at different positions
        sample = dataset[idx]

        # Full image
        full_img = sample['full_image'].permute(1, 2, 0).numpy()
        axes[i, 0].imshow(full_img)
        axes[i, 0].set_title(f'Instance {idx}: Full Image')
        axes[i, 0].axis('off')

        # Input mask (20%)
        input_mask = torch.zeros(32, 32)
        input_idx = sample['input_indices']
        y_in = input_idx // 32
        x_in = input_idx % 32
        input_mask[y_in, x_in] = 1
        axes[i, 1].imshow(input_mask.numpy(), cmap='Blues', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Input Mask ({len(input_idx)} pixels)')
        axes[i, 1].axis('off')

        # Output mask (20%)
        output_mask = torch.zeros(32, 32)
        output_idx = sample['output_indices']
        y_out = output_idx // 32
        x_out = output_idx % 32
        output_mask[y_out, x_out] = 1
        axes[i, 2].imshow(output_mask.numpy(), cmap='Reds', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Output Mask ({len(output_idx)} pixels)')
        axes[i, 2].axis('off')

        # Combined mask (40%)
        combined_mask = input_mask + output_mask
        axes[i, 3].imshow(combined_mask.numpy(), cmap='Greens', vmin=0, vmax=2)
        axes[i, 3].set_title('Combined (Input=1, Output=1)')
        axes[i, 3].axis('off')

    plt.suptitle('Deterministic Instance-Specific Sparse Sampling\n(Same instance = Same masks)',
                 fontsize=14, y=0.98)
    plt.tight_layout()

    save_path = 'deterministic_sampling_verification.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SPARSE CIFAR-10 DATASET VERIFICATION")
    print("=" * 60)
    print("\nVerifying that:")
    print("1. Sampling is deterministic (same seed = same masks)")
    print("2. Input and output pixels are non-overlapping")
    print("3. Same instance always returns same masks")
    print("4. Exactly 40% of pixels used (20% input + 20% output)")
    print()

    results = []

    # Run all tests
    results.append(("Deterministic Sampling", test_deterministic_sampling()))
    results.append(("Non-Overlapping", test_non_overlapping()))
    results.append(("Consistency", test_consistency()))
    results.append(("Coverage", test_coverage()))

    # Create visualization
    try:
        visualize_sampling()
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n🎉 All tests passed!")
        print("\nThe dataset correctly implements:")
        print("  ✓ Instance-specific deterministic masking")
        print("  ✓ Non-overlapping 20% input + 20% output")
        print("  ✓ Consistent masks across multiple accesses")
    else:
        print("\n❌ Some tests failed!")
        exit(1)
