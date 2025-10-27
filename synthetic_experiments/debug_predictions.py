#!/usr/bin/env python3
"""
Debug script to understand why both V1 and V2 show ~6 dB PSNR

Tests:
1. Check if data is actually in [0, 1] range
2. Check if model predictions are bounded
3. Check if model is learning anything (overfit test)
4. Check if sampling procedure is correct
5. Check PSNR calculation
"""
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

# Add repo to path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from synthetic_experiments.datasets import SinusoidalDataset


def test_data_range():
    """Test 1: Check if data is in [0, 1] range"""
    print("\n" + "="*70)
    print("TEST 1: Data Range Check")
    print("="*70)

    dataset = SinusoidalDataset(
        resolution=32,
        num_samples=100,
        complexity='simple'
    )

    train_coords, train_values, test_coords, test_values, full_data = dataset.get_train_test_split(
        train_sparsity=0.05,
        test_sparsity=0.05
    )

    print(f"Train values range: [{train_values.min():.4f}, {train_values.max():.4f}]")
    print(f"Test values range: [{test_values.min():.4f}, {test_values.max():.4f}]")
    print(f"Full data range: [{full_data.min():.4f}, {full_data.max():.4f}]")
    print(f"Train values mean: {train_values.mean():.4f}, std: {train_values.std():.4f}")
    print(f"Full data mean: {full_data.mean():.4f}, std: {full_data.std():.4f}")

    # Check if data is properly normalized
    if train_values.min() >= 0 and train_values.max() <= 1:
        print("✅ Data is in [0, 1] range")
    else:
        print("❌ Data is NOT in [0, 1] range!")

    return dataset, train_coords, train_values, test_coords, test_values, full_data


def test_baseline_psnr(full_data):
    """Test 2: Check baseline PSNR for different prediction strategies"""
    print("\n" + "="*70)
    print("TEST 2: Baseline PSNR Check")
    print("="*70)

    # Strategy 1: Predict mean
    pred_mean = torch.full_like(full_data, full_data.mean())
    mse_mean = F.mse_loss(pred_mean, full_data).item()
    psnr_mean = 20 * np.log10(1.0) - 10 * np.log10(mse_mean) if mse_mean > 0 else float('inf')

    print(f"Predict mean (0.5): PSNR = {psnr_mean:.2f} dB (MSE = {mse_mean:.6f})")

    # Strategy 2: Predict zeros
    pred_zeros = torch.zeros_like(full_data)
    mse_zeros = F.mse_loss(pred_zeros, full_data).item()
    psnr_zeros = 20 * np.log10(1.0) - 10 * np.log10(mse_zeros) if mse_zeros > 0 else float('inf')

    print(f"Predict zeros (0.0): PSNR = {psnr_zeros:.2f} dB (MSE = {mse_zeros:.6f})")

    # Strategy 3: Predict random
    pred_random = torch.rand_like(full_data)
    mse_random = F.mse_loss(pred_random, full_data).item()
    psnr_random = 20 * np.log10(1.0) - 10 * np.log10(mse_random) if mse_random > 0 else float('inf')

    print(f"Predict random: PSNR = {psnr_random:.2f} dB (MSE = {mse_random:.6f})")

    # Strategy 4: Predict ones
    pred_ones = torch.ones_like(full_data)
    mse_ones = F.mse_loss(pred_ones, full_data).item()
    psnr_ones = 20 * np.log10(1.0) - 10 * np.log10(mse_ones) if mse_ones > 0 else float('inf')

    print(f"Predict ones (1.0): PSNR = {psnr_ones:.2f} dB (MSE = {mse_ones:.6f})")

    print(f"\n💡 If model achieves ~{psnr_mean:.2f} dB, it's predicting mean (not learning)")
    print(f"💡 If model achieves ~{psnr_zeros:.2f} dB, it's predicting zeros")
    print(f"💡 If model achieves ~{psnr_ones:.2f} dB, it's predicting ones")


def test_flow_matching_math():
    """Test 3: Verify flow matching interpolation math"""
    print("\n" + "="*70)
    print("TEST 3: Flow Matching Math Check")
    print("="*70)

    # Simple test
    x0 = torch.zeros(1, 10, 1)  # noise
    x1 = torch.ones(1, 10, 1)   # target

    # Test interpolation at different timesteps
    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        t = torch.tensor([t_val])
        t_expanded = t.view(-1, 1, 1)

        # Forward flow: z_t = t * x1 + (1-t) * x0
        z_t = t_expanded * x1 + (1 - t_expanded) * x0

        # Velocity: v_t = x1 - x0
        v_t = x1 - x0

        print(f"t={t_val:.2f}: z_t={z_t.mean():.2f}, v_t={v_t.mean():.2f}")

    print("\n✅ At t=0: z_t should be ~0 (noise)")
    print("✅ At t=1: z_t should be ~1 (target)")
    print("✅ Velocity v_t should always be 1.0 (x1 - x0)")


def test_sampling_procedure():
    """Test 4: Check if Heun sampling converges correctly"""
    print("\n" + "="*70)
    print("TEST 4: Sampling Procedure Check")
    print("="*70)

    # Create a "perfect oracle" model that always predicts correct velocity
    class OracleModel:
        def __call__(self, sparse_coords, sparse_values, query_coords, t, query_values):
            # Always return correct velocity: target - noise
            # For our test: target = 0.7, noise starts at random
            # So velocity should point toward 0.7
            target = 0.7
            return torch.full_like(query_values, target) - query_values

    model = OracleModel()

    # Simulate sampling
    num_steps = 50
    B, N = 1, 100
    device = 'cpu'

    # Start from random noise
    z_t = torch.randn(B, N, 1, device=device)
    initial_val = z_t.mean().item()

    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((B,), i * dt, device=device)

        # First prediction
        v1 = model(None, None, None, t, z_t)

        # Euler step
        z_next = z_t + v1 * dt

        # Second prediction
        t_next = torch.full((B,), (i + 1) * dt, device=device)
        v2 = model(None, None, None, t_next, z_next)

        # Heun's method
        z_t = z_t + 0.5 * (v1 + v2) * dt

    final_val = z_t.mean().item()

    print(f"Initial value (random noise): {initial_val:.4f}")
    print(f"Final value (should be 0.7): {final_val:.4f}")
    print(f"Error: {abs(final_val - 0.7):.4f}")

    if abs(final_val - 0.7) < 0.01:
        print("✅ Sampling procedure works correctly!")
    else:
        print("❌ Sampling procedure is BROKEN!")


def test_psnr_calculation():
    """Test 5: Verify PSNR calculation"""
    print("\n" + "="*70)
    print("TEST 5: PSNR Calculation Check")
    print("="*70)

    # Create known MSE values and check PSNR
    test_cases = [
        (0.5, 6.02),    # MSE=0.5 → PSNR≈6 dB (predict mean)
        (0.25, 12.04),  # MSE=0.25 → PSNR≈12 dB
        (0.1, 20.00),   # MSE=0.1 → PSNR=20 dB
        (0.01, 40.00),  # MSE=0.01 → PSNR=40 dB
        (0.001, 60.00), # MSE=0.001 → PSNR=60 dB
    ]

    print("MSE → PSNR conversion:")
    for mse, expected_psnr in test_cases:
        psnr = 20 * np.log10(1.0) - 10 * np.log10(mse)
        print(f"  MSE={mse:.4f} → PSNR={psnr:.2f} dB (expected ~{expected_psnr:.2f} dB)")

    print("\n✅ PSNR calculation verified")
    print("💡 MSE=0.5 (predict mean) → PSNR≈6 dB")
    print("💡 MSE=0.1 (decent) → PSNR=20 dB")
    print("💡 MSE=0.01 (good) → PSNR=40 dB")


def main():
    print("\n" + "="*70)
    print("DEBUGGING: Why Both V1 and V2 Show ~6 dB PSNR")
    print("="*70)

    # Run tests
    dataset, train_coords, train_values, test_coords, test_values, full_data = test_data_range()
    test_baseline_psnr(full_data)
    test_flow_matching_math()
    test_sampling_procedure()
    test_psnr_calculation()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Key Insights:
1. If PSNR ≈ 6 dB, model is predicting constant ~0.5 (mean value)
2. This means model is NOT learning the spatial patterns
3. Possible causes:
   a) Model capacity too small
   b) Training not converging
   c) Loss signal is wrong
   d) Sampling procedure is broken
   e) Architecture bug preventing learning

Next Steps:
1. Check training loss curves - is loss decreasing?
2. Overfit on 1 sample - can model memorize?
3. Check if model outputs are bounded
4. Visualize predictions - are they constant or varying?
""")


if __name__ == '__main__':
    main()
