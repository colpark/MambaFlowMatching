#!/usr/bin/env python3
"""
Debug script to identify training issues

Tests:
1. Data generation
2. Model forward pass
3. Loss computation
4. Gradient flow
5. Parameter updates
"""
import sys
import os

# Add repository root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
synthetic_dir = os.path.dirname(script_dir)
repo_root = os.path.dirname(synthetic_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import torch.nn.functional as F
import numpy as np
from train_mamba_v1 import BaselineMAMBAFlow, conditional_flow
from synthetic_experiments.datasets import SinusoidalDataset

def test_data_generation():
    """Test 1: Can we generate synthetic data?"""
    print("\n" + "="*70)
    print("TEST 1: Data Generation")
    print("="*70)

    dataset = SinusoidalDataset(
        resolution=32,
        num_samples=10,
        complexity='simple',
        noise_level=0.0
    )

    train_coords, train_values, test_coords, test_values, full_data = dataset.get_train_test_split(
        train_sparsity=0.05,
        test_sparsity=0.05
    )

    print(f"✅ Dataset created")
    print(f"   Train coords: {train_coords.shape}")
    print(f"   Train values: {train_values.shape} | Range: [{train_values.min():.3f}, {train_values.max():.3f}]")
    print(f"   Test coords: {test_coords.shape}")
    print(f"   Test values: {test_values.shape} | Range: [{test_values.min():.3f}, {test_values.max():.3f}]")
    print(f"   Full data: {full_data.shape} | Range: [{full_data.min():.3f}, {full_data.max():.3f}]")

    return train_coords, train_values, test_coords, test_values, full_data


def test_model_forward(train_coords, train_values, test_coords):
    """Test 2: Can model do forward pass?"""
    print("\n" + "="*70)
    print("TEST 2: Model Forward Pass")
    print("="*70)

    model = BaselineMAMBAFlow(d_model=128, num_layers=4)

    # Take first sample
    train_coords_batch = train_coords[0:1]
    train_values_batch = train_values[0:1]
    test_coords_batch = test_coords[0:1]

    t = torch.rand(1)

    print(f"   Input shapes:")
    print(f"      Train coords: {train_coords_batch.shape}")
    print(f"      Train values: {train_values_batch.shape}")
    print(f"      Test coords: {test_coords_batch.shape}")
    print(f"      Time: {t.shape}, value: {t.item():.3f}")

    output = model(train_coords_batch, train_values_batch, test_coords_batch, t)

    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"   Output mean: {output.mean():.3f}, std: {output.std():.3f}")
    print(f"✅ Forward pass successful")

    return model


def test_loss_computation(model, train_coords, train_values, test_coords, test_values):
    """Test 3: Is loss computed correctly?"""
    print("\n" + "="*70)
    print("TEST 3: Loss Computation")
    print("="*70)

    # Take first sample
    train_coords_batch = train_coords[0:1]
    train_values_batch = train_values[0:1]
    test_coords_batch = test_coords[0:1]
    test_values_batch = test_values[0:1]

    t = torch.rand(1)

    # Flow matching
    noise = torch.randn_like(test_values_batch)
    z_t, v_t = conditional_flow(noise, test_values_batch, t)

    print(f"   Noise: mean={noise.mean():.3f}, std={noise.std():.3f}")
    print(f"   Target values: mean={test_values_batch.mean():.3f}, std={test_values_batch.std():.3f}")
    print(f"   Interpolated z_t: mean={z_t.mean():.3f}, std={z_t.std():.3f}")
    print(f"   Target velocity v_t: mean={v_t.mean():.3f}, std={v_t.std():.3f}")

    # Predict velocity
    pred_v = model(train_coords_batch, train_values_batch, test_coords_batch, t)

    print(f"   Predicted velocity: mean={pred_v.mean():.3f}, std={pred_v.std():.3f}")

    # Loss
    loss = F.mse_loss(pred_v, v_t)

    print(f"   Loss: {loss.item():.6f}")
    print(f"✅ Loss computation successful")

    return loss


def test_gradient_flow(model, train_coords, train_values, test_coords, test_values):
    """Test 4: Do gradients flow?"""
    print("\n" + "="*70)
    print("TEST 4: Gradient Flow")
    print("="*70)

    # Take first sample
    train_coords_batch = train_coords[0:1]
    train_values_batch = train_values[0:1]
    test_coords_batch = test_coords[0:1]
    test_values_batch = test_values[0:1]

    t = torch.rand(1)

    # Flow matching
    noise = torch.randn_like(test_values_batch)
    z_t, v_t = conditional_flow(noise, test_values_batch, t)

    # Predict velocity
    pred_v = model(train_coords_batch, train_values_batch, test_coords_batch, t)
    loss = F.mse_loss(pred_v, v_t)

    # Backward
    model.zero_grad()
    loss.backward()

    # Check gradients
    total_params = 0
    params_with_grad = 0
    grad_norms = []

    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None:
            params_with_grad += 1
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if len(grad_norms) <= 5:  # Print first 5
                print(f"   {name}: grad_norm={grad_norm:.6f}")

    print(f"   ...")
    print(f"   Parameters with gradients: {params_with_grad}/{total_params}")
    print(f"   Gradient norm stats:")
    print(f"      Mean: {np.mean(grad_norms):.6f}")
    print(f"      Std: {np.std(grad_norms):.6f}")
    print(f"      Max: {np.max(grad_norms):.6f}")
    print(f"      Min: {np.min(grad_norms):.6f}")

    if params_with_grad < total_params:
        print(f"⚠️  WARNING: {total_params - params_with_grad} parameters have no gradients!")
    else:
        print(f"✅ All parameters have gradients")

    return grad_norms


def test_parameter_updates(model, train_coords, train_values, test_coords, test_values):
    """Test 5: Do parameters actually update?"""
    print("\n" + "="*70)
    print("TEST 5: Parameter Updates")
    print("="*70)

    # Save initial parameters
    initial_params = {}
    for name, param in model.named_parameters():
        initial_params[name] = param.data.clone()

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Do one training step
    train_coords_batch = train_coords[0:1]
    train_values_batch = train_values[0:1]
    test_coords_batch = test_coords[0:1]
    test_values_batch = test_values[0:1]

    t = torch.rand(1)
    noise = torch.randn_like(test_values_batch)
    z_t, v_t = conditional_flow(noise, test_values_batch, t)

    pred_v = model(train_coords_batch, train_values_batch, test_coords_batch, t)
    loss = F.mse_loss(pred_v, v_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check if parameters changed
    num_changed = 0
    num_total = 0
    max_change = 0

    for name, param in model.named_parameters():
        num_total += 1
        change = (param.data - initial_params[name]).abs().max().item()
        if change > 1e-10:
            num_changed += 1
            max_change = max(max_change, change)
            if num_changed <= 5:  # Print first 5
                print(f"   {name}: max_change={change:.6e}")

    print(f"   ...")
    print(f"   Parameters changed: {num_changed}/{num_total}")
    print(f"   Maximum parameter change: {max_change:.6e}")

    if num_changed == num_total:
        print(f"✅ All parameters updated")
    else:
        print(f"⚠️  WARNING: {num_total - num_changed} parameters did not change!")

    return num_changed == num_total


def test_training_convergence():
    """Test 6: Does training actually converge?"""
    print("\n" + "="*70)
    print("TEST 6: Training Convergence (10 epochs)")
    print("="*70)

    # Create small dataset
    dataset = SinusoidalDataset(
        resolution=32,
        num_samples=50,
        complexity='simple',
        noise_level=0.0
    )

    train_coords, train_values, test_coords, test_values, full_data = dataset.get_train_test_split(
        train_sparsity=0.05,
        test_sparsity=0.05
    )

    # Create model
    model = BaselineMAMBAFlow(d_model=128, num_layers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []

    for epoch in range(10):
        epoch_loss = 0
        num_batches = 0

        for i in range(0, 50, 5):  # batch_size=5
            # Get batch
            train_coords_batch = train_coords[i:i+5]
            train_values_batch = train_values[i:i+5]
            test_coords_batch = test_coords[i:i+5]
            test_values_batch = test_values[i:i+5]

            # Sample timestep
            t = torch.rand(len(train_coords_batch))

            # Flow matching
            noise = torch.randn_like(test_values_batch)
            z_t, v_t = conditional_flow(noise, test_values_batch, t)

            # Predict velocity
            pred_v = model(train_coords_batch, train_values_batch, test_coords_batch, t)

            # Loss
            loss = F.mse_loss(pred_v, v_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        print(f"   Epoch {epoch+1}/10: Loss = {avg_loss:.6f}")

    # Check if loss is decreasing
    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = (initial_loss - final_loss) / initial_loss * 100

    print(f"\n   Initial loss: {initial_loss:.6f}")
    print(f"   Final loss: {final_loss:.6f}")
    print(f"   Reduction: {reduction:.2f}%")

    if final_loss < initial_loss * 0.8:
        print(f"✅ Training converged (>20% loss reduction)")
        return True
    else:
        print(f"⚠️  WARNING: Loss did not decrease significantly!")
        return False


def main():
    print("="*70)
    print("MAMBA V1 Training Diagnostic Tool")
    print("="*70)

    # Run all tests
    train_coords, train_values, test_coords, test_values, full_data = test_data_generation()
    model = test_model_forward(train_coords, train_values, test_coords)
    test_loss_computation(model, train_coords, train_values, test_coords, test_values)
    test_gradient_flow(model, train_coords, train_values, test_coords, test_values)
    test_parameter_updates(model, train_coords, train_values, test_coords, test_values)
    converged = test_training_convergence()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    if converged:
        print("✅ All tests passed - training should work!")
    else:
        print("⚠️  Some tests failed - investigate issues above")
    print("="*70)


if __name__ == '__main__':
    main()
