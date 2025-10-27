#!/usr/bin/env python3
"""
Train Improved Transformer with Selected Techniques

Supports all 10 improvement techniques through command-line flags
"""
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
synthetic_dir = os.path.dirname(script_dir)
repo_root = os.path.dirname(synthetic_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import torch.nn.functional as F
import argparse
import json
import time
from pathlib import Path

from synthetic_experiments.datasets import SinusoidalDataset
from synthetic_experiments.improvements.improved_transformer import (
    build_model_from_techniques,
    sample_heun,
)
from synthetic_experiments.improvements.techniques import (
    RectifiedFlowMixin,
    ConsistencyLossMixin,
    FrequencyPerceptualLoss,
    SelfConditioningMixin,
    EMA,
    NoiseSchedule,
    WarmupCosineScheduler,
    apply_gradient_clipping,
    TECHNIQUES,
)


def conditional_flow(x0, x1, t, schedule='linear'):
    """
    Conditional flow matching with optional noise schedule

    Args:
        x0: noise (B, N, 1)
        x1: target (B, N, 1)
        t: timestep (B,)
        schedule: 'linear', 'cosine', or 'sigmoid'
    """
    if schedule != 'linear':
        z_t, v_t, t_scheduled = NoiseSchedule.apply_schedule(x0, x1, t, schedule)
        return z_t, v_t
    else:
        # Standard linear
        t_expanded = t.view(-1, 1, 1)
        z_t = t_expanded * x1 + (1 - t_expanded) * x0
        v_t = x1 - x0
        return z_t, v_t


def train_epoch(
    model,
    dataset,
    optimizer,
    device,
    technique_ids,
    ema=None,
    perceptual_loss=None,
    epoch=0,
):
    """
    Train for one epoch with selected techniques

    Technique handling:
    - 1 (Rectified Flow): Applied in loss computation
    - 2 (Multi-scale PE): Built into model
    - 3 (Consistency Loss): Added to total loss
    - 4 (Perceptual Loss): Replaces standard MSE
    - 5 (AdaLN): Built into model
    - 6 (Self-conditioning): Applied in forward pass
    - 7 (EMA): Updated after each step
    - 8 (Noise Schedule): Applied in conditional_flow
    - 9 (Grad Clip + Warmup): Applied after backward
    - 10 (Stochastic Depth): Built into model
    """
    model.train()

    # Get data
    train_coords, train_values, test_coords, test_values, full_data = dataset.get_train_test_split(
        train_sparsity=0.05,
        test_sparsity=0.05
    )

    # Training config
    batch_size = 32
    num_samples = len(train_coords)

    total_loss = 0.0
    num_batches = 0

    # Technique flags
    use_rectified_flow = 1 in technique_ids
    use_consistency_loss = 3 in technique_ids
    use_perceptual_loss = 4 in technique_ids
    use_self_conditioning = 6 in technique_ids
    use_ema = 7 in technique_ids
    use_noise_schedule = 8 in technique_ids
    use_grad_clip = 9 in technique_ids

    # Get schedule type
    schedule = 'cosine' if use_noise_schedule else 'linear'

    # CRITICAL FIX: Create full coordinate grid for random query sampling
    H, W = dataset.resolution, dataset.resolution
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing='ij'
    )
    all_coords = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1).to(device)
    N_total = H * W  # 1024 for 32x32

    for i in range(0, num_samples, batch_size):
        # Get batch
        train_coords_batch = train_coords[i:i+batch_size].to(device)
        train_values_batch = train_values[i:i+batch_size].to(device)
        full_data_batch = full_data[i:i+batch_size].to(device)

        actual_batch_size = len(train_coords_batch)

        # CRITICAL FIX: Sample random query points from full field
        # Instead of fixed 51 test pixels, sample 200 random pixels each iteration
        num_query = 200  # More than 51, less than 1024 for efficiency
        query_indices = torch.randperm(N_total, device=device)[:num_query]

        # Get query coordinates and values
        test_coords_batch = all_coords[query_indices].unsqueeze(0).expand(actual_batch_size, -1, -1)
        test_values_batch = full_data_batch.view(actual_batch_size, N_total, 1)[:, query_indices, :]

        # Sample timestep
        t = torch.rand(actual_batch_size, device=device)

        # Flow matching
        noise = torch.randn_like(test_values_batch)

        if use_rectified_flow:
            # Technique 1: Rectified flow (for now, same as linear)
            # TODO: Implement full reflow in separate training phase
            z_t, v_t = conditional_flow(noise, test_values_batch, t, schedule)
        else:
            z_t, v_t = conditional_flow(noise, test_values_batch, t, schedule)

        # Predict velocity
        if use_self_conditioning:
            # Technique 6: Self-conditioning (two-pass forward)
            pred_v = SelfConditioningMixin.self_conditioning_forward(
                model, train_coords_batch, train_values_batch,
                test_coords_batch, t, z_t, use_sc_prob=0.5
            )
        else:
            pred_v = model(train_coords_batch, train_values_batch,
                          test_coords_batch, t, query_values=z_t)

        # Loss computation
        if use_perceptual_loss:
            # Technique 4: Perceptual loss (frequency domain)
            loss = perceptual_loss(pred_v, v_t)
        else:
            # Standard MSE
            loss = F.mse_loss(pred_v, v_t)

        # Technique 3: Consistency loss
        if use_consistency_loss:
            consistency = ConsistencyLossMixin.consistency_loss(
                model, noise, test_values_batch,
                train_coords_batch, train_values_batch, test_coords_batch,
                device, lambda_consistency=0.1
            )
            loss = loss + consistency

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Technique 9: Gradient clipping
        if use_grad_clip:
            apply_gradient_clipping(model, max_norm=1.0)

        optimizer.step()

        # Technique 7: EMA update
        if use_ema and ema is not None:
            ema.update()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, dataset, device, use_ema=False, ema=None):
    """
    Evaluate model on test set

    Returns PSNR on full field reconstruction
    """
    if use_ema and ema is not None:
        ema.apply_shadow()

    model.eval()

    train_coords, train_values, test_coords, test_values, full_data = dataset.get_train_test_split(
        train_sparsity=0.05,
        test_sparsity=0.05
    )

    # Evaluate on first 50 samples
    num_eval = min(50, len(full_data))
    train_coords = train_coords[:num_eval].to(device)
    train_values = train_values[:num_eval].to(device)
    full_data = full_data[:num_eval].to(device)

    # Create query coordinates for full field
    H, W = dataset.resolution, dataset.resolution
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing='ij'
    )
    query_coords = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)
    query_coords = query_coords.unsqueeze(0).expand(num_eval, -1, -1).to(device)

    psnrs = []

    with torch.no_grad():
        for idx in range(num_eval):
            train_coords_sample = train_coords[idx:idx+1]
            train_values_sample = train_values[idx:idx+1]
            query_coords_sample = query_coords[idx:idx+1]
            target = full_data[idx:idx+1]

            # Sample
            pred = sample_heun(model, train_coords_sample, train_values_sample,
                             query_coords_sample, num_steps=50, device=device)

            # Reshape
            pred = pred.reshape(1, 1, H, W)

            # PSNR
            mse = F.mse_loss(pred, target).item()
            if mse > 0:
                psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(torch.tensor(mse))
                psnrs.append(psnr.item())

    if use_ema and ema is not None:
        ema.restore()

    avg_psnr = sum(psnrs) / len(psnrs) if psnrs else 0.0
    return avg_psnr


def main():
    parser = argparse.ArgumentParser()

    # Data params
    parser.add_argument('--complexity', type=str, default='simple')
    parser.add_argument('--resolution', type=int, default=32)
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--noise_level', type=float, default=0.0)

    # Model params
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dim_feedforward', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training params
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='auto')

    # Techniques (comma-separated list of IDs)
    parser.add_argument('--techniques', type=str, default='',
                       help='Comma-separated technique IDs (e.g., "1,2,3")')

    # Version and save
    parser.add_argument('--version', type=int, default=3)
    parser.add_argument('--save_dir', type=str, default='checkpoints_improvements')
    parser.add_argument('--gpu_id', type=int, default=0)

    args = parser.parse_args()

    # Parse techniques
    if args.techniques:
        technique_ids = [int(x.strip()) for x in args.techniques.split(',')]
    else:
        technique_ids = []

    print(f"\n{'='*70}")
    print(f"Training v{args.version} with Techniques: {technique_ids}")
    print(f"{'='*70}\n")

    # Print technique descriptions
    for tid in technique_ids:
        if tid in TECHNIQUES:
            print(f"  ✓ {tid}: {TECHNIQUES[tid]['name']}")
    print()

    # Device
    if args.device == 'auto':
        device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Device: {device}\n")

    # Create dataset
    dataset = SinusoidalDataset(
        resolution=args.resolution,
        num_samples=args.num_samples,
        complexity=args.complexity,
        noise_level=args.noise_level
    )

    # Build model
    model = build_model_from_techniques(
        technique_ids,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Technique-specific components
    ema = None
    perceptual_loss = None
    scheduler = None

    if 7 in technique_ids:
        # EMA
        ema = EMA(model, decay=0.9999)
        print("✓ EMA enabled (decay=0.9999)")

    if 4 in technique_ids:
        # Perceptual loss
        perceptual_loss = FrequencyPerceptualLoss(lambda_freq=0.5).to(device)
        print("✓ Perceptual loss enabled (lambda_freq=0.5)")

    if 9 in technique_ids:
        # Warmup + cosine scheduler
        total_steps = args.epochs * (args.num_samples // 32)
        warmup_steps = 1000
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_steps, total_steps, args.lr, min_lr=1e-6
        )
        print(f"✓ Warmup scheduler enabled (warmup={warmup_steps}, total={total_steps})")

    print()

    # Training loop
    best_psnr = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_loss = train_epoch(
            model, dataset, optimizer, device,
            technique_ids, ema, perceptual_loss, epoch
        )

        # Update scheduler
        if scheduler is not None:
            lr = scheduler.step()
        else:
            lr = args.lr

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            psnr = evaluate(model, dataset, device, use_ema=(7 in technique_ids), ema=ema)

            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss:.6f} | "
                  f"PSNR: {psnr:.2f} dB | LR: {lr:.6f}")

            if psnr > best_psnr:
                best_psnr = psnr

                # Save checkpoint
                save_path = Path(args.save_dir) / f'v{args.version}'
                save_path.mkdir(parents=True, exist_ok=True)

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'psnr': psnr,
                    'techniques': technique_ids,
                    'version': args.version,
                }

                if ema is not None:
                    checkpoint['ema_shadow'] = ema.shadow

                torch.save(checkpoint, save_path / 'best_model.pth')
        else:
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss:.6f} | LR: {lr:.6f}")

        sys.stdout.flush()

    # Final evaluation
    final_psnr = evaluate(model, dataset, device, use_ema=(7 in technique_ids), ema=ema)
    elapsed = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"Training Complete - v{args.version}")
    print(f"{'='*70}")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Final PSNR: {final_psnr:.2f} dB")
    print(f"Training time: {elapsed/60:.1f} minutes")
    print(f"Techniques used: {technique_ids}")
    print(f"{'='*70}\n")

    # Save final results
    results = {
        'version': args.version,
        'techniques': technique_ids,
        'best_psnr': best_psnr,
        'final_psnr': final_psnr,
        'training_time': elapsed,
        'complexity': args.complexity,
    }

    results_path = Path(args.save_dir) / f'v{args.version}' / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"💾 Saved results to: {results_path}\n")


if __name__ == '__main__':
    main()
