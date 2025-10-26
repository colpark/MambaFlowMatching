"""
Improvement Techniques for Transformer Flow Matching

Each technique is a modular component that can be added to the base model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any


# ============================================================================
# Technique 1: Rectified Flow (Optimal Transport)
# ============================================================================

class RectifiedFlowMixin:
    """
    Replaces linear interpolation with optimal transport path.
    Learns straighter paths in probability space.
    """
    @staticmethod
    def rectified_flow(x0, x1, t):
        """
        Optimal transport interpolation (straight path)

        Standard: z_t = t*x1 + (1-t)*x0, v_t = x1 - x0
        Rectified: Learn to minimize path curvature
        """
        # For first iteration, use linear (same as standard)
        z_t = t * x1 + (1 - t) * x0
        v_t = x1 - x0
        return z_t, v_t

    @staticmethod
    def reflow_loss(model, x0, x1, sparse_coords, sparse_values, query_coords, device):
        """
        Reflow procedure: sample from model, then train on sampled paths
        This straightens the probability flow over iterations
        """
        # Sample a path from current model
        with torch.no_grad():
            # Start from x0 (noise)
            z_0 = x0

            # Integrate flow with Euler method
            num_steps = 10
            dt = 1.0 / num_steps
            z_t = z_0

            for i in range(num_steps):
                t = torch.full((z_0.shape[0],), i * dt, device=device)
                v = model(sparse_coords, sparse_values, query_coords, t, query_values=z_t)
                z_t = z_t + v * dt

            # z_t is now sampled x1'
            x1_sampled = z_t

        # Train on the sampled path (this makes it straighter)
        t = torch.rand(x0.shape[0], device=device)
        t_expanded = t.view(-1, 1, 1)

        z_t = t_expanded * x1_sampled + (1 - t_expanded) * x0
        v_target = x1_sampled - x0

        return z_t, v_target


# ============================================================================
# Technique 2: Multi-Scale Positional Encoding
# ============================================================================

class MultiScaleFourierFeatures(nn.Module):
    """
    Multi-scale Fourier encoding for better frequency representation
    Uses logarithmically-spaced frequency bands
    """
    def __init__(self, input_dim=2, num_frequencies=16, num_scales=6):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.num_scales = num_scales

        # Create frequency bands: [1, 2, 4, 8, 16, 32] × base_frequencies
        self.register_buffer('B', torch.randn(input_dim, num_frequencies) * 10.0)
        self.register_buffer('scale_factors', torch.pow(2.0, torch.arange(num_scales, dtype=torch.float32)))

    def forward(self, coords):
        """
        Args:
            coords: (B, N, input_dim)
        Returns:
            features: (B, N, input_dim + 2*num_frequencies*num_scales)
        """
        # Original coordinates
        features = [coords]

        # Multi-scale Fourier features
        for scale in self.scale_factors:
            scaled_B = self.B * scale
            proj = 2 * np.pi * coords @ scaled_B  # (B, N, num_frequencies)
            features.append(torch.sin(proj))
            features.append(torch.cos(proj))

        return torch.cat(features, dim=-1)

    @property
    def output_dim(self):
        return self.B.shape[0] + 2 * self.num_frequencies * self.num_scales


# ============================================================================
# Technique 3: Consistency Loss
# ============================================================================

class ConsistencyLossMixin:
    """
    Enforce temporal consistency in flow predictions
    """
    @staticmethod
    def consistency_loss(model, x0, x1, sparse_coords, sparse_values, query_coords, device, lambda_consistency=0.1):
        """
        Sample two timesteps and enforce prediction consistency
        """
        # Sample two timesteps
        t1 = torch.rand(x0.shape[0], device=device)
        t2 = torch.rand(x0.shape[0], device=device)

        t1_exp = t1.view(-1, 1, 1)
        t2_exp = t2.view(-1, 1, 1)

        # Interpolate at both timesteps
        z_t1 = t1_exp * x1 + (1 - t1_exp) * x0
        z_t2 = t2_exp * x1 + (1 - t2_exp) * x0

        # Get predictions
        v_t1 = model(sparse_coords, sparse_values, query_coords, t1, query_values=z_t1)
        v_t2 = model(sparse_coords, sparse_values, query_coords, t2, query_values=z_t2)

        # Consistency loss: predictions should be similar (weighted by time difference)
        time_diff = torch.abs(t1 - t2).view(-1, 1, 1)
        consistency = F.mse_loss(v_t1, v_t2, reduction='none') * (1.0 - time_diff)

        return lambda_consistency * consistency.mean()


# ============================================================================
# Technique 4: Perceptual Loss (Frequency Domain)
# ============================================================================

class FrequencyPerceptualLoss(nn.Module):
    """
    Loss in frequency domain to preserve spectral content
    """
    def __init__(self, lambda_freq=0.5):
        super().__init__()
        self.lambda_freq = lambda_freq

    def forward(self, pred, target):
        """
        Args:
            pred: (B, N, 1) predicted values
            target: (B, N, 1) target values
        """
        # Spatial MSE
        mse_loss = F.mse_loss(pred, target)

        # Reshape to 2D for FFT (assume square)
        B, N, C = pred.shape
        H = W = int(np.sqrt(N))

        if H * W != N:
            # Not square, skip frequency loss
            return mse_loss

        pred_2d = pred.reshape(B, C, H, W)
        target_2d = target.reshape(B, C, H, W)

        # FFT
        pred_fft = torch.fft.fft2(pred_2d)
        target_fft = torch.fft.fft2(target_2d)

        # Frequency domain MSE
        freq_loss = F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))

        # Combined loss
        return mse_loss + self.lambda_freq * freq_loss


# ============================================================================
# Technique 5: Adaptive Layer Normalization (AdaLN)
# ============================================================================

class AdaptiveLayerNorm(nn.Module):
    """
    Time-conditioned layer normalization
    scale and shift depend on timestep t
    """
    def __init__(self, d_model, time_embed_dim=128):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)

        # Learn time-dependent scale and shift
        self.scale_shift = nn.Sequential(
            nn.Linear(time_embed_dim, d_model * 2),
            nn.SiLU(),
        )

    def forward(self, x, time_embed):
        """
        Args:
            x: (B, N, d_model)
            time_embed: (B, time_embed_dim)
        """
        # Normalize
        x_norm = self.norm(x)

        # Get time-dependent scale and shift
        scale_shift = self.scale_shift(time_embed)  # (B, 2*d_model)
        scale, shift = scale_shift.chunk(2, dim=-1)  # Each (B, d_model)

        # Apply: scale * norm(x) + shift
        return x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ============================================================================
# Technique 6: Self-Conditioning
# ============================================================================

class SelfConditioningMixin:
    """
    Feed model's own previous prediction as input
    """
    @staticmethod
    def self_conditioning_forward(model, sparse_coords, sparse_values, query_coords, t, query_values, use_sc_prob=0.5):
        """
        Two-pass forward with self-conditioning

        Returns:
            pred: Final prediction to use for loss
            pred_for_sc: Prediction to use as self-conditioning in next iteration
        """
        # 50% of time, use self-conditioning
        use_sc = torch.rand(1).item() < use_sc_prob

        if use_sc:
            # First pass (no self-conditioning)
            with torch.no_grad():
                pred_sc = model(sparse_coords, sparse_values, query_coords, t, query_values=query_values)

            # Second pass (with self-conditioning)
            # Concatenate previous prediction to query_values
            query_values_sc = torch.cat([query_values, pred_sc.detach()], dim=-1)
            pred = model(sparse_coords, sparse_values, query_coords, t, query_values=query_values_sc)
        else:
            # No self-conditioning
            pred = model(sparse_coords, sparse_values, query_coords, t, query_values=query_values)

        return pred


# ============================================================================
# Technique 7: Exponential Moving Average (EMA)
# ============================================================================

class EMA:
    """
    Exponential Moving Average of model parameters
    """
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply shadow parameters to model (for evaluation)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ============================================================================
# Technique 8: Noise Schedule Optimization
# ============================================================================

class NoiseSchedule:
    """
    Different noise schedules for flow matching
    """
    @staticmethod
    def linear(t):
        """Standard linear schedule"""
        return t

    @staticmethod
    def cosine(t):
        """Cosine schedule (smoother)"""
        return 1.0 - torch.cos(0.5 * np.pi * t)

    @staticmethod
    def sigmoid(t, sharpness=2.0):
        """Sigmoid schedule (more time near 0 and 1)"""
        return torch.sigmoid(sharpness * (2 * t - 1)) / torch.sigmoid(torch.tensor(sharpness))

    @staticmethod
    def apply_schedule(x0, x1, t, schedule='cosine'):
        """
        Apply noise schedule to interpolation
        """
        if schedule == 'linear':
            t_scheduled = NoiseSchedule.linear(t)
        elif schedule == 'cosine':
            t_scheduled = NoiseSchedule.cosine(t)
        elif schedule == 'sigmoid':
            t_scheduled = NoiseSchedule.sigmoid(t)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        t_exp = t_scheduled.view(-1, 1, 1) if len(t.shape) == 1 else t_scheduled
        z_t = t_exp * x1 + (1 - t_exp) * x0
        v_t = x1 - x0

        return z_t, v_t, t_scheduled


# ============================================================================
# Technique 9: Gradient Clipping + Warmup
# ============================================================================

class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup and cosine decay
    """
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        """Update learning rate"""
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


def apply_gradient_clipping(model, max_norm=1.0):
    """Clip gradients to prevent explosion"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


# ============================================================================
# Technique 10: Stochastic Depth (DropPath)
# ============================================================================

class DropPath(nn.Module):
    """
    Drop entire residual paths during training (stochastic depth)
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """
        Args:
            x: input tensor
        Returns:
            x with paths dropped during training
        """
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, ...)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize

        return x.div(keep_prob) * random_tensor


def get_drop_path_schedule(num_layers, max_drop_prob=0.1):
    """
    Linear schedule: deeper layers have higher drop probability
    """
    return [i / (num_layers - 1) * max_drop_prob for i in range(num_layers)]


# ============================================================================
# Technique Configuration
# ============================================================================

TECHNIQUES = {
    1: {
        'name': 'Rectified Flow',
        'class': 'RectifiedFlowMixin',
        'params': {},
        'projected_gain': 6.5,
    },
    2: {
        'name': 'Multi-Scale Positional Encoding',
        'class': 'MultiScaleFourierFeatures',
        'params': {'num_frequencies': 16, 'num_scales': 6},
        'projected_gain': 5.5,
    },
    3: {
        'name': 'Consistency Loss',
        'class': 'ConsistencyLossMixin',
        'params': {'lambda_consistency': 0.1},
        'projected_gain': 4.5,
    },
    4: {
        'name': 'Perceptual Loss',
        'class': 'FrequencyPerceptualLoss',
        'params': {'lambda_freq': 0.5},
        'projected_gain': 4.0,
    },
    5: {
        'name': 'Adaptive LayerNorm',
        'class': 'AdaptiveLayerNorm',
        'params': {'time_embed_dim': 128},
        'projected_gain': 3.5,
    },
    6: {
        'name': 'Self-Conditioning',
        'class': 'SelfConditioningMixin',
        'params': {'use_sc_prob': 0.5},
        'projected_gain': 3.0,
    },
    7: {
        'name': 'EMA',
        'class': 'EMA',
        'params': {'decay': 0.9999},
        'projected_gain': 2.0,
    },
    8: {
        'name': 'Noise Schedule',
        'class': 'NoiseSchedule',
        'params': {'schedule': 'cosine'},
        'projected_gain': 2.0,
    },
    9: {
        'name': 'Gradient Clipping + Warmup',
        'class': 'WarmupCosineScheduler',
        'params': {'warmup_steps': 1000, 'max_norm': 1.0},
        'projected_gain': 1.5,
    },
    10: {
        'name': 'Stochastic Depth',
        'class': 'DropPath',
        'params': {'max_drop_prob': 0.1},
        'projected_gain': 1.5,
    },
}
