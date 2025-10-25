"""
Evaluation metrics for sparse image reconstruction

Unified metrics across all diffusion approaches for fair comparison
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim_skimage
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage


def psnr(pred, target, data_range=1.0):
    """
    Peak Signal-to-Noise Ratio

    Args:
        pred: (B, C, H, W) or (B, H, W, C) predicted images
        target: (B, C, H, W) or (B, H, W, C) target images
        data_range: Data range (1.0 for [0,1], 255 for [0,255])

    Returns:
        psnr: float, averaged over batch
    """
    if pred.dim() == 4:
        # Assume (B, C, H, W), convert to (B, H, W, C) for scikit-image
        pred = pred.permute(0, 2, 3, 1).cpu().numpy()
        target = target.permute(0, 2, 3, 1).cpu().numpy()
    else:
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

    psnr_vals = []
    for i in range(pred.shape[0]):
        psnr_val = psnr_skimage(target[i], pred[i], data_range=data_range)
        psnr_vals.append(psnr_val)

    return np.mean(psnr_vals)


def ssim(pred, target, data_range=1.0):
    """
    Structural Similarity Index

    Args:
        pred: (B, C, H, W) or (B, H, W, C) predicted images
        target: (B, C, H, W) or (B, H, W, C) target images
        data_range: Data range (1.0 for [0,1], 255 for [0,255])

    Returns:
        ssim: float, averaged over batch
    """
    if pred.dim() == 4:
        # Assume (B, C, H, W), convert to (B, H, W, C)
        pred = pred.permute(0, 2, 3, 1).cpu().numpy()
        target = target.permute(0, 2, 3, 1).cpu().numpy()
    else:
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

    ssim_vals = []
    for i in range(pred.shape[0]):
        ssim_val = ssim_skimage(
            target[i], pred[i],
            data_range=data_range,
            channel_axis=2 if pred.ndim == 4 else None
        )
        ssim_vals.append(ssim_val)

    return np.mean(ssim_vals)


def mse_on_pixels(pred_values, target_values):
    """
    Mean Squared Error on specific pixels

    Args:
        pred_values: (B, N, C) predicted pixel values
        target_values: (B, N, C) target pixel values

    Returns:
        mse: float
    """
    return F.mse_loss(pred_values, target_values).item()


def mae_on_pixels(pred_values, target_values):
    """
    Mean Absolute Error on specific pixels

    Args:
        pred_values: (B, N, C) predicted pixel values
        target_values: (B, N, C) target pixel values

    Returns:
        mae: float
    """
    return F.l1_loss(pred_values, target_values).item()


def reconstruct_full_image(pred_values, indices, image_size=32, n_channels=3):
    """
    Reconstruct full image from predicted sparse pixels

    Args:
        pred_values: (B, N, C) predicted pixel values
        indices: (B, N) or (N,) flat pixel indices
        image_size: Image height/width
        n_channels: Number of channels (3 for RGB)

    Returns:
        images: (B, C, H, W) reconstructed images (zeros for unpredicted pixels)
    """
    if pred_values.dim() == 2:
        # Add batch dimension
        pred_values = pred_values.unsqueeze(0)

    if indices.dim() == 1:
        # Broadcast to batch
        indices = indices.unsqueeze(0).expand(pred_values.shape[0], -1)

    B, N, C = pred_values.shape
    device = pred_values.device

    # Create empty images
    images = torch.zeros(B, C, image_size, image_size, device=device)

    # Fill in predicted pixels
    for b in range(B):
        for i, idx in enumerate(indices[b]):
            y = idx // image_size
            x = idx % image_size
            images[b, :, y, x] = pred_values[b, i]

    return images


class MetricsTracker:
    """
    Track metrics during training and evaluation

    Usage:
        tracker = MetricsTracker()
        for batch in loader:
            pred, target = model(batch)
            tracker.update(pred, target, batch)
        results = tracker.compute()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.mse_vals = []
        self.mae_vals = []
        self.psnr_vals = []
        self.ssim_vals = []

    def update(self, pred_values, target_values, pred_images=None, target_images=None):
        """
        Update metrics with batch

        Args:
            pred_values: (B, N, C) predicted output pixels
            target_values: (B, N, C) target output pixels
            pred_images: (B, C, H, W) optional full predicted images
            target_images: (B, C, H, W) optional full target images
        """
        # Pixel-level metrics
        self.mse_vals.append(mse_on_pixels(pred_values, target_values))
        self.mae_vals.append(mae_on_pixels(pred_values, target_values))

        # Image-level metrics (if full images provided)
        if pred_images is not None and target_images is not None:
            self.psnr_vals.append(psnr(pred_images, target_images))
            self.ssim_vals.append(ssim(pred_images, target_images))

    def compute(self):
        """
        Compute average metrics

        Returns:
            dict with averaged metrics
        """
        results = {
            'mse': np.mean(self.mse_vals) if self.mse_vals else None,
            'mae': np.mean(self.mae_vals) if self.mae_vals else None,
            'psnr': np.mean(self.psnr_vals) if self.psnr_vals else None,
            'ssim': np.mean(self.ssim_vals) if self.ssim_vals else None
        }
        return results

    def compute_std(self):
        """
        Compute standard deviations

        Returns:
            dict with standard deviations
        """
        results = {
            'mse_std': np.std(self.mse_vals) if self.mse_vals else None,
            'mae_std': np.std(self.mae_vals) if self.mae_vals else None,
            'psnr_std': np.std(self.psnr_vals) if self.psnr_vals else None,
            'ssim_std': np.std(self.ssim_vals) if self.ssim_vals else None
        }
        return results


def print_metrics(metrics, prefix=""):
    """
    Pretty print metrics

    Args:
        metrics: Dict of metrics
        prefix: Optional prefix for printing
    """
    print(f"{prefix}Metrics:")
    if metrics.get('mse') is not None:
        print(f"  MSE:  {metrics['mse']:.6f}")
    if metrics.get('mae') is not None:
        print(f"  MAE:  {metrics['mae']:.6f}")
    if metrics.get('psnr') is not None:
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
    if metrics.get('ssim') is not None:
        print(f"  SSIM: {metrics['ssim']:.4f}")


def visualize_predictions(
    input_coords, input_values,
    output_coords, pred_values, target_values,
    full_image, n_samples=4, image_size=32
):
    """
    Visualize predictions vs targets

    Args:
        input_coords: (B, N_in, 2) input coordinates
        input_values: (B, N_in, 3) input pixel values
        output_coords: (B, N_out, 2) output coordinates
        pred_values: (B, N_out, 3) predicted output values
        target_values: (B, N_out, 3) target output values
        full_image: (B, 3, H, W) full ground truth images
        n_samples: Number of samples to visualize
        image_size: Image size
    """
    import matplotlib.pyplot as plt

    n_samples = min(n_samples, input_coords.shape[0])

    # Get device from input tensors
    device = input_values.device

    fig, axes = plt.subplots(n_samples, 5, figsize=(20, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        # Convert coords to indices
        input_idx = (input_coords[i, :, 1] * (image_size - 1) * image_size +
                     input_coords[i, :, 0] * (image_size - 1)).long()
        output_idx = (output_coords[i, :, 1] * (image_size - 1) * image_size +
                      output_coords[i, :, 0] * (image_size - 1)).long()

        # Full image
        full_img = full_image[i].permute(1, 2, 0).cpu().numpy()
        axes[i, 0].imshow(full_img)
        axes[i, 0].set_title('Ground Truth')
        axes[i, 0].axis('off')

        # Input pixels
        input_img = torch.zeros(3, image_size, image_size, device=device)
        input_img.view(3, -1)[:, input_idx] = input_values[i].T
        axes[i, 1].imshow(input_img.permute(1, 2, 0).cpu().numpy())
        axes[i, 1].set_title(f'Input ({len(input_idx)} pixels)')
        axes[i, 1].axis('off')

        # Target output pixels
        target_img = torch.zeros(3, image_size, image_size, device=device)
        target_img.view(3, -1)[:, output_idx] = target_values[i].T
        axes[i, 2].imshow(target_img.permute(1, 2, 0).cpu().numpy())
        axes[i, 2].set_title(f'Target Output ({len(output_idx)} pixels)')
        axes[i, 2].axis('off')

        # Predicted output pixels
        pred_img = torch.zeros(3, image_size, image_size, device=device)
        pred_img.view(3, -1)[:, output_idx] = pred_values[i].T
        axes[i, 3].imshow(pred_img.permute(1, 2, 0).cpu().numpy())
        axes[i, 3].set_title('Predicted Output')
        axes[i, 3].axis('off')

        # Error map
        error = torch.abs(pred_values[i] - target_values[i]).mean(dim=-1)  # (N_out,)
        error_img = torch.zeros(image_size, image_size, device=device)
        error_img.view(-1)[output_idx] = error
        im = axes[i, 4].imshow(error_img.cpu().numpy(), cmap='hot', vmin=0, vmax=0.5)
        axes[i, 4].set_title(f'Error Map (MAE={error.mean():.3f})')
        axes[i, 4].axis('off')
        plt.colorbar(im, ax=axes[i, 4], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


# Test code
if __name__ == "__main__":
    print("Testing evaluation metrics...")

    # Create dummy data
    B, N, C = 4, 204, 3
    pred_values = torch.rand(B, N, C)
    target_values = torch.rand(B, N, C)

    pred_images = torch.rand(B, C, 32, 32)
    target_images = torch.rand(B, C, 32, 32)

    # Test pixel-level metrics
    print(f"MSE: {mse_on_pixels(pred_values, target_values):.6f}")
    print(f"MAE: {mae_on_pixels(pred_values, target_values):.6f}")

    # Test image-level metrics
    print(f"PSNR: {psnr(pred_images, target_images):.2f} dB")
    print(f"SSIM: {ssim(pred_images, target_images):.4f}")

    # Test metrics tracker
    print("\nTesting MetricsTracker...")
    tracker = MetricsTracker()
    for _ in range(5):
        tracker.update(pred_values, target_values, pred_images, target_images)

    results = tracker.compute()
    print_metrics(results)

    print("\nAll tests passed!")
