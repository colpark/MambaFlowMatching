"""
CIFAR-10 sparse dataset with instance-specific sampling

Provides 20% input pixels + 20% output pixels (non-overlapping)
"""

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np


class SparseCIFAR10Dataset(Dataset):
    """
    CIFAR-10 with instance-specific sparse sampling

    Each instance has:
    - 20% input pixels (randomly sampled)
    - 20% output pixels (randomly sampled, non-overlapping with input)
    - Remaining 60% unused
    """

    def __init__(
        self,
        root='./data',
        train=True,
        input_ratio=0.2,      # 20% input pixels
        output_ratio=0.2,     # 20% output pixels
        image_size=32,        # CIFAR-10 is 32x32
        download=True,
        seed=None
    ):
        """
        Args:
            root: Data root directory
            train: If True, use training set, else test set
            input_ratio: Fraction of pixels for input
            output_ratio: Fraction of pixels for output (non-overlapping)
            image_size: Image size (32 for CIFAR-10)
            download: Download dataset if not present
            seed: Random seed for reproducibility (None = different every time)
        """
        self.image_size = image_size
        self.input_ratio = input_ratio
        self.output_ratio = output_ratio
        self.total_pixels = image_size * image_size
        self.n_input_pixels = int(self.total_pixels * input_ratio)
        self.n_output_pixels = int(self.total_pixels * output_ratio)

        # Load CIFAR-10
        transform = transforms.Compose([
            transforms.ToTensor(),  # [0, 1] range
        ])

        self.dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            transform=transform,
            download=download
        )

        # Set random seed for reproducible sampling
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

        # Pre-generate sampling indices for each instance
        print(f"Generating instance-specific sparse sampling...")
        self._generate_sampling_indices()

    def _generate_sampling_indices(self):
        """Pre-generate random sampling indices for all instances"""
        self.input_indices = []
        self.output_indices = []

        for i in range(len(self.dataset)):
            # Random permutation of all pixel indices
            perm = self.rng.permutation(self.total_pixels)

            # First 20% for input
            input_idx = perm[:self.n_input_pixels]

            # Next 20% for output (non-overlapping)
            output_idx = perm[self.n_input_pixels:self.n_input_pixels + self.n_output_pixels]

            self.input_indices.append(input_idx)
            self.output_indices.append(output_idx)

    def __len__(self):
        return len(self.dataset)

    def _indices_to_coords(self, indices):
        """
        Convert flat indices to (x, y) coordinates in [0, 1]

        Args:
            indices: (N,) flat indices

        Returns:
            coords: (N, 2) normalized coordinates in [0, 1]
        """
        y = indices // self.image_size
        x = indices % self.image_size

        # Normalize to [0, 1]
        coords = np.stack([
            x / (self.image_size - 1),
            y / (self.image_size - 1)
        ], axis=1).astype(np.float32)

        return coords

    def __getitem__(self, idx):
        """
        Get sparse sample

        Returns:
            dict with:
                - input_coords: (N_in, 2) input pixel coordinates [0, 1]
                - input_values: (N_in, 3) input pixel RGB values [0, 1]
                - output_coords: (N_out, 2) output pixel coordinates [0, 1]
                - output_values: (N_out, 3) output pixel RGB values [0, 1]
                - full_image: (3, H, W) full image for evaluation
                - label: int class label
        """
        # Get full image
        image, label = self.dataset[idx]  # (3, 32, 32), int

        # Flatten image to (3, H*W)
        image_flat = image.view(3, -1)  # (3, 1024)

        # Get input pixels
        input_idx = self.input_indices[idx]
        input_coords = self._indices_to_coords(input_idx)
        input_values = image_flat[:, input_idx].T  # (N_in, 3)

        # Get output pixels
        output_idx = self.output_indices[idx]
        output_coords = self._indices_to_coords(output_idx)
        output_values = image_flat[:, output_idx].T  # (N_out, 3)

        return {
            'input_coords': torch.from_numpy(input_coords),
            'input_values': input_values,  # Already (N_in, 3)
            'output_coords': torch.from_numpy(output_coords),
            'output_values': output_values,  # Already (N_out, 3)
            'full_image': image,
            'label': label,
            'input_indices': torch.from_numpy(input_idx).long(),
            'output_indices': torch.from_numpy(output_idx).long()
        }


def visualize_sparse_sample(sample, figsize=(15, 3)):
    """
    Visualize a sparse sample

    Args:
        sample: Dict from SparseCIFAR10Dataset
        figsize: Figure size
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Full image
    full_img = sample['full_image'].permute(1, 2, 0).numpy()
    axes[0].imshow(full_img)
    axes[0].set_title('Full Image')
    axes[0].axis('off')

    # Input pixels
    input_img = torch.zeros(3, 32, 32)
    input_idx = sample['input_indices']
    input_vals = sample['input_values']
    input_img.view(3, -1)[:, input_idx] = input_vals.T
    axes[1].imshow(input_img.permute(1, 2, 0).numpy())
    axes[1].set_title(f'Input Pixels ({len(input_idx)})')
    axes[1].axis('off')

    # Output pixels
    output_img = torch.zeros(3, 32, 32)
    output_idx = sample['output_indices']
    output_vals = sample['output_values']
    output_img.view(3, -1)[:, output_idx] = output_vals.T
    axes[2].imshow(output_img.permute(1, 2, 0).numpy())
    axes[2].set_title(f'Output Pixels ({len(output_idx)})')
    axes[2].axis('off')

    # Input + Output combined
    combined_img = torch.zeros(3, 32, 32)
    combined_img.view(3, -1)[:, input_idx] = input_vals.T
    combined_img.view(3, -1)[:, output_idx] = output_vals.T
    axes[3].imshow(combined_img.permute(1, 2, 0).numpy())
    axes[3].set_title('Input + Output (40%)')
    axes[3].axis('off')

    plt.tight_layout()
    return fig


# Test code
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Creating sparse CIFAR-10 dataset...")
    dataset = SparseCIFAR10Dataset(
        root='./data',
        train=True,
        input_ratio=0.2,
        output_ratio=0.2,
        download=True,
        seed=42
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Pixels per image: {dataset.total_pixels}")
    print(f"Input pixels: {dataset.n_input_pixels} ({dataset.input_ratio*100}%)")
    print(f"Output pixels: {dataset.n_output_pixels} ({dataset.output_ratio*100}%)")

    # Get a sample
    sample = dataset[0]
    print(f"\nSample structure:")
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}, dtype={val.dtype}")
        else:
            print(f"  {key}: {val}")

    # Visualize
    print("\nVisualizing samples...")
    fig, axes = plt.subplots(3, 4, figsize=(15, 9))

    for i in range(3):
        sample = dataset[i]

        # Full image
        full_img = sample['full_image'].permute(1, 2, 0).numpy()
        axes[i, 0].imshow(full_img)
        axes[i, 0].set_title(f'Sample {i}: Full Image')
        axes[i, 0].axis('off')

        # Input pixels
        input_img = torch.zeros(3, 32, 32)
        input_idx = sample['input_indices']
        input_vals = sample['input_values']
        input_img.view(3, -1)[:, input_idx] = input_vals.T
        axes[i, 1].imshow(input_img.permute(1, 2, 0).numpy())
        axes[i, 1].set_title(f'Input ({len(input_idx)} pixels)')
        axes[i, 1].axis('off')

        # Output pixels
        output_img = torch.zeros(3, 32, 32)
        output_idx = sample['output_indices']
        output_vals = sample['output_values']
        output_img.view(3, -1)[:, output_idx] = output_vals.T
        axes[i, 2].imshow(output_img.permute(1, 2, 0).numpy())
        axes[i, 2].set_title(f'Output ({len(output_idx)} pixels)')
        axes[i, 2].axis('off')

        # Combined
        combined_img = torch.zeros(3, 32, 32)
        combined_img.view(3, -1)[:, input_idx] = input_vals.T
        combined_img.view(3, -1)[:, output_idx] = output_vals.T
        axes[i, 3].imshow(combined_img.permute(1, 2, 0).numpy())
        axes[i, 3].set_title('Input + Output')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig('sparse_cifar10_samples.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to sparse_cifar10_samples.png")
    plt.show()
