"""
Synthetic Sinusoidal Dataset Generator

Creates various complexity levels of sinusoidal functions for testing
sparse reconstruction methods in a controlled environment.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


class SinusoidalDataset:
    """
    Generate synthetic 2D sinusoidal patterns with varying complexity

    Complexity Levels:
    1. Simple: Single frequency sine wave
    2. Multi-frequency: Sum of 2-3 frequencies
    3. Radial: Radial frequency patterns
    4. Interference: Wave interference patterns
    5. Modulated: Amplitude/frequency modulated
    6. Composite: Complex superposition of patterns
    """

    def __init__(
        self,
        resolution: int = 32,
        num_samples: int = 1000,
        complexity: str = 'simple',
        noise_level: float = 0.0,
        seed: int = 42
    ):
        self.resolution = resolution
        self.num_samples = num_samples
        self.complexity = complexity
        self.noise_level = noise_level

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Create coordinate grid
        x = np.linspace(0, 2 * np.pi, resolution)
        y = np.linspace(0, 2 * np.pi, resolution)
        self.X, self.Y = np.meshgrid(x, y)

        # Generate dataset
        self.data = self._generate_dataset()

    def _generate_dataset(self) -> torch.Tensor:
        """Generate dataset based on complexity level"""

        if self.complexity == 'simple':
            return self._generate_simple()
        elif self.complexity == 'multi_frequency':
            return self._generate_multi_frequency()
        elif self.complexity == 'radial':
            return self._generate_radial()
        elif self.complexity == 'interference':
            return self._generate_interference()
        elif self.complexity == 'modulated':
            return self._generate_modulated()
        elif self.complexity == 'composite':
            return self._generate_composite()
        elif self.complexity == 'all':
            return self._generate_all_types()
        else:
            raise ValueError(f"Unknown complexity: {self.complexity}")

    def _generate_simple(self) -> torch.Tensor:
        """
        Simple: Single frequency sine wave
        z = sin(k_x * x + k_y * y + φ)
        """
        data = []
        for _ in range(self.num_samples):
            # Random frequency and phase
            k_x = np.random.uniform(1, 3)
            k_y = np.random.uniform(1, 3)
            phi = np.random.uniform(0, 2 * np.pi)

            # Generate pattern
            z = np.sin(k_x * self.X + k_y * self.Y + phi)

            # Add noise
            if self.noise_level > 0:
                z += np.random.randn(*z.shape) * self.noise_level

            # Normalize to [0, 1]
            z = (z + 1) / 2

            data.append(z)

        return torch.FloatTensor(np.array(data)).unsqueeze(1)  # (N, 1, H, W)

    def _generate_multi_frequency(self) -> torch.Tensor:
        """
        Multi-frequency: Sum of 2-3 sine waves with different frequencies
        z = Σ A_i * sin(k_xi * x + k_yi * y + φ_i)
        """
        data = []
        for _ in range(self.num_samples):
            num_freqs = np.random.randint(2, 4)
            z = np.zeros_like(self.X)

            for i in range(num_freqs):
                k_x = np.random.uniform(0.5, 5)
                k_y = np.random.uniform(0.5, 5)
                phi = np.random.uniform(0, 2 * np.pi)
                amp = np.random.uniform(0.3, 1.0)

                z += amp * np.sin(k_x * self.X + k_y * self.Y + phi)

            # Normalize
            z = z / num_freqs

            if self.noise_level > 0:
                z += np.random.randn(*z.shape) * self.noise_level

            z = (z + 1) / 2
            data.append(z)

        return torch.FloatTensor(np.array(data)).unsqueeze(1)

    def _generate_radial(self) -> torch.Tensor:
        """
        Radial: Radial frequency patterns
        z = sin(k_r * r + φ) where r = sqrt(x² + y²)
        """
        data = []

        # Center coordinates
        center_x = self.X.mean()
        center_y = self.Y.mean()
        R = np.sqrt((self.X - center_x)**2 + (self.Y - center_y)**2)

        for _ in range(self.num_samples):
            k_r = np.random.uniform(2, 8)
            phi = np.random.uniform(0, 2 * np.pi)

            z = np.sin(k_r * R + phi)

            if self.noise_level > 0:
                z += np.random.randn(*z.shape) * self.noise_level

            z = (z + 1) / 2
            data.append(z)

        return torch.FloatTensor(np.array(data)).unsqueeze(1)

    def _generate_interference(self) -> torch.Tensor:
        """
        Interference: Two plane waves interfering
        z = sin(k1_x * x + k1_y * y) + sin(k2_x * x + k2_y * y)
        Creates Moiré-like patterns
        """
        data = []
        for _ in range(self.num_samples):
            # First wave
            k1_x = np.random.uniform(2, 6)
            k1_y = np.random.uniform(2, 6)
            phi1 = np.random.uniform(0, 2 * np.pi)

            # Second wave (slightly different frequency)
            k2_x = k1_x + np.random.uniform(-1, 1)
            k2_y = k1_y + np.random.uniform(-1, 1)
            phi2 = np.random.uniform(0, 2 * np.pi)

            z = np.sin(k1_x * self.X + k1_y * self.Y + phi1) + \
                np.sin(k2_x * self.X + k2_y * self.Y + phi2)

            if self.noise_level > 0:
                z += np.random.randn(*z.shape) * self.noise_level

            z = (z + 1) / 2
            data.append(z)

        return torch.FloatTensor(np.array(data)).unsqueeze(1)

    def _generate_modulated(self) -> torch.Tensor:
        """
        Modulated: Amplitude or frequency modulation
        AM: z = [1 + m * sin(k_m * x)] * sin(k_c * x)
        FM: z = sin(k_c * x + β * sin(k_m * x))
        """
        data = []
        for _ in range(self.num_samples):
            # Choose AM or FM randomly
            if np.random.rand() < 0.5:
                # Amplitude Modulation
                k_carrier_x = np.random.uniform(3, 7)
                k_carrier_y = np.random.uniform(3, 7)
                k_mod_x = np.random.uniform(0.5, 2)
                k_mod_y = np.random.uniform(0.5, 2)
                m = np.random.uniform(0.5, 1.0)  # Modulation index

                modulator = 1 + m * np.sin(k_mod_x * self.X + k_mod_y * self.Y)
                carrier = np.sin(k_carrier_x * self.X + k_carrier_y * self.Y)
                z = modulator * carrier
            else:
                # Frequency Modulation
                k_carrier = np.random.uniform(3, 7)
                k_mod = np.random.uniform(0.5, 2)
                beta = np.random.uniform(1, 3)  # Modulation index

                z = np.sin(k_carrier * self.X +
                          beta * np.sin(k_mod * self.X + k_mod * self.Y))

            if self.noise_level > 0:
                z += np.random.randn(*z.shape) * self.noise_level

            z = (z + 1) / 2
            data.append(z)

        return torch.FloatTensor(np.array(data)).unsqueeze(1)

    def _generate_composite(self) -> torch.Tensor:
        """
        Composite: Complex superposition of multiple pattern types
        Combines radial + linear + modulated components
        """
        data = []

        center_x = self.X.mean()
        center_y = self.Y.mean()
        R = np.sqrt((self.X - center_x)**2 + (self.Y - center_y)**2)
        Theta = np.arctan2(self.Y - center_y, self.X - center_x)

        for _ in range(self.num_samples):
            z = np.zeros_like(self.X)

            # Radial component
            k_r = np.random.uniform(2, 6)
            z += 0.4 * np.sin(k_r * R)

            # Angular component
            k_theta = np.random.randint(2, 6)
            z += 0.3 * np.sin(k_theta * Theta)

            # Linear wave
            k_x = np.random.uniform(2, 5)
            k_y = np.random.uniform(2, 5)
            z += 0.3 * np.sin(k_x * self.X + k_y * self.Y)

            if self.noise_level > 0:
                z += np.random.randn(*z.shape) * self.noise_level

            z = (z + 1) / 2
            data.append(z)

        return torch.FloatTensor(np.array(data)).unsqueeze(1)

    def _generate_all_types(self) -> torch.Tensor:
        """
        Generate dataset with all pattern types mixed
        """
        patterns_per_type = self.num_samples // 6

        data = []
        for pattern_type in ['simple', 'multi_frequency', 'radial',
                            'interference', 'modulated', 'composite']:
            temp_gen = SinusoidalDataset(
                resolution=self.resolution,
                num_samples=patterns_per_type,
                complexity=pattern_type,
                noise_level=self.noise_level,
                seed=np.random.randint(0, 10000)
            )
            data.append(temp_gen.data)

        return torch.cat(data, dim=0)

    def get_train_test_split(
        self,
        train_sparsity: float = 0.05,
        test_sparsity: float = 0.05,
        strategy: str = 'random'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split observations into training and testing sets (disjoint)

        Args:
            train_sparsity: Fraction of pixels for training (0.05 = 5%)
            test_sparsity: Fraction of pixels for testing (0.05 = 5%)
            strategy: 'random', 'uniform', 'edge-aware'

        Returns:
            train_coords: (N, num_train, 2) training coordinates
            train_values: (N, num_train, 1) training observed values
            test_coords: (N, num_test, 2) testing coordinates
            test_values: (N, num_test, 1) testing observed values
            full_images: (N, 1, H, W) complete ground truth
        """
        N, C, H, W = self.data.shape
        num_train = int(H * W * train_sparsity)
        num_test = int(H * W * test_sparsity)

        train_coords_list = []
        train_values_list = []
        test_coords_list = []
        test_values_list = []

        for i in range(N):
            if strategy == 'random':
                # Random sampling with disjoint train/test sets
                all_indices = torch.randperm(H * W)
                train_indices = all_indices[:num_train]
                test_indices = all_indices[num_train:num_train + num_test]

            elif strategy == 'uniform':
                # Uniform grid sampling (still disjoint)
                stride = int(np.sqrt(1 / (train_sparsity + test_sparsity)))
                all_y = torch.arange(0, H, stride).repeat_interleave(len(torch.arange(0, W, stride)))
                all_x = torch.arange(0, W, stride).repeat(len(torch.arange(0, H, stride)))

                # Split into train/test
                perm = torch.randperm(len(all_y))
                train_indices_subset = perm[:min(num_train, len(perm))]
                test_indices_subset = perm[num_train:min(num_train + num_test, len(perm))]

                train_y_coords = all_y[train_indices_subset]
                train_x_coords = all_x[train_indices_subset]
                test_y_coords = all_y[test_indices_subset]
                test_x_coords = all_x[test_indices_subset]

                train_indices = train_y_coords * W + train_x_coords
                test_indices = test_y_coords * W + test_x_coords

            elif strategy == 'edge_aware':
                # Sample more from high-gradient regions
                img = self.data[i, 0].numpy()
                grad_x = np.abs(np.gradient(img, axis=0))
                grad_y = np.abs(np.gradient(img, axis=1))
                gradient_mag = grad_x + grad_y

                # Normalize to probabilities
                probs = gradient_mag.flatten()
                probs = probs / probs.sum()

                # Sample based on gradient magnitude (without replacement)
                all_indices = np.random.choice(
                    H * W,
                    size=num_train + num_test,
                    replace=False,
                    p=probs
                )
                train_indices = torch.from_numpy(all_indices[:num_train])
                test_indices = torch.from_numpy(all_indices[num_train:num_train + num_test])

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # Training set
            train_y_coords = train_indices // W
            train_x_coords = train_indices % W

            # Normalize coordinates to [-1, 1]
            train_coords = torch.stack([
                2 * train_x_coords.float() / (W - 1) - 1,
                2 * train_y_coords.float() / (H - 1) - 1
            ], dim=-1)  # (num_train, 2)

            train_values = self.data[i, 0, train_y_coords, train_x_coords].unsqueeze(-1)  # (num_train, 1)

            # Testing set
            test_y_coords = test_indices // W
            test_x_coords = test_indices % W

            test_coords = torch.stack([
                2 * test_x_coords.float() / (W - 1) - 1,
                2 * test_y_coords.float() / (H - 1) - 1
            ], dim=-1)  # (num_test, 2)

            test_values = self.data[i, 0, test_y_coords, test_x_coords].unsqueeze(-1)  # (num_test, 1)

            train_coords_list.append(train_coords)
            train_values_list.append(train_values)
            test_coords_list.append(test_coords)
            test_values_list.append(test_values)

        train_coords = torch.stack(train_coords_list, dim=0)  # (N, num_train, 2)
        train_values = torch.stack(train_values_list, dim=0)  # (N, num_train, 1)
        test_coords = torch.stack(test_coords_list, dim=0)    # (N, num_test, 2)
        test_values = torch.stack(test_values_list, dim=0)    # (N, num_test, 1)

        return train_coords, train_values, test_coords, test_values, self.data

    def get_sparse_observations(
        self,
        sparsity: float = 0.05,
        strategy: str = 'random'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample sparse observations from the dataset (legacy method)

        Args:
            sparsity: Fraction of pixels to observe (0.05 = 5%)
            strategy: 'random', 'uniform', 'edge-aware'

        Returns:
            coords: (N, num_sparse, 2) coordinates
            values: (N, num_sparse, 1) observed values
            full_images: (N, 1, H, W) complete ground truth
        """
        N, C, H, W = self.data.shape
        num_sparse = int(H * W * sparsity)

        coords_list = []
        values_list = []

        for i in range(N):
            if strategy == 'random':
                # Random sampling
                indices = torch.randperm(H * W)[:num_sparse]
                y_coords = indices // W
                x_coords = indices % W

            elif strategy == 'uniform':
                # Uniform grid sampling
                stride = int(np.sqrt(1 / sparsity))
                y_coords = torch.arange(0, H, stride).repeat_interleave(len(torch.arange(0, W, stride)))
                x_coords = torch.arange(0, W, stride).repeat(len(torch.arange(0, H, stride)))

                # Trim to exact number
                if len(y_coords) > num_sparse:
                    indices = torch.randperm(len(y_coords))[:num_sparse]
                    y_coords = y_coords[indices]
                    x_coords = x_coords[indices]

            elif strategy == 'edge_aware':
                # Sample more from high-gradient regions
                img = self.data[i, 0].numpy()
                grad_x = np.abs(np.gradient(img, axis=0))
                grad_y = np.abs(np.gradient(img, axis=1))
                gradient_mag = grad_x + grad_y

                # Normalize to probabilities
                probs = gradient_mag.flatten()
                probs = probs / probs.sum()

                # Sample based on gradient magnitude
                indices = np.random.choice(H * W, size=num_sparse, replace=False, p=probs)
                y_coords = torch.from_numpy(indices // W)
                x_coords = torch.from_numpy(indices % W)

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # Normalize coordinates to [-1, 1]
            coords = torch.stack([
                2 * x_coords.float() / (W - 1) - 1,
                2 * y_coords.float() / (H - 1) - 1
            ], dim=-1)  # (num_sparse, 2)

            # Get values at sampled coordinates
            values = self.data[i, 0, y_coords, x_coords].unsqueeze(-1)  # (num_sparse, 1)

            coords_list.append(coords)
            values_list.append(values)

        coords = torch.stack(coords_list, dim=0)  # (N, num_sparse, 2)
        values = torch.stack(values_list, dim=0)  # (N, num_sparse, 1)

        return coords, values, self.data

    def visualize_samples(self, num_samples: int = 6, save_path: Optional[str] = None):
        """Visualize random samples from the dataset"""
        indices = np.random.choice(len(self.data), size=num_samples, replace=False)

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        for idx, ax in zip(indices, axes):
            img = self.data[idx, 0].numpy()
            im = ax.imshow(img, cmap='RdBu', vmin=0, vmax=1)
            ax.set_title(f'Sample {idx}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        plt.suptitle(f'Sinusoidal Dataset - Complexity: {self.complexity}', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def visualize_sparse_vs_full(
        self,
        sample_idx: int = 0,
        sparsity: float = 0.2,
        strategy: str = 'random',
        save_path: Optional[str] = None
    ):
        """Visualize sparse observations vs full image"""
        coords, values, full = self.get_sparse_observations(sparsity, strategy)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Full image
        axes[0].imshow(full[sample_idx, 0].numpy(), cmap='RdBu', vmin=0, vmax=1)
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')

        # Sparse observations
        sparse_img = np.zeros_like(full[sample_idx, 0].numpy())
        coords_denorm = (coords[sample_idx].numpy() + 1) / 2 * (self.resolution - 1)
        x_coords = coords_denorm[:, 0].astype(int)
        y_coords = coords_denorm[:, 1].astype(int)
        sparse_img[y_coords, x_coords] = values[sample_idx, :, 0].numpy()

        axes[1].imshow(sparse_img, cmap='RdBu', vmin=0, vmax=1)
        axes[1].set_title(f'Sparse Observations ({sparsity*100:.0f}%, {strategy})')
        axes[1].axis('off')

        plt.suptitle(f'Sample {sample_idx} - Complexity: {self.complexity}', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


def create_benchmark_datasets(
    resolution: int = 32,
    num_samples_per_complexity: int = 500,
    noise_levels: List[float] = [0.0, 0.05, 0.1],
    save_dir: str = 'synthetic_experiments/datasets/generated'
) -> dict:
    """
    Create comprehensive benchmark datasets

    Returns dictionary of datasets for all complexity levels and noise levels
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    datasets = {}
    complexities = ['simple', 'multi_frequency', 'radial', 'interference',
                   'modulated', 'composite']

    for complexity in complexities:
        for noise in noise_levels:
            key = f"{complexity}_noise{noise:.2f}"

            print(f"Generating {key}...")
            dataset = SinusoidalDataset(
                resolution=resolution,
                num_samples=num_samples_per_complexity,
                complexity=complexity,
                noise_level=noise
            )

            datasets[key] = dataset

            # Save dataset
            save_path = os.path.join(save_dir, f"{key}.pt")
            torch.save({
                'data': dataset.data,
                'resolution': resolution,
                'complexity': complexity,
                'noise_level': noise
            }, save_path)

            # Visualize samples
            vis_path = os.path.join(save_dir, f"{key}_samples.png")
            dataset.visualize_samples(num_samples=6, save_path=vis_path)

            # Visualize sparse vs full
            sparse_path = os.path.join(save_dir, f"{key}_sparse.png")
            dataset.visualize_sparse_vs_full(
                sample_idx=0,
                sparsity=0.2,
                strategy='random',
                save_path=sparse_path
            )

    print(f"\n✅ Created {len(datasets)} benchmark datasets")
    print(f"📁 Saved to: {save_dir}")

    return datasets


if __name__ == '__main__':
    # Example usage
    print("Creating benchmark sinusoidal datasets...\n")

    datasets = create_benchmark_datasets(
        resolution=32,
        num_samples_per_complexity=500,
        noise_levels=[0.0, 0.05, 0.1]
    )

    print("\nDataset Statistics:")
    for name, dataset in datasets.items():
        print(f"  {name}: {dataset.data.shape}")
