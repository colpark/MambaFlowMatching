"""
Space-Filling Curve Utilities for Improved Spatial Locality

Implements Morton (Z-order) and Hilbert curves to order 2D coordinates
into 1D sequences with better spatial locality than row-major ordering.

Benefits for MAMBA:
- Neighboring pixels in 2D are also neighbors in 1D sequence
- State space model maintains better spatial coherence
- Bidirectional processing follows spatially meaningful paths
"""
import torch
import numpy as np


def morton_encode(x, y):
    """
    Encode 2D coordinates to Morton (Z-order) code

    Interleaves bits of x and y coordinates to create a single integer
    that preserves 2D locality in 1D space.

    Example for 4x4 grid:
    Row-major: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    Morton:    0,1,4,5,2,3,6,7,8,9,12,13,10,11,14,15

    Args:
        x: int or array of x coordinates (0 to resolution-1)
        y: int or array of y coordinates (0 to resolution-1)

    Returns:
        Morton code(s)
    """
    def part1by1(n):
        """Separate bits by 1 position"""
        n = np.asarray(n, dtype=np.uint32)
        n = (n | (n << 8)) & 0x00FF00FF
        n = (n | (n << 4)) & 0x0F0F0F0F
        n = (n | (n << 2)) & 0x33333333
        n = (n | (n << 1)) & 0x55555555
        return n

    return part1by1(x) | (part1by1(y) << 1)


def morton_decode(code):
    """
    Decode Morton code back to 2D coordinates

    Args:
        code: Morton code(s)

    Returns:
        (x, y) coordinates
    """
    def compact1by1(n):
        """Compact bits separated by 1 position"""
        n = np.asarray(n, dtype=np.uint32)
        n = (n & 0x55555555)
        n = (n | (n >> 1)) & 0x33333333
        n = (n | (n >> 2)) & 0x0F0F0F0F
        n = (n | (n >> 4)) & 0x00FF00FF
        n = (n | (n >> 8)) & 0x0000FFFF
        return n

    code = np.asarray(code, dtype=np.uint32)
    x = compact1by1(code)
    y = compact1by1(code >> 1)
    return x, y


def get_morton_order_indices(resolution):
    """
    Get indices to reorder flat coordinates array into Morton order

    Args:
        resolution: Grid resolution (e.g., 32 for 32x32)

    Returns:
        indices: Array of indices to reorder from row-major to Morton order

    Example:
        coords = create_row_major_coords(32)  # (1024, 2) in row-major order
        morton_indices = get_morton_order_indices(32)
        morton_coords = coords[morton_indices]  # Now in Morton order!
    """
    # Generate all (x, y) pairs in row-major order
    y, x = np.meshgrid(np.arange(resolution), np.arange(resolution), indexing='ij')
    x_flat = x.flatten()
    y_flat = y.flatten()

    # Compute Morton codes for each coordinate
    morton_codes = morton_encode(x_flat, y_flat)

    # Get indices that would sort by Morton code
    morton_order_indices = np.argsort(morton_codes)

    return morton_order_indices


def get_hilbert_order_indices(resolution):
    """
    Get indices to reorder flat coordinates array into Hilbert curve order

    Hilbert curve has better locality than Morton but is more complex.
    For now, we use Morton as it's simpler and still effective.

    Args:
        resolution: Grid resolution (must be power of 2)

    Returns:
        indices: Array of indices to reorder from row-major to Hilbert order
    """
    # Simple implementation: use Morton for now
    # TODO: Implement true Hilbert curve if needed
    return get_morton_order_indices(resolution)


def reorder_coordinates_morton(coords, resolution):
    """
    Reorder 2D coordinates from row-major to Morton order

    Args:
        coords: (N, 2) tensor of normalized coordinates in [0, 1]
        resolution: Grid resolution used to compute Morton codes

    Returns:
        reordered_coords: (N, 2) tensor in Morton order
        indices: Indices used for reordering (for inverse operation)
    """
    # Convert normalized coords [0, 1] to discrete grid [0, resolution-1]
    coords_discrete = (coords * (resolution - 1)).long()

    # Get Morton order indices
    morton_indices = get_morton_order_indices(resolution)
    morton_indices = torch.from_numpy(morton_indices).to(coords.device)

    # Reorder coordinates
    reordered_coords = coords[morton_indices]

    return reordered_coords, morton_indices


def reorder_sequence_morton(sequence, resolution):
    """
    Reorder a sequence (tokens/features) from row-major to Morton order

    Args:
        sequence: (B, N, D) tensor where N = resolution^2
        resolution: Grid resolution

    Returns:
        reordered_sequence: (B, N, D) tensor in Morton order
        indices: Indices for inverse reordering
    """
    morton_indices = get_morton_order_indices(resolution)
    morton_indices = torch.from_numpy(morton_indices).to(sequence.device)

    # Reorder along sequence dimension
    reordered_sequence = sequence[:, morton_indices, :]

    return reordered_sequence, morton_indices


def inverse_reorder_sequence(sequence, indices):
    """
    Inverse reordering to go from Morton order back to row-major

    Args:
        sequence: (B, N, D) tensor in Morton order
        indices: Indices from reorder_sequence_morton

    Returns:
        original_sequence: (B, N, D) tensor in original row-major order
    """
    # Create inverse permutation
    inverse_indices = torch.argsort(indices)

    # Reorder back
    original_sequence = sequence[:, inverse_indices, :]

    return original_sequence


def visualize_ordering(resolution=8, save_path=None):
    """
    Visualize different orderings for debugging

    Args:
        resolution: Grid resolution
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt

    # Row-major order
    row_major = np.arange(resolution * resolution).reshape(resolution, resolution)

    # Morton order
    morton_indices = get_morton_order_indices(resolution)
    morton_order = np.empty_like(row_major)
    y, x = np.meshgrid(np.arange(resolution), np.arange(resolution), indexing='ij')
    morton_order[y.flatten(), x.flatten()] = np.arange(resolution * resolution)[morton_indices]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot row-major
    im1 = axes[0].imshow(row_major, cmap='viridis')
    axes[0].set_title(f'Row-Major Order ({resolution}×{resolution})')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0], label='Sequence Index')

    # Plot Morton
    im2 = axes[1].imshow(morton_order, cmap='viridis')
    axes[1].set_title(f'Morton (Z-Order) ({resolution}×{resolution})')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[1], label='Sequence Index')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ordering visualization to {save_path}")
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    # Test Morton encoding/decoding
    print("Testing Morton curve encoding...")

    # Test single coordinate
    x, y = 3, 2
    code = morton_encode(x, y)
    x_dec, y_dec = morton_decode(code)
    print(f"Encode ({x}, {y}) → {code}")
    print(f"Decode {code} → ({x_dec}, {y_dec})")
    assert x == x_dec and y == y_dec, "Encoding/decoding mismatch!"

    # Test ordering
    resolution = 8
    print(f"\nTesting {resolution}×{resolution} grid ordering...")
    morton_indices = get_morton_order_indices(resolution)
    print(f"First 16 indices in Morton order: {morton_indices[:16]}")

    # Visualize
    print("\nGenerating visualization...")
    visualize_ordering(resolution=8, save_path='morton_ordering.png')
    visualize_ordering(resolution=16, save_path='morton_ordering_16.png')

    print("\n✓ Morton curve implementation working correctly!")
