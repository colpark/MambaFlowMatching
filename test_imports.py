#!/usr/bin/env python3
"""
Quick test to verify all imports work correctly
"""
import sys
import os

# Add repo root to path
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)

print("Testing imports...")

try:
    from core.neural_fields.perceiver import FourierFeatures
    print("✓ core.neural_fields.perceiver imported")
except ImportError as e:
    print(f"✗ Failed to import core.neural_fields.perceiver: {e}")
    sys.exit(1)

try:
    from core.sparse.cifar10_sparse import SparseCIFAR10Dataset
    print("✓ core.sparse.cifar10_sparse imported")
except ImportError as e:
    print(f"✗ Failed to import core.sparse.cifar10_sparse: {e}")
    sys.exit(1)

try:
    from core.sparse.metrics import MetricsTracker
    print("✓ core.sparse.metrics imported")
except ImportError as e:
    print(f"✗ Failed to import core.sparse.metrics: {e}")
    sys.exit(1)

print("\n✓ All core module imports successful!")
print("\nNow testing training scripts...")

# Test V1 training script can be imported
sys.path.insert(0, os.path.join(repo_root, 'v1', 'training'))
try:
    from train_mamba_standalone import SSMBlockFast, MambaBlock
    print("✓ V1 train_mamba_standalone components imported")
except ImportError as e:
    print(f"✗ Failed to import V1 components: {e}")
    sys.exit(1)

print("\n✅ All imports working correctly!")
print("\nYou can now run:")
print("  cd v1/training && ./run_mamba_training.sh")
print("  cd v2/training && ./run_mamba_v2_training.sh")
