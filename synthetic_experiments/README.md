# Synthetic Experiments for Method Comparison

Controlled experimental environment using synthetic sinusoidal datasets to compare different sparse reconstruction methods.

---

## 🎯 Purpose

Test and compare the 20 improvement methods from `docs/PERFORMANCE_IMPROVEMENTS.md` in a simplified, reproducible environment with:
- **Known ground truth**: Exact sinusoidal functions
- **Controlled complexity**: 6 difficulty levels
- **Fast iteration**: Quick training (minutes vs hours)
- **Clear metrics**: Quantitative comparison on identical data
- **Ablation studies**: Isolate individual improvements

---

## 📊 Synthetic Datasets

### Dataset Complexity Levels

#### **1. Simple** - Single Frequency Sine Wave
```
z = sin(k_x * x + k_y * y + φ)
```
- **Purpose**: Test basic reconstruction capability
- **Expected PSNR**: 35-40 dB (easy)
- **Key Challenge**: Minimal - single frequency

#### **2. Multi-Frequency** - Sum of 2-3 Frequencies
```
z = Σ A_i * sin(k_xi * x + k_yi * y + φ_i)
```
- **Purpose**: Test multi-scale reconstruction
- **Expected PSNR**: 30-35 dB (moderate)
- **Key Challenge**: Multiple frequency components

#### **3. Radial** - Radial Frequency Patterns
```
z = sin(k_r * r + φ) where r = sqrt(x² + y²)
```
- **Purpose**: Test rotational symmetry handling
- **Expected PSNR**: 32-37 dB (moderate)
- **Key Challenge**: Circular patterns vs linear processing

#### **4. Interference** - Wave Interference Patterns
```
z = sin(k1_x * x + k1_y * y) + sin(k2_x * x + k2_y * y)
```
- **Purpose**: Test Moiré-like pattern reconstruction
- **Expected PSNR**: 28-33 dB (hard)
- **Key Challenge**: Beating patterns, local vs global

#### **5. Modulated** - AM/FM Modulation
```
AM: z = [1 + m * sin(k_m * x)] * sin(k_c * x)
FM: z = sin(k_c * x + β * sin(k_m * x))
```
- **Purpose**: Test amplitude/frequency variation
- **Expected PSNR**: 25-30 dB (hard)
- **Key Challenge**: Non-stationary patterns

#### **6. Composite** - Complex Superposition
```
z = radial + angular + linear components
```
- **Purpose**: Test realistic complex patterns
- **Expected PSNR**: 22-28 dB (very hard)
- **Key Challenge**: Multiple simultaneous patterns

### Noise Levels

- **0.0**: Clean (baseline performance)
- **0.05**: Low noise (robustness test)
- **0.1**: High noise (stress test)

---

## 🚀 Quick Start

### 1. Generate Synthetic Datasets

```bash
cd synthetic_experiments/datasets
python sinusoidal_generator.py
```

This creates:
- 18 dataset variants (6 complexities × 3 noise levels)
- 500 samples per variant
- Saved to `datasets/generated/`

### 2. Train Baseline

```bash
cd synthetic_experiments/baselines
python train_baseline_mamba.py \
    --complexity simple \
    --epochs 100 \
    --batch_size 32 \
    --train_sparsity 0.05 \
    --test_sparsity 0.05
```

**Training Strategy**:
- **Train**: Random 5% of pixels as input observations
- **Test**: Different random 5% of pixels during training (disjoint from train)
- **Evaluation**: Reconstruct full 100% field (all 1024 pixels for 32×32)

**Expected Results**:
- Training time: ~5-10 minutes on CPU
- Final PSNR: 35-40 dB on simple dataset (evaluated on full field)
- Checkpoints saved to `baselines/checkpoints/`

### 3. Evaluate and Compare

```bash
cd synthetic_experiments/evaluation
python compare_methods.py \
    --complexities simple multi_frequency radial \
    --num_samples 100
```

**Outputs**:
- `results/comparison_plot.png`: Visual comparison
- `results/comparison_table.md`: Numerical results
- `results/{complexity}_comparison.json`: Detailed metrics

---

## 📁 Directory Structure

```
synthetic_experiments/
├── README.md                          # This file
├── datasets/                          # Dataset generation
│   ├── sinusoidal_generator.py       # Main generator
│   ├── generated/                    # Generated datasets
│   │   ├── simple_noise0.00.pt       # Clean simple patterns
│   │   ├── radial_noise0.05.pt       # Radial with noise
│   │   └── ...                       # 18 total variants
│   └── __init__.py
│
├── baselines/                        # Baseline implementations
│   ├── train_baseline_mamba.py       # V1 baseline on synthetic
│   ├── checkpoints/                  # Trained baselines
│   └── __init__.py
│
├── methods/                          # Improvement methods
│   ├── content_aware_sampling.py     # Solution #2
│   ├── wavelet_sparse.py            # Solution #3
│   ├── hierarchical_mamba.py        # Solution #5
│   ├── latent_diffusion.py          # Solution #9
│   └── ... (implement 20 methods)
│
├── evaluation/                       # Evaluation scripts
│   ├── compare_methods.py           # Main comparison
│   ├── metrics.py                   # Metric calculations
│   └── visualize.py                 # Visualization tools
│
├── results/                         # Experiment results
│   ├── comparison_plot.png
│   ├── comparison_table.md
│   └── {complexity}_comparison.json
│
└── visualizations/                  # Generated plots
    ├── dataset_samples/
    └── reconstruction_comparisons/
```

---

## 🔬 Implementing Improvement Methods

Each improvement method should follow this template:

```python
# synthetic_experiments/methods/my_method.py
import torch
import torch.nn as nn

class MyMethod(nn.Module):
    """
    Implementation of improvement method X from PERFORMANCE_IMPROVEMENTS.md
    """
    def __init__(self, d_model=128, **kwargs):
        super().__init__()
        # Initialize components

    def forward(self, sparse_coords, sparse_values, query_coords, t):
        # Implement method
        return predicted_values

def train_my_method(dataset, **kwargs):
    """Training function"""
    # Training loop
    pass

if __name__ == '__main__':
    # Train on all complexity levels
    pass
```

### Priority Implementation Order

**Phase 1** (Quick wins, 1-2 days each):
1. Content-Aware Sampling (#2)
2. Perceptual Loss (#15)
3. Curriculum Learning (#14)

**Phase 2** (Major improvements, 3-5 days each):
4. Hierarchical MAMBA (#5)
5. Wavelet Sparse (#3)
6. Rectified Flow (#8)

**Phase 3** (Advanced, 1-2 weeks each):
7. Latent Diffusion (#9)
8. Parallel BiMamba (#4)
9. Consistency Models (#7)

---

## 📈 Evaluation Metrics

### Reconstruction Quality
- **PSNR**: Peak Signal-to-Noise Ratio (dB) - higher is better
- **MSE**: Mean Squared Error - lower is better
- **MAE**: Mean Absolute Error - lower is better
- **Correlation**: Pearson correlation with ground truth

### Frequency Domain
- **Frequency Error**: MSE in FFT domain
- **Spectral Similarity**: Compares frequency content

### Perceptual Quality
- **Relative Error**: Normalized error metric
- **Edge Preservation**: Gradient similarity

---

## 🎯 Expected Performance Targets

### Baseline (V1 MAMBA)

| Complexity | PSNR (dB) | MSE | Training Time |
|-----------|-----------|-----|---------------|
| Simple | 35-40 | 0.0001-0.001 | 5 min |
| Multi-frequency | 30-35 | 0.001-0.003 | 7 min |
| Radial | 32-37 | 0.0005-0.002 | 6 min |
| Interference | 28-33 | 0.002-0.005 | 8 min |
| Modulated | 25-30 | 0.003-0.01 | 10 min |
| Composite | 22-28 | 0.005-0.02 | 12 min |

### Target Improvements (Method-specific)

**Content-Aware Sampling (#2)**:
- **Gain**: +3-5 dB across all complexities
- **Speed**: Same (sampling strategy change)
- **Best for**: Interference, Composite (high spatial variation)

**Wavelet Sparse (#3)**:
- **Gain**: +4-6 dB on multi-frequency patterns
- **Speed**: 5-10x faster (sparse wavelet coefficients)
- **Best for**: Multi-frequency, Modulated

**Hierarchical MAMBA (#5)**:
- **Gain**: +5-7 dB across all complexities
- **Speed**: 2-3x faster (multi-scale processing)
- **Best for**: All complexities (universal improvement)

**Latent Diffusion (#9)**:
- **Gain**: +1-2 dB (small images, less compression benefit)
- **Speed**: 8-16x faster (smaller latent space)
- **Best for**: Training speed optimization

---

## 🧪 Ablation Studies

### Sparsity Levels
Test reconstruction quality vs observation density:
```bash
for sparsity in 0.02 0.05 0.1 0.15 0.2; do
    python train_baseline_mamba.py --train_sparsity $sparsity --test_sparsity $sparsity
done
```

**Note**: Both train and test use the same sparsity level but are disjoint sets

### Architecture Variations
- **Depth**: 2, 4, 6, 8 layers
- **Width**: 64, 128, 256, 512 dimensions
- **Fourier features**: 8, 16, 32, 64 frequencies

### Sampling Strategies
- Random uniform
- Uniform grid
- Edge-aware (gradient-based)
- Learned adaptive

---

## 📊 Visualization Examples

### Dataset Samples
```python
from synthetic_experiments.datasets import SinusoidalDataset

dataset = SinusoidalDataset(resolution=32, num_samples=100, complexity='interference')
dataset.visualize_samples(num_samples=6, save_path='samples.png')

# Visualize train/test split
train_coords, train_values, test_coords, test_values, full = dataset.get_train_test_split(
    train_sparsity=0.05, test_sparsity=0.05
)
# train_coords: 5% pixels for training
# test_coords: different 5% pixels for testing (disjoint)
# full: complete 100% ground truth
```

### Reconstruction Comparison
```python
from synthetic_experiments.evaluation import visualize_reconstruction

visualize_reconstruction(
    model=trained_model,
    dataset=test_dataset,
    num_samples=4,
    save_path='reconstruction.png'
)
```

---

## 🔧 Tips for Synthetic Experiments

### 1. Start Simple
- Begin with `simple` complexity, clean data (noise=0.0)
- Verify basic reconstruction works (PSNR > 35 dB)
- Then increase complexity gradually

### 2. Fast Iteration
- Use small models (d_model=64-128, 2-4 layers)
- Train for 50-100 epochs (5-10 minutes)
- Evaluate on 50-100 test samples

### 3. Controlled Comparison
- Keep all hyperparameters identical except the method
- Same datasets, same 5%+5% disjoint sampling strategy
- Same evaluation on full 100% field
- Multiple random seeds for statistical significance

### 4. Debug with Visualizations
- Plot reconstructions to see qualitative differences
- Visualize attention maps, learned features
- Check frequency domain representations

### 5. Scale Up Gradually
- Prove method works on synthetic first
- Then apply to real CIFAR-10 data
- Synthetic → Real transfer validation

---

## 🎯 Sampling Strategy

### 5%+5% Disjoint Sampling

The synthetic experiments use a **disjoint train/test sampling strategy** to test true generalization:

**Training Phase**:
1. Sample random 5% of pixels as **training observations** (e.g., 51 pixels for 32×32)
2. Sample different random 5% as **test observations** (51 different pixels, disjoint from train)
3. Model learns to predict test pixels given train pixels
4. Loss computed only on test pixels (not full field)

**Evaluation Phase**:
- Use same 5% training observations as input
- Reconstruct **full 100% field** (all 1024 pixels for 32×32)
- Compute metrics (PSNR, MSE, etc.) on complete reconstruction vs ground truth

**Why This Strategy?**:
- **True Generalization**: Model never sees test pixels during training
- **Realistic**: Mimics real-world sparse observation scenarios
- **Challenging**: 5% is very sparse (vs 20% in many papers)
- **Fair Comparison**: Same disjoint sets for all methods

**Comparison to Alternatives**:
```
OLD (many papers):  Train on 20% → Reconstruct full 100%
                    (easier, more observations)

NEW (this work):    Train on 5% → Predict different 5% → Evaluate on 100%
                    (harder, tests generalization)
```

---

## 📚 References

### Sinusoidal Pattern Analysis
- **Fourier Analysis**: Understanding frequency decomposition
- **Wave Interference**: Physical wave superposition
- **Modulation Theory**: AM/FM signal processing
- **Radial Basis Functions**: Circular symmetry in learning

### Sparse Reconstruction Theory
- **Compressed Sensing**: Sparse signal recovery
- **Nyquist-Shannon**: Sampling theory fundamentals
- **Bandlimited Signals**: Frequency domain constraints
- **Interpolation**: Reconstruction from samples

---

## 🎉 Expected Outcomes

After completing synthetic experiments, you should be able to:

1. **Rank Methods**: Clear performance ordering for each complexity
2. **Understand Trade-offs**: Quality vs speed vs complexity
3. **Identify Best Practices**: Which methods work for which patterns
4. **Transfer Insights**: Apply learnings to real CIFAR-10 data
5. **Publication**: Comprehensive ablation study results

**Success Criteria**:
- ✅ All 6 complexities tested
- ✅ 3+ improvement methods implemented and compared
- ✅ Statistical significance (multiple seeds)
- ✅ Clear 5-10x improvement over baseline
- ✅ Comprehensive visualizations and tables

---

## 🤝 Contributing

To add a new method:

1. Create `methods/my_method.py` with model and training
2. Add to `evaluation/compare_methods.py` checkpoint dict
3. Run comparison on all complexities
4. Update this README with results

---

## 📞 Quick Commands Reference

```bash
# Generate datasets
python datasets/sinusoidal_generator.py

# Train baseline with 5%+5% sampling
python baselines/train_baseline_mamba.py \
    --complexity simple \
    --epochs 100 \
    --train_sparsity 0.05 \
    --test_sparsity 0.05

# Compare methods (evaluates on full 100% field)
python evaluation/compare_methods.py \
    --complexities simple radial composite \
    --train_sparsity 0.05 \
    --test_sparsity 0.05

# Visualize specific method
python evaluation/visualize.py --method baseline --complexity interference
```

---

**Made with ❤️ for controlled method comparison**
