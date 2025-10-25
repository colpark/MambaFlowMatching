# MAMBA Flow Matching for Sparse Neural Fields

State space models (MAMBA) combined with flow matching for high-quality image generation from sparse (20%) pixel observations. Supports zero-shot super-resolution at arbitrary scales.

## 🎯 Key Features

- **MAMBA Architecture**: Linear-complexity state space models for efficient sequence processing
- **Flow Matching**: Continuous normalizing flows for high-quality generation
- **Sparse Training**: Learn from only 20% of pixels with deterministic masking
- **Zero-Shot Super-Resolution**: Generate at 64×, 96×, 128×, 256× without training at those resolutions
- **Four Architectures**: V1 (MAMBA baseline), V2 (bidirectional + perceiver), V3 (Morton curves), V4 (Transformer comparison)

## 📊 Results

### V1 (Baseline)
- Unidirectional MAMBA with 6 layers
- Single cross-attention layer
- Row-major sequence ordering
- PSNR: ~28 dB, SSIM: ~0.85

### V2 (Complex - Bidirectional + Perceiver)
- **Bidirectional MAMBA**: 4 forward + 4 backward = 8 layers
- **Lightweight Perceiver**: Query self-attention for spatial coherence
- **Expected improvements**:
  - 70-80% reduction in background speckles
  - +3-5 dB PSNR improvement
  - Smoother, more coherent spatial fields
- **Trade-off**: +71% computational cost

### V3 (Clean - Morton Curves)
- **Same architecture as V1** (6 layers, same parameters)
- **Morton (Z-order) curve**: Better spatial locality in sequences
- **Expected improvements**:
  - Better spatial coherence
  - Reduced artifacts from spatially-aware processing
  - +1-2 dB PSNR improvement
- **Trade-off**: Zero additional cost!

### V4 (Comparison - Transformer)
- **Standard Transformer encoder** instead of MAMBA
- **Multi-head self-attention**: Global context vs sequential state
- **Purpose**: Benchmark MAMBA's linear O(N) vs Transformer's quadratic O(N²)
- **Expected**:
  - 10-20x slower training than V1
  - Tests if global attention helps sparse neural fields
  - Fair comparison with same depth and dimension
- **Trade-off**: Much higher computational cost

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/MambaFlowMatching.git
cd MambaFlowMatching

# Install dependencies
pip install -r requirements.txt
```

### Training

**V1 (Baseline):**
```bash
cd v1/training
./run_mamba_training.sh
```

**V2 (Bidirectional + Perceiver):**
```bash
cd v2/training
./run_mamba_v2_training.sh
```

**V3 (Morton Curves - Recommended):**
```bash
cd v3/training
./run_mamba_v3_training.sh
```

**V4 (Transformer Comparison):**
```bash
cd v4/training
./run_transformer_v4_training.sh
```

### Evaluation

**Super-Resolution Evaluation:**
```bash
cd v1/evaluation
./eval_superres.sh  # Tests 64×, 96×, 128×, 256× resolutions
```

**V1 vs V2 Comparison:**
```bash
cd v2/evaluation
python eval_v1_vs_v2.py \
    --v1_checkpoint ../../v1/training/checkpoints_mamba/mamba_best.pth \
    --v2_checkpoint ../training/checkpoints_mamba_v2/mamba_v2_best.pth \
    --num_samples 20
```

## 📁 Repository Structure

```
MambaFlowMatching/
├── core/                           # Core modules
│   ├── neural_fields/             # Fourier features, perceiver components
│   ├── sparse/                    # Sparse dataset handling, metrics
│   └── diffusion/                 # Flow matching utilities
│
├── v1/                            # V1 Architecture (Baseline)
│   ├── training/                  # Training scripts
│   │   ├── train_mamba_standalone.py
│   │   ├── run_mamba_training.sh
│   │   ├── monitor_training.sh
│   │   └── stop_mamba_training.sh
│   └── evaluation/                # Evaluation scripts
│       ├── eval_superresolution.py
│       ├── eval_superres.sh
│       ├── eval_sde_multiscale.py
│       └── eval_sde.sh
│
├── v2/                            # V2 Architecture (Bidirectional)
│   ├── training/                  # Training scripts
│   │   ├── train_mamba_v2.py
│   │   └── run_mamba_v2_training.sh
│   └── evaluation/                # Evaluation scripts
│       └── eval_v1_vs_v2.py
│
├── v3/                            # V3 Architecture (Morton Curves)
│   ├── training/                  # Training scripts
│   │   ├── train_mamba_v3_morton.py
│   │   └── run_mamba_v3_training.sh
│   └── evaluation/                # Evaluation scripts (TBD)
│
├── v4/                            # V4 Architecture (Transformer)
│   ├── training/                  # Training scripts
│   │   ├── train_transformer_v4.py
│   │   └── run_transformer_v4_training.sh
│   └── evaluation/                # Evaluation scripts (TBD)
│
├── docs/                          # Documentation
│   ├── README.md                  # Original documentation
│   ├── README_V2.md              # V2 architecture details
│   ├── README_V3.md              # V3 Morton curves guide
│   ├── README_V4.md              # V4 Transformer comparison guide
│   ├── README_SUPERRES.md        # Super-resolution guide
│   ├── README_SDE.md             # SDE sampling guide
│   ├── QUICKSTART_EVAL.md        # Quick evaluation guide
│   ├── QUICKSTART_SDE.md         # Quick SDE guide
│   └── TRAINING_README.md        # Training guide
│
├── scripts/                       # Utility scripts
│   ├── remote_setup.sh           # Remote server setup
│   └── verify_deterministic_masking.py
│
└── requirements.txt              # Python dependencies
```

## 🔬 Architecture Details

### V1 Architecture
```
Input Coordinates → Fourier Features → MAMBA (6 layers) → Cross-Attention → Decoder → Output
```

- **MAMBA**: 6 unidirectional layers (left → right)
- **Cross-Attention**: Single layer for input-query interaction
- **d_model**: 512 (default)
- **Parameters**: ~15M

### V2 Architecture
```
Input Coordinates → Fourier Features → Bidirectional MAMBA (8 layers) → Lightweight Perceiver → Decoder → Output
```

- **Bidirectional MAMBA**: 4 forward + 4 backward = 8 layers
- **Lightweight Perceiver**: 2 iterations with query self-attention
- **d_model**: 256 (default)
- **Parameters**: ~7M (53% fewer than V1)

**Key V2 Improvements:**
1. **Bidirectional Context**: Every pixel sees information from both directions
2. **Query Self-Attention**: Neighboring query pixels communicate for spatial coherence
3. **Iterative Refinement**: 2-iteration perceiver for coarse-to-fine processing

### V3 Architecture
```
Input Coordinates → Fourier Features → Morton Reorder → MAMBA (6 layers) → Restore Order → Cross-Attention → Decoder → Output
```

- **MAMBA**: 6 unidirectional layers (same as V1)
- **Morton Curves**: Z-order sequencing for better spatial locality
- **d_model**: 512 (same as V1)
- **Parameters**: ~15M (identical to V1)

**Key V3 Improvements:**
1. **Better Spatial Locality**: Neighbors in 2D are also neighbors in 1D sequence
2. **Zero Extra Cost**: Same computational complexity as V1
3. **Clean Improvement**: Only sequence ordering changes, architecture unchanged

### V4 Architecture
```
Input Coordinates → Fourier Features → Positional Encoding → Transformer (6 layers) → Cross-Attention → Decoder → Output
```

- **Transformer**: 6 layers with multi-head self-attention
- **Positional Encoding**: Sinusoidal position embeddings
- **d_model**: 512 (same as V1)
- **Parameters**: ~15M (similar to V1)
- **Complexity**: O(N²) quadratic attention

**Key V4 Purpose:**
1. **Benchmark MAMBA**: Compare linear vs quadratic complexity in practice
2. **Global vs Sequential**: Test if full attention helps sparse neural fields
3. **Fair Comparison**: Same depth and dimension as V1
4. **Research Baseline**: Standard architecture for comparison

### Architecture Comparison

| Feature | V1 | V2 | V3 | V4 |
|---------|----|----|-----|----|
| **Encoder** | MAMBA | Bidirectional MAMBA | MAMBA | Transformer |
| **Attention** | 1 cross-attn | Perceiver + self-attn | 1 cross-attn | Self + Cross |
| **Ordering** | Row-major | Row-major | Morton curve | Row-major |
| **Complexity** | O(N) | O(N) | O(N) | O(N²) |
| **d_model** | 512 | 256 | 512 | 512 |
| **Layers** | 6 | 8 | 6 | 6 |
| **Parameters** | 15M | 7M | 15M | 15M |
| **Compute Cost** | 1.0x | 1.7x | 1.0x | 10-20x |
| **Philosophy** | Baseline | Architectural | Ordering | Comparison |

## 📊 Training Configuration

### Default Parameters

**V1:**
```bash
d_model=512
num_layers=6
batch_size=64
lr=1e-4
epochs=1000
```

**V2:**
```bash
d_model=256
num_layers=8  # 4 forward + 4 backward
batch_size=64
lr=1e-4
epochs=1000
perceiver_iterations=2
perceiver_heads=8
```

**V3:**
```bash
d_model=512          # Same as V1
num_layers=6         # Same as V1
batch_size=64
lr=1e-4
epochs=1000
morton_ordering=True # NEW: Enabled by default
```

**V4:**
```bash
d_model=512              # Same as V1
num_layers=6             # Same as V1
num_heads=8              # Multi-head attention
dim_feedforward=2048     # FFN dimension
batch_size=64
lr=1e-4
epochs=1000
```

### Dataset

- **CIFAR-10**: 32×32 RGB images
- **Sparse Sampling**: 20% of pixels randomly selected (deterministic per image)
- **Train/Val Split**: Standard CIFAR-10 split

## 🎨 Sampling Methods

Three sampling methods supported:

1. **Heun ODE Solver** (default):
   - Second-order accuracy
   - Deterministic
   - Good baseline quality

2. **SDE Sampling**:
   - Adds Langevin dynamics
   - Stochastic exploration
   - Temperature-controlled noise

3. **DDIM Sampling**:
   - Non-uniform timestep schedule
   - Faster convergence
   - Configurable stochasticity (eta)

## 📈 Evaluation Metrics

- **PSNR** (Peak Signal-to-Noise Ratio): Measures reconstruction quality
- **SSIM** (Structural Similarity Index): Perceptual similarity metric
- **MSE** (Mean Squared Error): Pixel-wise error
- **MAE** (Mean Absolute Error): Average absolute difference

## 🔧 Advanced Usage

### Custom Training Configuration

```bash
# V1 with custom settings
cd v1/training
D_MODEL=512 NUM_LAYERS=6 BATCH_SIZE=32 LR=5e-5 ./run_mamba_training.sh

# V2 with custom settings
cd v2/training
D_MODEL=256 NUM_LAYERS=8 BATCH_SIZE=32 LR=5e-5 ./run_mamba_v2_training.sh
```

### Monitor Training Progress

```bash
# V1
cd v1/training
./monitor_training.sh

# V2
cd v2/training
tail -f training_v2_output.log
```

### Stop Training

```bash
# V1
cd v1/training
./stop_mamba_training.sh

# V2
cd v2/training
kill $(cat training_v2.pid)
```

## 📚 Documentation

See the `docs/` directory for detailed documentation:

- **README_V2.md**: Comprehensive V2 architecture guide with design decisions
- **README_V3.md**: V3 Morton curves implementation and spatial locality
- **README_V4.md**: V4 Transformer comparison and complexity analysis
- **README_SUPERRES.md**: Super-resolution evaluation guide
- **README_SDE.md**: SDE and DDIM sampling methods
- **QUICKSTART_EVAL.md**: Quick evaluation reference
- **QUICKSTART_SDE.md**: Quick SDE sampling reference
- **TRAINING_README.md**: Detailed training guide

## 🐛 Troubleshooting

### Out of Memory
```bash
# Reduce batch size or model dimension
D_MODEL=128 BATCH_SIZE=32 ./run_mamba_training.sh
```

### Slow Training
```bash
# Reduce number of workers or model size
NUM_WORKERS=2 D_MODEL=256 ./run_mamba_training.sh
```

### Noisy Results with V1
- This is expected! Try V3 (Morton curves) for better spatial coherence with zero extra cost
- Or use V2 architecture for cleaner results through bidirectional processing

### V4 Training Too Slow
- V4 is 10-20x slower than V1 due to quadratic O(N²) attention
- This is expected and by design for comparison purposes
- Reduce batch size or use V1/V3 for faster training

## 🤝 Contributing

Contributions welcome! Please open issues or pull requests.

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **MAMBA**: Gu & Dao (2023) - Mamba: Linear-Time Sequence Modeling
- **Transformer**: Vaswani et al. (2017) - Attention Is All You Need
- **Flow Matching**: Lipman et al. (2023) - Flow Matching for Generative Modeling
- **Perceiver**: Jaegle et al. (2021) - Perceiver: General Perception with Iterative Attention
- **Neural Fields**: Tancik et al. (2020) - Fourier Features Let Networks Learn High Frequency Functions
- **Morton Curves**: Morton, G.M. (1966) - A Computer Oriented Geodetic Data Base

## 📞 Contact

For questions or issues, please open a GitHub issue.

---

**Made with ❤️ for sparse neural field generation**
