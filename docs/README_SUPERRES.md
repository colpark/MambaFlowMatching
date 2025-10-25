# Super-Resolution Evaluation

Comprehensive evaluation of MAMBA Diffusion model's zero-shot super-resolution capacity.

## Overview

This evaluation script tests the model's ability to generate high-resolution images from sparse 32x32 input data. The model is trained on 32x32 CIFAR-10 images with 20% sparse observations and evaluated on its ability to generate outputs at:

- **64x64** (2x upscaling)
- **96x96** (3x upscaling)
- **128x128** (4x upscaling)
- **256x256** (8x upscaling)

## Key Features

- **Zero-shot Super-Resolution**: Model generates higher resolution outputs without being explicitly trained for super-resolution
- **Sparse Input**: Uses only 20% of 32x32 pixels as conditioning
- **Multiple Resolutions**: Evaluates at 4 different target resolutions
- **Comprehensive Metrics**: Computes PSNR, SSIM, MSE, and MAE for each resolution
- **Visual Comparison**: Generates side-by-side visualizations

## Quick Start

### Basic Usage

```bash
# Evaluate with default settings (best checkpoint)
./eval_superres.sh

# Evaluate specific checkpoint
CHECKPOINT=checkpoints_mamba/mamba_epoch_0100.pth ./eval_superres.sh

# Evaluate more samples
NUM_SAMPLES=20 ./eval_superres.sh

# Use CPU instead of GPU
DEVICE=cpu ./eval_superres.sh
```

### Advanced Usage

```bash
# Custom configuration
CHECKPOINT=checkpoints_mamba/mamba_best.pth \
RESOLUTIONS="64 128 256" \
NUM_SAMPLES=25 \
NUM_STEPS=100 \
SAVE_DIR=eval_highquality \
./eval_superres.sh
```

### Python Direct Usage

```bash
# Full control with Python
python eval_superresolution.py \
    --checkpoint checkpoints_mamba/mamba_best.pth \
    --resolutions 64 96 128 256 \
    --num_samples 10 \
    --num_steps 50 \
    --save_dir eval_superres \
    --device auto
```

## Configuration Options

### Bash Script (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `CHECKPOINT` | `checkpoints_mamba/mamba_best.pth` | Path to model checkpoint |
| `RESOLUTIONS` | `64 96 128 256` | Target resolutions (space-separated) |
| `NUM_SAMPLES` | `10` | Number of test samples to evaluate |
| `NUM_STEPS` | `50` | Number of sampling steps (higher = better quality) |
| `BATCH_SIZE` | `1` | Batch size for evaluation |
| `SAVE_DIR` | `eval_superres` | Directory to save results |
| `D_MODEL` | `512` | Model dimension (must match checkpoint) |
| `NUM_LAYERS` | `6` | Number of layers (must match checkpoint) |
| `DEVICE` | `auto` | Device: `auto`, `cuda`, or `cpu` |

### Python Script Arguments

```bash
python eval_superresolution.py --help
```

```
--checkpoint PATH      Path to model checkpoint (required)
--resolutions N [N...] Target resolutions (default: 64 96 128 256)
--num_samples N        Number of samples (default: 10)
--num_steps N          Sampling steps (default: 50)
--batch_size N         Batch size (default: 1)
--save_dir DIR         Save directory (default: eval_superres)
--device DEVICE        Device: auto/cuda/cpu (default: auto)
--d_model N            Model dimension (default: 512)
--num_layers N         Number of layers (default: 6)
```

## Output Files

After evaluation, the following files are created in the save directory:

```
eval_superres/
├── metrics.txt                          # Detailed metrics report
├── superresolution_comparison.png       # Side-by-side comparison (all resolutions)
├── superres_64x64.png                   # Individual 64x64 results
├── superres_96x96.png                   # Individual 96x96 results
├── superres_128x128.png                 # Individual 128x128 results
└── superres_256x256.png                 # Individual 256x256 results
```

### Metrics File Format

```
Super-Resolution Evaluation Results
Generated: 2024-01-XX HH:MM:SS
============================================================

AVERAGE METRICS
------------------------------------------------------------

64x64 Resolution:
  MSE: 0.001234 ± 0.000123
  MAE: 0.023456 ± 0.002345
  PSNR: 28.45 ± 2.34 dB
  SSIM: 0.8567 ± 0.0234

96x96 Resolution:
  ...

[Additional resolutions...]

============================================================

PER-SAMPLE METRICS
------------------------------------------------------------
[Detailed per-sample results...]
```

## Understanding the Results

### Metrics Explained

1. **PSNR (Peak Signal-to-Noise Ratio)**
   - Higher is better
   - Typical range: 20-40 dB
   - >30 dB: Good quality
   - >35 dB: Excellent quality

2. **SSIM (Structural Similarity Index)**
   - Range: 0-1 (higher is better)
   - >0.9: Excellent structural similarity
   - >0.8: Good structural similarity
   - >0.7: Acceptable structural similarity

3. **MSE (Mean Squared Error)**
   - Lower is better
   - Measures pixel-wise error
   - Range: 0-1 for normalized images

4. **MAE (Mean Absolute Error)**
   - Lower is better
   - Measures average pixel difference
   - Range: 0-1 for normalized images

### Expected Performance

Performance typically degrades with higher upscaling factors:

- **64x64 (2x)**: Best performance, highest PSNR/SSIM
- **96x96 (3x)**: Good performance
- **128x128 (4x)**: Moderate performance
- **256x256 (8x)**: Most challenging, lower metrics

## Examples

### Evaluate Best Model

```bash
# Quick evaluation
./eval_superres.sh

# Expected output:
# ✓ 64x64: PSNR=30.5dB, SSIM=0.87
# ✓ 96x96: PSNR=28.2dB, SSIM=0.83
# ✓ 128x128: PSNR=26.8dB, SSIM=0.79
# ✓ 256x256: PSNR=24.3dB, SSIM=0.72
```

### High-Quality Evaluation

```bash
# More samples, more steps
NUM_SAMPLES=50 NUM_STEPS=100 ./eval_superres.sh
```

### Single Resolution Focus

```bash
# Only evaluate 256x256
RESOLUTIONS="256" NUM_SAMPLES=20 ./eval_superres.sh
```

### Compare Multiple Checkpoints

```bash
# Evaluate epoch 100
CHECKPOINT=checkpoints_mamba/mamba_epoch_0100.pth \
SAVE_DIR=eval_epoch100 \
./eval_superres.sh

# Evaluate epoch 500
CHECKPOINT=checkpoints_mamba/mamba_epoch_0500.pth \
SAVE_DIR=eval_epoch500 \
./eval_superres.sh

# Compare results
diff eval_epoch100/metrics.txt eval_epoch500/metrics.txt
```

## Technical Details

### Input Configuration

- **Training Resolution**: 32x32 (CIFAR-10)
- **Sparse Input**: 20% of pixels (204/1024 pixels)
- **Coordinate Encoding**: Fourier features with normalized [0,1] coordinates
- **Color Space**: RGB in [0,1] range

### Sampling Process

1. Create dense coordinate grid at target resolution
2. Use sparse 32x32 input as conditioning (20% pixels)
3. Run Heun ODE solver for flow matching
4. Generate RGB values for all target coordinates
5. Reshape to target resolution image

### Comparison Baseline

Ground truth is created by bicubic upscaling the original 32x32 image to the target resolution. This provides a fair comparison since:
- The model only sees 20% of 32x32 pixels
- Ground truth represents the best bicubic reconstruction
- Metrics measure how well the model reconstructs vs. interpolation

## Troubleshooting

### Checkpoint Not Found

```bash
# List available checkpoints
ls -lh checkpoints_mamba/

# Specify correct path
CHECKPOINT=checkpoints_mamba/mamba_best.pth ./eval_superres.sh
```

### Out of Memory

```bash
# Reduce batch size or use CPU
BATCH_SIZE=1 ./eval_superres.sh
# or
DEVICE=cpu ./eval_superres.sh
```

### Slow Evaluation

```bash
# Reduce samples or steps
NUM_SAMPLES=5 NUM_STEPS=25 ./eval_superres.sh
```

### Model Architecture Mismatch

```bash
# Specify correct model parameters
D_MODEL=256 NUM_LAYERS=4 ./eval_superres.sh
```

## Integration with Training

### Evaluate During Training

```bash
# In one terminal: train
./run_mamba_training.sh

# In another terminal: evaluate periodically
watch -n 3600 'CHECKPOINT=checkpoints_mamba/mamba_latest.pth ./eval_superres.sh'
```

### Automated Evaluation Pipeline

```bash
#!/bin/bash
# evaluate_all_checkpoints.sh

for checkpoint in checkpoints_mamba/mamba_epoch_*.pth; do
    epoch=$(basename $checkpoint .pth | grep -o '[0-9]*')
    echo "Evaluating epoch $epoch"
    CHECKPOINT=$checkpoint \
    SAVE_DIR=eval_epoch_${epoch} \
    NUM_SAMPLES=10 \
    ./eval_superres.sh
done

# Summarize results
echo "Summary of all evaluations:"
grep "PSNR" eval_epoch_*/metrics.txt
```

## References

- **Training Script**: `train_mamba_standalone.py`
- **Training Runner**: `run_mamba_training.sh`
- **Dataset**: CIFAR-10 with sparse sampling (`core/sparse/cifar10_sparse.py`)
- **Model**: MAMBA Diffusion with flow matching

## Citation

If you use this evaluation in your research, please cite the relevant papers on:
- Flow Matching for generative modeling
- MAMBA architecture for sequence modeling
- Neural fields for continuous representations
