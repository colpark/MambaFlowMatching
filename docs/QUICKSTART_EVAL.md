# Super-Resolution Evaluation - Quick Start

## One-Line Evaluation

```bash
./eval_superres.sh
```

That's it! This will:
- Load the best trained checkpoint
- Evaluate on 10 test samples
- Generate outputs at 64x64, 96x96, 128x128, and 256x256
- Save metrics and visualizations to `eval_superres/`

## View Results

```bash
# View metrics
cat eval_superres/metrics.txt

# View images (macOS)
open eval_superres/*.png

# View images (Linux)
xdg-open eval_superres/*.png
```

## Common Scenarios

### Use Different Checkpoint

```bash
CHECKPOINT=checkpoints_mamba/mamba_epoch_0500.pth ./eval_superres.sh
```

### More Samples for Better Statistics

```bash
NUM_SAMPLES=50 ./eval_superres.sh
```

### Higher Quality (More Steps)

```bash
NUM_STEPS=100 ./eval_superres.sh
```

### Focus on Specific Resolution

```bash
RESOLUTIONS="256" ./eval_superres.sh
```

### CPU Mode (No GPU)

```bash
DEVICE=cpu ./eval_superres.sh
```

## Understanding Output

The script evaluates how well the model can generate high-resolution images from only 20% of 32x32 pixels:

- **Input**: 204 sparse pixels from 32x32 image (20%)
- **Output**: Full dense images at 64x64, 96x96, 128x128, 256x256
- **Metrics**: PSNR (higher better), SSIM (closer to 1 better)

### Good Results Look Like:

- **64x64**: PSNR > 28 dB, SSIM > 0.85
- **96x96**: PSNR > 26 dB, SSIM > 0.80
- **128x128**: PSNR > 24 dB, SSIM > 0.75
- **256x256**: PSNR > 22 dB, SSIM > 0.70

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Checkpoint not found" | Run training first: `./run_mamba_training.sh` |
| "Out of memory" | Use CPU: `DEVICE=cpu ./eval_superres.sh` |
| Takes too long | Reduce samples: `NUM_SAMPLES=5 ./eval_superres.sh` |

## Full Documentation

For detailed options and advanced usage, see [README_SUPERRES.md](README_SUPERRES.md)
