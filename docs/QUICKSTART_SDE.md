# SDE Sampling - Quick Start

## One Command

```bash
./eval_sde.sh
```

Done! This compares **ODE vs SDE vs DDIM** at 32x32, 64x64, and 96x96 resolutions.

---

## View Results

```bash
# Check which sampler is best
cat eval_sde_multiscale/metrics_comparison.txt | grep "PSNR"

# View images (macOS)
open eval_sde_multiscale/*.png

# View images (Linux)
xdg-open eval_sde_multiscale/*.png
```

---

## Common Scenarios

### Test SDE Only (Recommended)

```bash
SAMPLERS="sde" ./eval_sde.sh
```

### Smoother Backgrounds (Less Speckles)

```bash
TEMPERATURE=0.7 ./eval_sde.sh
```

### More Detail (Less Smoothing)

```bash
TEMPERATURE=0.3 ./eval_sde.sh
```

### Higher Quality (More Steps)

```bash
NUM_STEPS=100 ./eval_sde.sh
```

### More Samples (Better Statistics)

```bash
NUM_SAMPLES=50 ./eval_sde.sh
```

### Faster Evaluation (DDIM)

```bash
SAMPLERS="ddim" NUM_STEPS=30 ./eval_sde.sh
```

---

## What to Expect

### Good Results

**SDE should show**:
- ✅ Smoother backgrounds
- ✅ Less speckled artifacts
- ✅ PSNR: +1-2 dB over Heun
- ✅ SSIM: +0.02-0.05 over Heun

**Visual comparison**: Look at `multiscale_comparison.png`
- Left columns: Heun (noisy)
- Middle columns: SDE (smooth)
- Right columns: DDIM (fast)

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Still noisy | `TEMPERATURE=0.7 ./eval_sde.sh` |
| Too blurry | `TEMPERATURE=0.3 ./eval_sde.sh` |
| Out of memory | `DEVICE=cpu ./eval_sde.sh` |
| Takes too long | `NUM_SAMPLES=5 ./eval_sde.sh` |

---

## Full Docs

For detailed information, see:
- [README_SDE.md](README_SDE.md) - Complete guide
- [SUMMARY_SDE.md](SUMMARY_SDE.md) - Technical summary
