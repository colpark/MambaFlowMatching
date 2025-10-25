# MAMBA Diffusion Training - Quick Start Guide

## Overview
This directory contains a complete standalone training infrastructure for the MAMBA diffusion model that matches the original `mamba_diffusion.ipynb` exactly.

## Files
- `train_mamba_standalone.py` - Standalone training script (1000 epochs default)
- `run_mamba_training.sh` - Start training in background
- `stop_mamba_training.sh` - Stop running training
- `monitor_training.sh` - Check training status and view logs

## Quick Start

### 1. Start Training
```bash
./run_mamba_training.sh
```

This will:
- Start training in the background (survives terminal/screen closure)
- Save logs to `training_output.log`
- Save PID to `training.pid`
- Train for 1000 epochs by default
- Save checkpoints every 10 epochs

### 2. Monitor Training
```bash
# Check status and view latest output
./monitor_training.sh

# Or watch logs in real-time
tail -f training_output.log
```

### 3. Stop Training
```bash
./stop_mamba_training.sh
```

## Checkpoints
Checkpoints are saved to `checkpoints_mamba/`:
- `mamba_best.pth` - Best validation loss
- `mamba_latest.pth` - Latest epoch
- `mamba_epoch_XXXX.pth` - Every 10 epochs (e.g., 0010, 0020, 0030, ...)

## Configuration
You can customize training by setting environment variables:

```bash
# Train for 500 epochs instead of 1000
EPOCHS=500 ./run_mamba_training.sh

# Use larger batch size
BATCH_SIZE=128 ./run_mamba_training.sh

# Save checkpoints every 20 epochs
SAVE_EVERY=20 ./run_mamba_training.sh

# Combine multiple settings
EPOCHS=2000 BATCH_SIZE=128 SAVE_EVERY=50 ./run_mamba_training.sh
```

Available options:
- `EPOCHS` (default: 1000)
- `BATCH_SIZE` (default: 64)
- `LR` (default: 1e-4)
- `SAVE_EVERY` (default: 10)
- `EVAL_EVERY` (default: 10)
- `VISUALIZE_EVERY` (default: 50)
- `SAVE_DIR` (default: checkpoints_mamba)
- `D_MODEL` (default: 512)
- `NUM_LAYERS` (default: 6)
- `NUM_WORKERS` (default: 4)

## Direct Python Execution
You can also run the training script directly:

```bash
python train_mamba_standalone.py --epochs 1000 --batch_size 64 --save_every 10
```

Use `--help` to see all available options:
```bash
python train_mamba_standalone.py --help
```

## Training Will Continue Even If:
- ✅ You close the terminal
- ✅ You log out of SSH
- ✅ Your screen turns off
- ✅ Your computer goes to sleep
- ✅ You disconnect from WiFi (briefly)

This is because `run_mamba_training.sh` uses `nohup` to run training as a background process.

## Troubleshooting

### Training Already Running
If you see "Training is already running", you have two options:
1. Stop it first: `./stop_mamba_training.sh`
2. Let it continue and monitor: `./monitor_training.sh`

### Check If Training Is Actually Running
```bash
./monitor_training.sh
```

### View Full Logs
```bash
cat training_output.log
```

### Resume From Checkpoint
The script automatically saves checkpoints. To resume from a specific checkpoint, you'll need to modify the training script to load from a checkpoint path (not currently implemented - it always starts fresh).

## Model Architecture
This uses the exact MAMBA architecture from the original notebook:
- **SSM State Space Model** with O(N) complexity (vectorized implementation)
- **Fourier Features** for continuous coordinate encoding
- **Flow Matching** training objective
- **Local Cross-Attention** with KNN search
- **RBF Prior** for smooth interpolation
- **6 MAMBA layers**, 512-dim hidden size (default)

## Dataset
- CIFAR-10 training and test sets
- Sparse-to-dense setup (20% → 100% pixels)
- Morton curve ordering for spatial structure
