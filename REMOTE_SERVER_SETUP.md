# Remote Server Setup Instructions

## ✅ Issue Fixed

The `ModuleNotFoundError: No module named 'core'` has been resolved in commit `07e810e`.

## 🚀 Setup on Remote Server

### Step 1: Pull Latest Changes

```bash
cd ~/MambaFlowMatching  # or wherever you cloned the repo
git pull origin main
```

### Step 2: Verify Imports Work

```bash
python test_imports.py
```

Expected output:
```
✓ core.neural_fields.perceiver imported
✓ core.sparse.cifar10_sparse imported
✓ core.sparse.metrics imported
✓ V1 train_mamba_standalone components imported
✅ All imports working correctly!
```

### Step 3: Train V2 Model

```bash
cd v2/training
./run_mamba_v2_training.sh
```

### Step 4: Monitor Training

```bash
tail -f training_v2_output.log
```

## 🔧 What Was Fixed

### Before (Broken)
```python
# v2/training/train_mamba_v2.py
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # Only gets v2/
sys.path.insert(0, parent_dir)  # Can't find core/ from v2/
```

### After (Working)
```python
# v2/training/train_mamba_v2.py
script_dir = os.path.dirname(os.path.abspath(__file__))  # v2/training/
v2_dir = os.path.dirname(script_dir)                      # v2/
repo_root = os.path.dirname(v2_dir)                       # MambaFlowMatching/
sys.path.insert(0, repo_root)  # ✓ Can now find core/
```

## 📁 Repository Structure

```
MambaFlowMatching/          ← repo_root added to sys.path
├── core/                   ← Now accessible from any script
│   ├── neural_fields/
│   └── sparse/
├── v1/
│   ├── training/
│   │   └── train_mamba_standalone.py  ← Fixed
│   └── evaluation/
│       └── eval_*.py                  ← All fixed
└── v2/
    ├── training/
    │   └── train_mamba_v2.py          ← Fixed
    └── evaluation/
        └── eval_v1_vs_v2.py           ← Fixed
```

## 🎯 Training Commands

### V1 (Baseline)
```bash
cd ~/MambaFlowMatching/v1/training
./run_mamba_training.sh

# Monitor
./monitor_training.sh
```

### V2 (Improved)
```bash
cd ~/MambaFlowMatching/v2/training
./run_mamba_v2_training.sh

# Monitor
tail -f training_v2_output.log
```

### Custom Configuration
```bash
# V2 with custom settings
cd ~/MambaFlowMatching/v2/training
D_MODEL=256 NUM_LAYERS=8 BATCH_SIZE=32 ./run_mamba_v2_training.sh
```

## 📊 Expected Behavior

After pulling the latest changes, training should start successfully:

```
============================================================
MAMBA Diffusion V2 Training Runner
============================================================
...
V2 Improvements:
  ✓ Bidirectional MAMBA (4 forward + 4 backward = 8 total layers)
  ✓ Lightweight Perceiver (2 iterations)
  ✓ Query self-attention for spatial coherence
...
Initializing MAMBA Diffusion V2:
  d_model: 256
  num_layers: 8 (total MAMBA layers)
  Bidirectional MAMBA: 4 forward + 4 backward layers
  Lightweight Perceiver: 2 iterations, 8 heads
...
Epoch 1/1000: 100%|██████████| 782/782 [XX:XX<XX:XX, X.XXs/it, loss=X.XXX]
```

## 🆘 Troubleshooting

### Still Getting Import Errors?

1. **Verify you pulled latest changes:**
   ```bash
   git log --oneline -1
   # Should show: 07e810e Add import test script for verification
   ```

2. **Check Python can find modules:**
   ```bash
   cd ~/MambaFlowMatching
   python -c "import sys; print(sys.path)"
   python test_imports.py
   ```

3. **Verify file structure:**
   ```bash
   ls -la core/
   ls -la v2/training/
   ```

### Dependencies Missing?

```bash
pip install -r requirements.txt
```

### CUDA/GPU Issues?

```bash
# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Run on CPU if needed
DEVICE=cpu ./run_mamba_v2_training.sh
```

## 📈 Training Progress

Training outputs:
- **Checkpoints**: `checkpoints_mamba_v2/`
  - `mamba_v2_best.pth` - Best validation loss
  - `mamba_v2_latest.pth` - Latest epoch
  - `mamba_v2_epoch_XXXX.pth` - Every 10 epochs

- **Logs**: `training_v2_output.log`
- **Visualizations**: Saved every 50 epochs

## 🎯 After Training

Evaluate super-resolution:
```bash
cd ~/MambaFlowMatching/v1/evaluation
./eval_superres.sh
```

Compare V1 vs V2:
```bash
cd ~/MambaFlowMatching/v2/evaluation
python eval_v1_vs_v2.py \
    --v1_checkpoint ../../v1/training/checkpoints_mamba/mamba_best.pth \
    --v2_checkpoint ../training/checkpoints_mamba_v2/mamba_v2_best.pth \
    --num_samples 20
```

## 📞 Success Indicators

You'll know it's working when you see:
- ✅ No import errors
- ✅ Model initialization messages
- ✅ Progress bar showing epoch completion
- ✅ Loss values decreasing
- ✅ Checkpoints being saved
- ✅ Visualizations generated every 50 epochs

---

**Commit**: `07e810e` - All import issues resolved
**Status**: Ready to train 🚀
