# Critical Fixes Applied to Address 6 dB PSNR Issue

## Problem Identified

Both V1 (MAMBA) and V2 (Transformer) were capped at ~6 dB PSNR, which indicates models were predicting constant value ~0.5 (mean) instead of learning spatial patterns.

**Root Cause**: Training/evaluation distribution mismatch - model trained on fixed 51 test pixels but evaluated on full 1024 pixels.

---

## Fixes Applied

### Fix 1: Train on Random Query Points ✅ CRITICAL

**File**: `train_improved.py`
**Lines**: 114-139

**Problem**:
- Training: Model learned to predict 51 specific test pixel positions
- Evaluation: Model had to predict ALL 1024 pixel positions
- 973 positions were completely unseen during training
- Model defaulted to predicting mean (~0.5) for unseen positions

**Solution**:
```python
# Create full coordinate grid
H, W = dataset.resolution, dataset.resolution
y_grid, x_grid = torch.meshgrid(
    torch.linspace(-1, 1, H),
    torch.linspace(-1, 1, W),
    indexing='ij'
)
all_coords = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1).to(device)

# Sample random 200 query points each iteration
num_query = 200
query_indices = torch.randperm(N_total, device=device)[:num_query]
test_coords_batch = all_coords[query_indices].unsqueeze(0).expand(actual_batch_size, -1, -1)
test_values_batch = full_data_batch.view(actual_batch_size, N_total, 1)[:, query_indices, :]
```

**Expected Gain**: +20-25 dB PSNR

**Impact**: Model now sees diverse positions during training, learns spatial patterns across full field, can generalize to any query position at test time.

---

### Fix 2: Clamp ODE Outputs ✅ IMPORTANT

**File**: `improved_transformer.py`
**Lines**: 354-356

**Problem**:
- ODE integration can diverge outside [0, 1] data range
- Unbounded predictions compared against [0, 1] targets
- Large errors inflate MSE, reduce PSNR

**Solution**:
```python
# After ODE integration completes
z_t = torch.clamp(z_t, 0.0, 1.0)
```

**Expected Gain**: +2-3 dB PSNR

**Impact**: Prevents ODE from diverging, ensures predictions stay in valid range.

---

### Fix 3: Learnable Fourier Features ✅ OPTIMIZATION

**File**: `improved_transformer.py`
**Line**: 147

**Problem**:
- Fourier frequencies were random and fixed: `self.register_buffer('B', ...)`
- Random frequencies may not align with data structure
- Cannot adapt to data during training

**Solution**:
```python
# Changed from buffer to parameter
self.B = nn.Parameter(torch.randn(2, num_frequencies) * 10.0)
```

**Expected Gain**: +1-2 dB PSNR

**Impact**: Network can learn optimal frequencies for encoding spatial coordinates, better representation of data structure.

---

## Expected Performance

| Configuration | PSNR | Explanation |
|---------------|------|-------------|
| **Before fixes** | ~6 dB | Predicting mean (broken) |
| **After Fix 1** | ~25-30 dB | Learning spatial structure |
| **After Fix 1+2** | ~27-33 dB | + Bounded outputs |
| **After Fix 1+2+3** | ~28-35 dB | + Optimized encoding |
| **+ Improvement techniques** | ~40-50 dB | v3-v102 combinations |

---

## Verification Plan

### Step 1: Quick Sanity Test (Recommended)

Train a single version with fixes for 50 epochs:

```bash
cd synthetic_experiments/improvements

python3 train_improved.py \
    --techniques "" \
    --version 999 \
    --epochs 50 \
    --num_samples 100 \
    --resolution 32 \
    --gpu_id 0 \
    --save_dir checkpoints_test
```

**Expected result**: PSNR should be 20-30 dB (not 6 dB!)

### Step 2: Overfit Test (Diagnostic)

Verify model can memorize single sample:

```bash
python3 train_improved.py \
    --techniques "" \
    --version 998 \
    --epochs 500 \
    --num_samples 1 \
    --resolution 32 \
    --gpu_id 0 \
    --save_dir checkpoints_overfit
```

**Expected result**: PSNR > 40 dB (near-perfect fit)

### Step 3: Full Experiment Suite

If sanity tests pass, run all 100 versions:

```bash
./run_all.sh
```

---

## Code Changes Summary

### Modified Files

1. **`train_improved.py`**
   - Lines 114-139: Random query point sampling
   - Changed from fixed test set to random subset of full field

2. **`improved_transformer.py`**
   - Line 147: Learnable Fourier features (buffer → parameter)
   - Lines 354-356: Output clamping in sampling function

### Backward Compatibility

✅ Changes are backward compatible:
- Existing checkpoints will load (Fourier `B` dimension unchanged)
- Same model architecture (no new layers)
- Same hyperparameters (only training procedure changed)

⚠️ Note: Models trained with old code will still show ~6 dB PSNR because they learned wrong distribution. Need to retrain.

---

## Understanding the Fixes

### Why Fix 1 is Critical

**Before**:
```
Training data:
  Input: 51 observed pixels at positions [0, 5, 10, 15, ...]
  Output: 51 test pixels at positions [1, 6, 11, 16, ...]

Evaluation:
  Input: Same 51 observed pixels
  Output: ALL 1024 pixels (including 973 never-seen positions)

Result: Model has no idea what to predict for unseen positions → predicts mean
```

**After**:
```
Training data (each iteration different):
  Input: 51 observed pixels
  Output: Random 200 pixels (e.g., iteration 1: [5, 89, 234, ...])
         Next iteration: [12, 456, 789, ...] (completely different!)

Evaluation:
  Input: 51 observed pixels
  Output: ALL 1024 pixels (but model has seen diverse positions during training)

Result: Model learns spatial interpolation, can predict any position
```

### Why Fix 2 Matters

The ODE integration formula is:
```
z_t = z_{t-1} + velocity * dt
```

If velocity predictions are wrong, z_t can drift to [-100, 100] or any range. When comparing to ground truth in [0, 1], MSE explodes:

```
Prediction: [-5.2, 3.8, 0.9, -2.1, ...]
Ground truth: [0.3, 0.7, 0.2, 0.5, ...]
MSE: Very large → PSNR very low
```

Clamping prevents divergence:
```
Prediction (before clamp): [-5.2, 3.8, 0.9, -2.1, ...]
Prediction (after clamp): [0.0, 1.0, 0.9, 0.0, ...]
MSE: Much smaller → PSNR much higher
```

### Why Fix 3 Helps

Fourier features encode coordinates as:
```
features = [sin(2π * B[0] * x), cos(2π * B[0] * x),
            sin(2π * B[1] * x), cos(2π * B[1] * x), ...]
```

If data has frequency f=1.0 but B has random values [23.4, -15.2, ...], the encoding won't capture the structure.

By making B learnable, it can adapt:
```
Initial (random): B = [23.4, -15.2, 8.7, ...]
After training: B = [1.0, 2.0, 4.0, ...] (aligned with data frequencies)
```

---

## Next Steps

1. ✅ Fixes applied to codebase
2. ⏳ Run sanity test (Step 1 above)
3. ⏳ If successful, run full experiment suite
4. ⏳ Compare results: expect 28-35 dB baseline (vs previous 6 dB)
5. ⏳ Best improvement combinations (v3-v102) should reach 40-50 dB

---

## Troubleshooting

### If PSNR is still ~6 dB after fixes:

**Check 1**: Verify fixes are in effect
```bash
grep -n "num_query = 200" train_improved.py
grep -n "nn.Parameter" improved_transformer.py
grep -n "torch.clamp" improved_transformer.py
```

**Check 2**: Training loss curve
- If loss doesn't decrease: learning rate or optimizer issue
- If loss decreases but PSNR low: evaluation procedure issue

**Check 3**: Model outputs during training
- Print `pred_v.min()`, `pred_v.max()`, `pred_v.mean()`
- Should be reasonable values (not NaN, not huge)

**Check 4**: Overfit test
- Can model memorize 1 sample?
- If not: architecture or capacity issue

### If PSNR improves but only to ~15-20 dB:

- Try increasing `num_query` from 200 to 400
- Try increasing `train_sparsity` from 0.05 to 0.10
- Check if improvement techniques are working (try technique 2, 4, or 5)

---

## References

- **ROOT_CAUSE_ANALYSIS.md**: Detailed problem analysis
- **ARCHITECTURE_ANALYSIS.md**: V1 baseline architecture breakdown
- **V1_VS_V2_COMPARISON.md**: Architecture comparison (note: sequence length hypothesis was wrong)
- **debug_predictions.py**: Diagnostic script for testing

---

**Status**: ✅ All critical fixes applied
**Date**: 2025-10-26
**Impact**: Expected +22-29 dB PSNR improvement
**Action Required**: Run sanity test to verify
