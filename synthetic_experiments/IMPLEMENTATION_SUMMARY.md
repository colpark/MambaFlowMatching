# Implementation Summary: Fix for 6 dB PSNR Issue

**Date**: 2025-10-26
**Status**: ✅ Fixes Implemented and Ready for Testing

---

## Problem Statement

Both V1 (MAMBA) and V2 (Transformer) architectures were capped at ~6 dB PSNR, indicating models were predicting constant mean value (~0.5) instead of learning spatial patterns.

---

## Root Cause Analysis

### Initial Hypothesis (WRONG)
- V1 MAMBA has sequence length dependency bug (`dt = 1/N`)
- V2 Transformer doesn't have this bug
- Therefore only V1 should fail

### Critical Insight (CORRECT)
User feedback: **"transformers also achieves no better than 6db"**

This proved the sequence length hypothesis was WRONG. Both architectures fail equally, indicating a deeper shared issue.

### True Root Cause (IDENTIFIED)

**Training/Evaluation Distribution Mismatch**:

```
Training:
  - Model learns: 51 input pixels → 51 specific test pixels
  - Trains on fixed positions: [0, 5, 10, 15, ..., 250]
  - Loss computed only on these 51 positions

Evaluation:
  - Model must predict: 51 input pixels → ALL 1024 pixels
  - 973 positions (95%) were NEVER seen during training
  - Model defaults to safe strategy: predict mean ~0.5
  - Result: RMSE ≈ 0.5 → PSNR ≈ 6 dB
```

**Secondary Issues**:
- Unbounded ODE can diverge outside [0, 1]
- Fixed random Fourier features may not capture data structure

---

## Solutions Implemented

### Fix 1: Random Query Point Sampling (CRITICAL)

**Impact**: +20-25 dB expected
**File**: `synthetic_experiments/improvements/train_improved.py`
**Lines**: 114-139

**Change**:
```python
# OLD: Fixed 51 test pixels every iteration
test_coords_batch = test_coords[i:i+batch_size]
test_values_batch = test_values[i:i+batch_size]

# NEW: Random 200 pixels from full field each iteration
all_coords = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)
query_indices = torch.randperm(1024)[:200]
test_coords_batch = all_coords[query_indices].unsqueeze(0).expand(B, -1, -1)
test_values_batch = full_data_batch.view(B, 1024, 1)[:, query_indices, :]
```

**Benefit**: Model sees diverse positions during training, learns spatial interpolation patterns, can generalize to any query position.

---

### Fix 2: Output Clamping (IMPORTANT)

**Impact**: +2-3 dB expected
**File**: `synthetic_experiments/improvements/improved_transformer.py`
**Lines**: 354-356

**Change**:
```python
# After ODE integration
z_t = torch.clamp(z_t, 0.0, 1.0)
```

**Benefit**: Prevents ODE from diverging outside valid [0, 1] range, reduces error from out-of-bounds predictions.

---

### Fix 3: Learnable Fourier Features (OPTIMIZATION)

**Impact**: +1-2 dB expected
**File**: `synthetic_experiments/improvements/improved_transformer.py`
**Line**: 147

**Change**:
```python
# OLD: Fixed random frequencies
self.register_buffer('B', torch.randn(2, num_frequencies) * 10.0)

# NEW: Learnable frequencies
self.B = nn.Parameter(torch.randn(2, num_frequencies) * 10.0)
```

**Benefit**: Network can optimize frequencies to match data structure, better coordinate encoding.

---

## Expected Results

| Configuration | PSNR | Status |
|---------------|------|--------|
| **Before fixes** | ~6 dB | ❌ Broken |
| **After Fix 1** | ~25-30 dB | 🎯 Target |
| **After Fix 1+2** | ~27-33 dB | 🎯 Target |
| **After Fix 1+2+3** | ~28-35 dB | 🎯 Target |
| **+ Best techniques (v3-v102)** | ~40-50 dB | 🚀 Goal |

---

## Files Modified

```
synthetic_experiments/improvements/
├── train_improved.py           ✏️  MODIFIED (Fix 1)
├── improved_transformer.py     ✏️  MODIFIED (Fix 2, Fix 3)
├── FIXES_APPLIED.md            ✨ NEW (Detailed documentation)
├── test_fixes.sh               ✨ NEW (Verification script)
└── README.md                   ✏️  UPDATED (Added warning)

synthetic_experiments/
├── ROOT_CAUSE_ANALYSIS.md      ✨ NEW (Problem analysis)
├── ARCHITECTURE_ANALYSIS.md    ✨ NEW (V1 analysis)
├── V1_VS_V2_COMPARISON.md      ✨ NEW (Architecture comparison)
├── debug_predictions.py        ✨ NEW (Diagnostic script)
└── IMPLEMENTATION_SUMMARY.md   ✨ NEW (This file)
```

---

## Verification Steps

### Step 1: Verify Fixes in Code

```bash
cd synthetic_experiments/improvements
./test_fixes.sh
```

Expected output:
```
✅ Fix 1: Random query point sampling found
✅ Fix 2: Output clamping found
✅ Fix 3: Learnable Fourier features found
```

---

### Step 2: Quick Training Test (if PyTorch installed)

```bash
python3 train_improved.py \
    --techniques "" \
    --version 999 \
    --epochs 10 \
    --num_samples 100 \
    --resolution 32 \
    --gpu_id 0 \
    --save_dir checkpoints_test
```

**Expected**: PSNR > 15 dB (ideally 20-30 dB)
**Time**: ~2-5 minutes on GPU, ~20-30 minutes on CPU

---

### Step 3: Overfit Test (Diagnostic)

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

**Expected**: PSNR > 40 dB (model memorizes single sample perfectly)
**Purpose**: Confirms model architecture can learn

---

### Step 4: Run Full Experiment Suite

```bash
./run_all.sh
```

**Time**: ~30-40 hours (100 versions × 100 epochs)
**Hardware**: 4 GPUs × 2 concurrent jobs = 8 parallel

---

## Technical Details

### Why Fix 1 is Critical

The fundamental issue was a **train/test distribution mismatch**:

**Training Distribution**:
- P(pixel_position | observed_pixels) learned only for 51 specific positions
- Model became a "lookup table" for these positions
- Never learned general spatial interpolation

**Test Distribution**:
- Must predict P(pixel_position | observed_pixels) for ALL 1024 positions
- 973 positions are out-of-distribution
- Model has no learned behavior → defaults to mean

**Solution**:
- Train on random subsets (200 pixels) each iteration
- After many iterations, model sees all positions with diverse contexts
- Learns general spatial interpolation: f(observed, query_position) → value

### Mathematics of the Fix

**Old approach**:
```
Loss = MSE(model(input, test_coords_fixed), target[test_coords_fixed])
```
This only teaches: `model(input, position_5) = value_5`, `model(input, position_10) = value_10`, etc.

**New approach**:
```
random_positions = sample(all_positions, 200)
Loss = MSE(model(input, random_positions), target[random_positions])
```
Over many iterations with different random_positions, this teaches the general function:
```
model(input, any_position) = interpolated_value
```

---

## Backward Compatibility

✅ **Checkpoints**: Old checkpoints will load (architecture unchanged)
⚠️ **Performance**: Old checkpoints still show ~6 dB (need retraining)
✅ **Hyperparameters**: No changes to model size, layers, etc.
✅ **API**: All command-line arguments unchanged

---

## What If Fixes Don't Work?

### If PSNR is still ~6 dB:

1. **Check fixes are active**:
   ```bash
   grep "num_query = 200" train_improved.py
   grep "torch.clamp" improved_transformer.py
   grep "nn.Parameter" improved_transformer.py
   ```

2. **Check training loss**:
   - If loss doesn't decrease: learning rate or optimizer issue
   - If loss decreases but PSNR low: evaluation procedure issue

3. **Run overfit test**:
   - If can't overfit single sample: architecture or capacity problem
   - If can overfit: generalization or evaluation issue

### If PSNR improves to 15-20 dB (partial success):

- Increase `num_query` from 200 to 400 in `train_improved.py:134`
- Increase `train_sparsity` from 0.05 to 0.10 in `train_improved.py:91`
- Try improvement techniques (2, 4, 5 are most impactful)

---

## References

### Documentation
- **FIXES_APPLIED.md**: Detailed fix explanation and verification
- **ROOT_CAUSE_ANALYSIS.md**: Complete problem analysis with hypotheses
- **ARCHITECTURE_ANALYSIS.md**: V1 MAMBA architecture breakdown
- **V1_VS_V2_COMPARISON.md**: Why sequence length wasn't the issue
- **IMPROVEMENT_TECHNIQUES.md**: Description of 10 improvement techniques

### Scripts
- **test_fixes.sh**: Quick verification of fixes
- **debug_predictions.py**: Diagnostic testing (requires PyTorch)
- **run_all.sh**: Main orchestration script for 100 versions
- **status.py**: Real-time progress monitoring

---

## Timeline

1. **Initial Problem** (reported by user):
   - Both V1 and V2 capped at ~6 dB PSNR

2. **First Analysis** (incorrect):
   - Hypothesized V1 sequence length bug
   - Created detailed V1/V2 comparison

3. **Critical Insight** (correct):
   - User confirmed V2 also shows ~6 dB
   - Invalidated sequence length hypothesis
   - Identified train/test distribution mismatch

4. **Solution** (implemented):
   - Random query point sampling
   - Output clamping
   - Learnable Fourier features

5. **Current Status**:
   - ✅ All fixes implemented
   - ⏳ Awaiting verification testing
   - 🎯 Expected: 28-35 dB baseline

---

## Success Criteria

### Minimum Success (Must Achieve)
- ✅ Baseline PSNR > 20 dB (vs previous 6 dB)
- ✅ Model can overfit single sample to > 40 dB
- ✅ Training loss decreases consistently

### Target Success (Expected)
- 🎯 Baseline PSNR: 28-35 dB
- 🎯 Best technique combinations: > 40 dB
- 🎯 Consistent improvements across all 100 versions

### Stretch Goals (Possible)
- 🚀 Best combinations reach 45-50 dB
- 🚀 Identify optimal technique synergies
- 🚀 Achieve state-of-the-art sparse reconstruction

---

## Conclusion

Three critical fixes have been implemented to address the 6 dB PSNR issue:

1. **Random query point sampling** - Solves train/test distribution mismatch
2. **Output clamping** - Prevents ODE divergence
3. **Learnable Fourier features** - Optimizes coordinate encoding

**Expected improvement**: From 6 dB → 28-35 dB baseline (+22-29 dB gain)

**Action required**: Run verification tests to confirm fixes work as expected.

**Documentation complete**: All analysis, fixes, and verification procedures documented.

---

**Status**: ✅ READY FOR TESTING
**Next Step**: Run `./test_fixes.sh` or `./run_all.sh`
