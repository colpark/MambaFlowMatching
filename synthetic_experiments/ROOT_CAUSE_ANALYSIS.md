# Root Cause Analysis: Why Both V1 and V2 Show ~6 dB PSNR

## Executive Summary

**Critical Finding**: If V2 Transformer also shows ~6 dB PSNR, then the sequence length hypothesis is WRONG.

**New Hypothesis**: The problem is NOT architecture-specific. It's a fundamental issue with:
1. Flow matching training setup
2. Model capacity vs task difficulty
3. Sparse observation conditioning
4. Or evaluation procedure

---

## What 6 dB PSNR Means

### PSNR Math
```
PSNR = 20 * log10(MAX / RMSE) = 20 * log10(1.0 / RMSE)

6 dB PSNR → RMSE ≈ 0.5
```

### Baseline Strategies

| Strategy | Expected RMSE | Expected PSNR | What It Means |
|----------|---------------|---------------|---------------|
| Predict mean (0.5) | ~0.5 | ~6 dB | Not learning patterns |
| Predict zeros (0.0) | ~0.5 | ~6 dB | Not learning |
| Predict ones (1.0) | ~0.5 | ~6 dB | Not learning |
| Random guess | ~0.289 | ~10.8 dB | Worse than mean |
| Linear interpolation | ~0.1 | ~20 dB | Learning structure |
| Good model | ~0.032 | ~30 dB | Learning well |

**Conclusion**: 6 dB PSNR means the model is predicting approximately constant value ~0.5, which is the mean of data in [0, 1].

**The model is NOT learning spatial patterns!**

---

## Why Previous Analysis Was Wrong

### Previous Hypothesis (INCORRECT)
- V1 MAMBA has `dt = 1/N` bug causing sequence length dependency
- V2 Transformer doesn't have this bug
- Therefore V2 should work much better

### Reality (CORRECT)
- **User feedback**: "transformers also achieves no better than 6db"
- Both V1 and V2 fail equally
- Therefore the problem is NOT sequence length
- Must be a deeper, shared issue

---

## Revised Problem Analysis

### What Both Models Share

#### 1. **Same Data Pipeline**
```python
# In datasets/sinusoidal_generator.py
z = np.sin(2 * np.pi * freq_x * X) + np.sin(2 * np.pi * freq_y * Y)
z = (z + 1) / 2  # Normalize to [0, 1]
```

**Issues**:
- Simple sinusoids might have minimal variance
- [0, 1] normalization without mean centering
- No data augmentation

#### 2. **Same Training Setup**
```python
# Flow matching
t = torch.rand(B, device=device)
z_t = t * x1 + (1 - t) * x0  # Linear interpolation
v_t = x1 - x0  # Constant velocity

pred_v = model(..., query_values=z_t)
loss = F.mse_loss(pred_v, v_t)
```

**Issues**:
- Predicting velocity field v_t = x1 - x0
- Training on 51 test pixels (5%)
- Evaluation on 1024 pixels (100%)
- Huge distribution shift!

#### 3. **Same Fourier Features**
```python
self.register_buffer('B', torch.randn(2, num_frequencies) * 10.0)
```

**Issues**:
- Fixed random frequencies
- Not learnable
- May not capture data structure

#### 4. **Same Unbounded Decoder**
```python
self.decoder = nn.Sequential(
    nn.Linear(d_model, d_model),
    nn.ReLU(),  # or GELU
    nn.Linear(d_model, 1)  # No sigmoid!
)
```

**Issues**:
- Can predict any value [-∞, +∞]
- Data is in [0, 1]
- Network might output wrong range

#### 5. **Same Evaluation Procedure**
```python
# Start from random noise
z_t = torch.randn(B, N, 1, device=device)

# Integrate ODE with Heun
for i in range(num_steps):
    v = model(..., query_values=z_t)
    z_t = z_t + dt * v  # Update
```

**Issues**:
- Starts from Gaussian noise (unbounded)
- Model must learn to map noise → [0, 1]
- If model outputs are wrong, ODE diverges

---

## Hypothesis Testing Framework

### Hypothesis 1: Model Cannot Learn (Capacity)
**Test**: Overfit on single sample for 1000 epochs
**Expected**: If PSNR < 40 dB, model lacks capacity
**Fix**: Increase d_model, num_layers, or num_frequencies

### Hypothesis 2: Training Not Converging
**Test**: Plot training loss over 100 epochs
**Expected**: If loss doesn't decrease, training is broken
**Fix**: Adjust learning rate, batch size, or optimizer

### Hypothesis 3: Flow Matching Setup Is Wrong
**Test**: Manually compute velocity field and check gradient signal
**Expected**: If gradients vanish or explode, flow is wrong
**Fix**: Use different interpolation or ODE solver

### Hypothesis 4: Sparse Conditioning Fails
**Test**: Train with 50% observations instead of 5%
**Expected**: If PSNR improves significantly, sparsity is issue
**Fix**: Increase training sparsity or use data augmentation

### Hypothesis 5: Evaluation Procedure Broken
**Test**: Evaluate with ground truth query values instead of sampling
**Expected**: If PSNR improves dramatically, sampling is broken
**Fix**: Fix ODE solver or use different sampling method

### Hypothesis 6: Unbounded Outputs
**Test**: Add sigmoid to decoder and retrain
**Expected**: If PSNR improves, output range mismatch is issue
**Fix**: Always use sigmoid for [0, 1] data

### Hypothesis 7: Data Is Too Simple
**Test**: Visualize data samples - check for variation
**Expected**: If all samples look similar, data lacks diversity
**Fix**: Increase frequency range or use more complex patterns

---

## Most Likely Root Causes (Ranked)

### 1. **Training on Wrong Pixels** 🔴 CRITICAL

**Problem**:
```python
# During training
train_coords: (B, 51, 2)  # 5% observed
train_values: (B, 51, 1)
test_coords: (B, 51, 2)   # 5% test (disjoint!)
test_values: (B, 51, 1)

# Model learns: sparse → sparse mapping
loss = MSE(pred[test_coords], target[test_coords])
```

But during evaluation:
```python
# During evaluation
train_coords: (B, 51, 2)   # Same 5% observed
query_coords: (B, 1024, 2)  # 100% full field!

# Model must generalize: sparse → FULL mapping
```

**Why this causes 6 dB**:
- Model trains to predict 51 specific pixels
- Never sees the remaining 973 pixels during training
- At evaluation, must predict completely unseen positions
- Model defaults to safe strategy: predict mean ~0.5
- Result: RMSE ≈ 0.5 → PSNR ≈ 6 dB

**Evidence**:
- Training loss might be low (good at 51 pixels)
- Evaluation PSNR is terrible (bad at 1024 pixels)
- This is a MASSIVE distribution shift

**Fix**:
```python
# Option A: Train on full field
query_coords = all_1024_coordinates
loss = MSE(pred[all], target[all])

# Option B: Random subset each iteration
query_indices = random.sample(range(1024), 200)
query_coords = all_coords[query_indices]
loss = MSE(pred[query_coords], target[query_coords])
```

---

### 2. **Unbounded Decoder Outputs** 🔴 CRITICAL

**Problem**:
```python
# Decoder outputs any value
output = decoder(features)  # Range: [-∞, +∞]

# But data is bounded
target ∈ [0, 1]
```

During ODE integration:
```python
z_t = torch.randn(...)  # Start: mean=0, std=1
for step in range(50):
    v = model(...)  # Velocity: unbounded!
    z_t = z_t + dt * v  # z_t can go anywhere
```

**Why this causes 6 dB**:
- Model outputs go outside [0, 1]
- ODE integration diverges
- Final z_t might be in [-5, +5] or any range
- When clipped to [0, 1] for PSNR, becomes constant ~0.5
- Result: PSNR ≈ 6 dB

**Fix**:
```python
self.decoder = nn.Sequential(
    nn.Linear(d_model, d_model),
    nn.GELU(),
    nn.Linear(d_model, 1),
    nn.Sigmoid()  # Force [0, 1] output
)
```

---

### 3. **Inadequate Training Signal** 🟡 IMPORTANT

**Problem**:
- Only 51 training pixels (5%)
- Must infer patterns from extreme sparsity
- No regularization or inductive bias

**Why this causes poor performance**:
- Not enough signal to learn structure
- Model underfits the pattern
- Falls back to predicting mean

**Fix**:
- Increase train_sparsity to 10% or 20%
- Use data augmentation
- Add perceptual loss or regularization

---

### 4. **Fixed Random Fourier Features** 🟡 IMPORTANT

**Problem**:
```python
self.register_buffer('B', torch.randn(2, num_frequencies) * 10.0)
```

Random frequencies might not align with data structure:
- Data frequency: 2π * 1.0 (low frequency)
- Random B: 10 * randn() ≈ [-30, +30]
- Mismatch means poor coordinate encoding

**Fix**:
```python
self.B = nn.Parameter(torch.randn(2, num_frequencies) * 10.0)
```

---

### 5. **No Data Normalization** 🟢 RECOMMENDED

**Problem**:
```python
# Data is in [0, 1]
values ∈ [0, 1]

# But networks prefer zero-mean
ideal: values ∈ [-1, +1] or mean=0, std=1
```

**Fix**:
```python
# Normalize
mean, std = 0.5, 0.289
values_norm = (values - mean) / std

# Train on normalized
...

# Denormalize outputs
outputs = outputs * std + mean
```

---

## Diagnostic Experiments (Priority Order)

### Experiment 1: Sanity Check - Overfit Single Sample
**Purpose**: Verify model CAN learn at all

```bash
python3 train_improved.py \
    --num_samples 1 \
    --epochs 1000 \
    --batch_size 1 \
    --save_dir checkpoints_overfit
```

**Expected**: PSNR > 40 dB (model memorizes pattern)
**If fails**: Model capacity or architecture is fundamentally broken

---

### Experiment 2: Train on Full Field
**Purpose**: Test if training distribution shift is the problem

```python
# Modify train_improved.py
# Instead of:
query_coords = test_coords  # 51 pixels

# Use:
H, W = 32, 32
y_grid, x_grid = torch.meshgrid(...)
query_coords = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)
# Now training on all 1024 pixels
```

**Expected**: If PSNR jumps to 25-35 dB, distribution shift is the root cause

---

### Experiment 3: Add Sigmoid to Decoder
**Purpose**: Test if unbounded outputs are the problem

```python
# In improved_transformer.py
self.decoder = nn.Sequential(
    nn.Linear(d_model, d_model),
    nn.GELU(),
    nn.Linear(d_model, 1),
    nn.Sigmoid()  # Add this!
)
```

**Expected**: If PSNR improves by 2-5 dB, output range is a factor

---

### Experiment 4: Increase Training Sparsity
**Purpose**: Test if 5% is too sparse

```bash
python3 train_improved.py \
    --train_sparsity 0.20 \
    --test_sparsity 0.05
```

**Expected**: If PSNR improves significantly, sparsity is an issue

---

### Experiment 5: Visualize Training Loss
**Purpose**: Check if training is converging

```python
# Check if loss decreases over epochs
# If loss stays constant, training is broken
```

---

## Recommended Fix Strategy

### Phase 1: Critical Fixes (Try First)

1. **Fix Training Distribution**
   ```python
   # Train on random 200 pixels per iteration (not fixed 51 test)
   query_indices = torch.randperm(1024)[:200]
   query_coords = all_coords[query_indices]
   ```

2. **Add Sigmoid**
   ```python
   self.decoder = nn.Sequential(..., nn.Sigmoid())
   ```

3. **Increase Training Sparsity**
   ```python
   train_sparsity = 0.10  # 102 pixels instead of 51
   ```

**Expected gain**: +20-25 dB PSNR (from 6 to 26-31 dB)

### Phase 2: Optimization Fixes

4. **Learnable Fourier Features**
5. **Data Normalization**
6. **Perceptual Loss**

**Expected additional gain**: +5-10 dB PSNR (total 31-41 dB)

---

## Code Changes Required

### Fix 1: Training on Random Query Points

```python
# In train_improved.py, train_epoch function

def train_epoch(...):
    # ... existing code ...

    # NEW: Create full coordinate grid
    H, W = dataset.resolution, dataset.resolution
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing='ij'
    )
    all_coords = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)
    all_coords = all_coords.unsqueeze(0).expand(batch_size, -1, -1)

    for i in range(0, num_samples, batch_size):
        train_coords_batch = train_coords[i:i+batch_size].to(device)
        train_values_batch = train_values[i:i+batch_size].to(device)
        full_data_batch = full_data[i:i+batch_size].to(device)

        # Sample random query points each iteration
        num_query = 200  # More than 51, less than 1024
        query_indices = torch.randperm(1024)[:num_query]

        query_coords_batch = all_coords[:, query_indices, :]
        query_values_batch = full_data_batch.view(batch_size, -1, 1)[:, query_indices, :]

        # ... rest of training loop ...
```

### Fix 2: Add Sigmoid

```python
# In improved_transformer.py, line ~260

self.decoder = nn.Sequential(
    nn.Linear(d_model, d_model),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(d_model, 1),
    nn.Sigmoid()  # ADD THIS LINE
)
```

### Fix 3: Learnable Fourier Features

```python
# In improved_transformer.py, line ~174
# Change from:
self.register_buffer('B', torch.randn(2, num_frequencies) * 10.0)

# To:
self.B = nn.Parameter(torch.randn(2, num_frequencies) * 10.0)
```

---

## Expected Results After Fixes

| Configuration | Expected PSNR | Explanation |
|---------------|---------------|-------------|
| Current (broken) | ~6 dB | Predicting mean |
| + Random query points | ~25-30 dB | Learning spatial structure |
| + Sigmoid output | +2-3 dB | Bounded outputs |
| + Learnable Fourier | +1-2 dB | Better coordinate encoding |
| + Normalization | +1-2 dB | Better training dynamics |
| **Total** | **~30-37 dB** | Substantial improvement |
| + Best techniques (v3-v102) | **~40-50 dB** | State-of-the-art |

---

## Why User Was Confused

**User's observation**: "Performance is limited around 6db psnr. There is something inherently wrong about the setting."

**User was RIGHT** - but for different reasons than initially analyzed:

1. ✅ Correct: Something is fundamentally wrong
2. ❌ Wrong assumption: It's sequence length (dt = 1/N bug)
3. ✅ Actual issue: Training/evaluation distribution mismatch + unbounded outputs

**Why the confusion**:
- V1 MAMBA *does* have the dt bug
- But that's not why PSNR is 6 dB
- V2 Transformer proves this (same PSNR, no dt bug)
- Real issue is shared between both architectures

---

## Next Steps

1. **Immediate**: Implement Fix 1 (random query points) and test
2. **Quick win**: Add sigmoid (Fix 2)
3. **Validation**: Run overfit experiment (1 sample, 1000 epochs)
4. **Analysis**: Plot training loss curves to verify convergence
5. **Iteration**: If still broken, move to Hypothesis 3 (flow matching setup)

---

## Conclusion

The 6 dB PSNR is NOT caused by sequence length dependency. Both V1 and V2 fail equally, proving it's a shared issue.

**Most likely root cause**: Training on fixed 51 test pixels but evaluating on full 1024 pixels creates massive distribution shift. Model defaults to predicting mean (~0.5) for unseen positions.

**Solution**: Train on random subsets of full field each iteration, add sigmoid to bound outputs, increase training sparsity.

**Expected improvement**: From 6 dB → 30-37 dB after fixes.
