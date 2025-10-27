# V1 Baseline Architecture Analysis

## 🔍 Problem: Performance Capped at ~6 dB PSNR

You're absolutely right - there's something fundamentally wrong if performance is capped this low. Let me analyze every component systematically.

---

## 📊 Data Pipeline Analysis

### Input Data Characteristics

**Resolution**: 32×32 = 1,024 pixels

**Value Range**: [0, 1] (normalized via `z = (z + 1) / 2`)
- Raw sinusoids are in [-1, 1]
- Shifted to [0, 1] for training

**Observation Counts**:
- Train: 5% × 1024 = **51 pixels**
- Test: 5% × 1024 = **51 pixels** (disjoint from train)
- Full reconstruction: **1024 pixels**

**Critical Issues**:

1. **Extremely Sparse Training Signal**: Only 51 pixels to learn from!
2. **Value Range Mismatch**: Data in [0, 1] but no explicit clamping in model
3. **No Normalization Statistics**: Using raw values without mean/std normalization

---

## 🧠 Model Architecture Breakdown

### Layer-by-Layer Analysis

#### 1. **Fourier Feature Encoding**
```python
self.B = torch.randn(2, num_frequencies) * 10.0
proj = coords @ self.B  # (B, N, 2) @ (2, 16) = (B, N, 16)
features = [sin(proj), cos(proj)]  # (B, N, 32)
```

**What it learns**: High-frequency coordinate embeddings
**Issue**: Random initialization (`randn * 10.0`) - no learned frequency selection
**Output dim**: 32 (2 × 16 frequencies)

#### 2. **Input Projection**
```python
input_proj: Linear(33, 128)  # 32 (coords) + 1 (value) → 128
```

**What it learns**: Projects sparse observations + coordinates to model dimension
**Input**: 51 tokens of 33 dims (32 Fourier + 1 value)
**Output**: 51 tokens of 128 dims

#### 3. **Query Projection**
```python
query_proj: Linear(33, 128)  # 32 (coords) + 1 (noisy value) → 128
```

**What it learns**: Projects query points + noisy values
**Input**: 51 tokens of 33 dims (during training) or 1024 (during eval)
**Output**: Same number of 128-dim tokens

#### 4. **Time Embedding**
```python
SinusoidalTimeEmbedding(128)
```

**What it learns**: Nothing (fixed sinusoidal encoding)
**Purpose**: Conditions model on diffusion timestep t ∈ [0, 1]
**Added to ALL tokens** via broadcasting

#### 5. **Sequence Concatenation**
```python
seq = [input_tokens, query_tokens]  # (B, 51+51, 128) during training
                                     # (B, 51+1024, 128) during eval
```

**Critical Issue**: **Sequence length changes between train/test!**
- Train: 102 tokens total
- Eval: 1075 tokens total

This is a **MAJOR PROBLEM** for SSM which has fixed dt = 1/N discretization!

#### 6. **MAMBA Blocks (×4 layers)**
```python
for mamba in self.mamba_blocks:
    seq = mamba(seq)
```

**Each MAMBA block**:
```python
class SimpleMambaBlock:
    - LayerNorm
    - SimpleSSM (d_state=8)
        - A_log: learnable diagonal state transition (-exp clipped to [1e-8, 10])
        - B: Linear(128 → 8) input projection
        - C: Linear(8 → 128) output projection
        - D: skip connection (128 params)
        - Sequential state update with dt = 1/N
    - Gated residual connection
```

**What each layer learns**:
- State dynamics A (8 params)
- Input mapping B (128×8 = 1024 params)
- Output mapping C (8×128 = 1024 params)
- Skip connection D (128 params)
- Gate network (128×128 = 16,384 params)

**Total per block**: ~18,568 parameters
**4 blocks**: ~74,272 parameters

**CRITICAL ISSUE**: `dt = 1/N` where N changes between train (102) and eval (1075)!
```python
dt = 1.0 / N  # Line 51 in train_mamba_v1.py
A_bar = torch.exp(dt * A)  # Discretization depends on sequence length!
```

This means the model trains with different dynamics than it evaluates with!

#### 7. **Sequence Splitting**
```python
N_sparse = 51
input_seq = seq[:, :51, :]    # Extract input tokens
query_seq = seq[:, 51:, :]    # Extract query tokens
```

#### 8. **Cross-Attention**
```python
MultiheadAttention(d_model=128, num_heads=4)
attended = cross_attn(query_seq, input_seq, input_seq)
```

**What it learns**: How to attend from query points to sparse observations
**Params**: 4 × (128²×3 + 128) = ~196,608 parameters

#### 9. **MLP Decoder**
```python
Sequential(
    Linear(128, 128),
    ReLU(),
    Linear(128, 1)
)
```

**What it learns**: Maps attended features to scalar values
**Output range**: Unbounded! No sigmoid/tanh to constrain to [0, 1]
**Params**: 128×128 + 128×1 = 16,512 parameters

**Total Model Parameters**: ~287,000

---

## 🎯 Loss Function Analysis

### Training Objective

```python
# Flow matching loss
t = torch.rand(B)  # Random timestep in [0, 1]
z_t = t * x1 + (1-t) * x0  # Linear interpolation
v_t = x1 - x0  # Constant velocity field

pred_v = model(..., query_values=z_t)
loss = F.mse_loss(pred_v, v_t)
```

**What the model learns**: Predict velocity field v_t = x1 - x0

**Training**:
- Input: 51 sparse observations
- Query: 51 test pixels (disjoint)
- Loss computed on 51 test pixels only

**Evaluation**:
- Input: 51 sparse observations
- Query: **ALL 1024 pixels**
- Metrics computed on full 1024 pixel reconstruction

---

## ⚠️ CRITICAL PROBLEMS IDENTIFIED

### 1. **Sequence Length Mismatch** (MOST CRITICAL)

**Train sequence**: 51 + 51 = 102 tokens
**Eval sequence**: 51 + 1024 = 1075 tokens

**Why this kills performance**:
```python
dt = 1.0 / N  # N = 102 in train, N = 1075 in eval
A_bar = torch.exp(dt * A)
```

The SSM discretization `A_bar` is **10× different** between train and eval!
- Train: dt = 1/102 ≈ 0.0098
- Eval: dt = 1/1075 ≈ 0.0009

The model learns state dynamics for one dt, then runs with a completely different dt at test time.

### 2. **Unbounded Output Range**

Data is in [0, 1], but decoder outputs are unbounded:
```python
self.decoder = nn.Sequential(
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 1)  # No activation!
)
```

Predictions can be anything: [-∞, +∞] instead of [0, 1]

### 3. **Fixed (Non-Learned) Fourier Features**

```python
self.B = torch.randn(2, num_frequencies) * 10.0  # Fixed, not learnable!
```

The frequencies are random and never optimized. Some might be completely useless.

### 4. **Tiny Training Signal**

Only 51 pixels to learn from, predicting 51 different pixels.
No overlap means the model has to **generalize** from 5% to 100% immediately.

### 5. **No Data Normalization**

Values are in [0, 1] but no explicit normalization:
- No mean centering
- No std normalization
- Raw values fed directly to network

### 6. **Velocity Field is Too Simple**

```python
v_t = x1 - x0  # Constant velocity
```

This assumes straight-line paths in data space. For complex patterns, this might be suboptimal.

---

## 📈 Expected PSNR Analysis

### Why 6 dB is suspiciously low

**PSNR formula**: `PSNR = 20 * log10(MAX / RMSE)`

For MAX = 1.0:
- **6 dB** ⇒ RMSE ≈ 0.5 (predicting mean value ~0.5)
- **20 dB** ⇒ RMSE ≈ 0.1 (decent reconstruction)
- **30 dB** ⇒ RMSE ≈ 0.032 (good reconstruction)
- **40 dB** ⇒ RMSE ≈ 0.01 (excellent reconstruction)

**6 dB suggests the model is barely better than predicting constant 0.5!**

### Sanity Check Baselines

**Predict mean** (constant 0.5): ~6 dB PSNR
**Nearest neighbor**: ~10-15 dB PSNR
**Linear interpolation**: ~15-20 dB PSNR
**Good flow model**: ~30-40 dB PSNR

Your results match "predict mean" - the model isn't learning anything useful!

---

## 🔧 ROOT CAUSE HYPOTHESIS

### Primary Culprit: Sequence Length Mismatch

The SSM's `dt = 1/N` makes training and evaluation use **fundamentally different dynamics**:

1. Model trains with N=102, learns state transition A_bar = exp(dt * A) with dt=1/102
2. At eval, N=1075, uses A_bar = exp(dt * A) with dt=1/1075
3. State evolution is **10× slower** at eval time
4. Model completely breaks down

**Analogy**: Imagine training a car to drive at 60 mph, then forcing it to drive at 6 mph at test time - all the learned dynamics are wrong!

### Secondary Issues

2. **Unbounded outputs** - model can predict anything, no constraint to [0, 1]
3. **Random Fourier features** - frequencies never optimized for the data
4. **Extreme sparsity** - 51 pixels is very little signal
5. **No normalization** - training is harder without proper scaling

---

## 🎯 FIXES REQUIRED (Priority Order)

### **FIX 1: Sequence Length Independence** (CRITICAL)

**Option A**: Fixed sequence length (pad/crop)
```python
# Always use same length for train and eval
MAX_SEQ_LEN = 1024 + 51  # Fixed
# Pad query_tokens to MAX_SEQ_LEN - 51 during training
```

**Option B**: Length-independent dt (BETTER)
```python
# Don't tie dt to sequence length
dt = 1.0 / 1024  # Fixed dt based on spatial resolution, not sequence length
# OR make dt learnable
self.dt = nn.Parameter(torch.tensor(0.01))
```

**Option C**: Use position embeddings instead of SSM (transformer already works!)

### **FIX 2: Bounded Output**

```python
self.decoder = nn.Sequential(
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()  # Constrain to [0, 1]
)
```

### **FIX 3: Learnable Fourier Features**

```python
self.B = nn.Parameter(torch.randn(2, num_frequencies) * 10.0)  # Make learnable!
```

### **FIX 4: Data Normalization**

```python
# Normalize to zero mean, unit variance
mean, std = dataset.get_statistics()
values_normalized = (values - mean) / std
# Denormalize outputs
outputs = outputs * std + mean
```

### **FIX 5: More Training Observations**

Increase to 10% or 20%:
```python
train_sparsity = 0.10  # 102 pixels instead of 51
test_sparsity = 0.10
```

---

## 📊 Expected Improvements

| Fix | Expected PSNR Gain |
|-----|-------------------|
| FIX 1 (sequence length) | +10-15 dB (CRITICAL) |
| FIX 2 (bounded output) | +2-3 dB |
| FIX 3 (learnable Fourier) | +1-2 dB |
| FIX 4 (normalization) | +1-2 dB |
| FIX 5 (more observations) | +3-5 dB |

**Combined**: 35-40 dB PSNR (vs current 6 dB)

---

## 🔬 Diagnostic Experiments

### Experiment 1: Test Sequence Length Hypothesis

```python
# Modify eval to use same sequence length as training
# Pad to 102 tokens instead of 1075
# If PSNR jumps dramatically, this is the issue
```

### Experiment 2: Test Bounded Output

```python
# Add sigmoid to decoder
# Check if predictions stay in [0, 1] range
```

### Experiment 3: Sanity Check - Overfit Single Image

```python
# Train on 1 image with 1000 epochs
# Should get PSNR > 40 dB if model can learn at all
# If not, architecture is fundamentally broken
```

---

## 💡 Recommended Action Plan

1. **Immediate**: Fix sequence length issue (use Transformer V2 which doesn't have this problem, or fix MAMBA dt)
2. **Quick wins**: Add sigmoid, make Fourier learnable
3. **Validation**: Run overfit experiment on single image
4. **Scale up**: Increase to 10% observations
5. **Compare**: V2 Transformer should work better (no sequence length issue)

---

## Summary

The V1 MAMBA baseline has a **critical architectural bug**: the SSM discretization (`dt = 1/N`) changes between training and evaluation, completely breaking the learned dynamics. This alone explains the ~6 dB PSNR cap.

Secondary issues (unbounded outputs, random features, etc.) compound the problem but are not the primary cause.

**The good news**: This is fixable! The Transformer V2 should work much better since it doesn't have the sequence-length-dependent dynamics issue.
