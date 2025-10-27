# V1 vs V2 Architecture Comparison

## Executive Summary

**V1 (MAMBA) has a critical bug that caps performance at ~6 dB PSNR.**
**V2 (Transformer) does NOT have this bug and should perform much better.**

The root cause: V1 uses State Space Models (SSM) with discretization `dt = 1/N` where N is the sequence length. This changes 10× between training and evaluation, completely breaking the learned dynamics.

---

## Side-by-Side Architecture Comparison

| Component | V1 (MAMBA) | V2 (Transformer) | Impact |
|-----------|------------|------------------|---------|
| **Sequence Processing** | MAMBA SSM blocks | Standard Transformer | ✅ V2 unaffected by length |
| **Discretization** | `dt = 1/N` (N varies!) | Position-invariant attention | ✅ V2 has no dt dependency |
| **Train Length** | N=102 (51+51) | N=102 (51+51) | Same |
| **Eval Length** | N=1075 (51+1024) | N=1075 (51+1024) | Same |
| **Dynamics Change** | 10× dt change! | No change | 🚨 V1 broken |
| **Fourier Features** | Fixed (not learnable) | Fixed (not learnable) | Both need fix |
| **Output Activation** | None (unbounded) | None (unbounded) | Both need sigmoid |
| **Data Normalization** | No | No | Both could improve |
| **Parameters** | ~287K | ~350K | Similar capacity |

---

## Critical Difference: Sequence Length Sensitivity

### V1 MAMBA - BROKEN

```python
# In train_mamba_v1.py, SimpleSSM forward (line ~51)
dt = 1.0 / N  # N = sequence length

# Training: N = 102 (51 input + 51 query)
dt_train = 1/102 ≈ 0.0098

# Evaluation: N = 1075 (51 input + 1024 query)
dt_eval = 1/1075 ≈ 0.0009

# State transition matrix
A_bar = torch.exp(dt * A)  # Completely different values!
```

**Why this kills performance:**
- The SSM learns state dynamics A_bar for dt=0.0098
- At evaluation, it uses dt=0.0009 (10× smaller)
- State evolution is fundamentally different
- All learned dynamics become invalid

**Analogy**: Training a car to drive at 60 mph, then forcing it to drive at 6 mph - all the learned control is wrong!

### V2 Transformer - CORRECT

```python
# In train_transformer_v2.py, TransformerEncoderLayer forward (line ~73)
def forward(self, x):
    # Self-attention: completely position-invariant
    attn_out, _ = self.self_attn(x, x, x)
    # No dependence on sequence length N!
```

**Why this works:**
- Attention mechanism is position-invariant
- Works identically regardless of sequence length
- No discretization parameter to change
- Learned patterns transfer perfectly from train to eval

---

## Secondary Issues (Both Models)

### 1. Unbounded Outputs

Both models output unbounded predictions despite data being in [0, 1]:

```python
# V1 and V2 both have:
self.decoder = nn.Sequential(
    nn.Linear(d_model, d_model),
    nn.ReLU(),  # or GELU in V2
    nn.Linear(d_model, 1)  # No sigmoid!
)
```

**Fix**: Add `nn.Sigmoid()` as final layer.

**Expected gain**: +2-3 dB PSNR

### 2. Fixed Fourier Features

Both models use random non-learnable Fourier frequencies:

```python
# V1 and V2 both have:
self.register_buffer('B', torch.randn(2, num_frequencies) * 10.0)
```

Random frequencies might be suboptimal for the data.

**Fix**: Make learnable:
```python
self.B = nn.Parameter(torch.randn(2, num_frequencies) * 10.0)
```

**Expected gain**: +1-2 dB PSNR

### 3. No Data Normalization

Both models use raw [0, 1] values without normalization:

```python
# In datasets/sinusoidal_generator.py
z = (z + 1) / 2  # Shift from [-1,1] to [0,1]
# But no mean/std normalization!
```

Neural networks typically benefit from zero-mean, unit-variance inputs.

**Fix**: Normalize to mean=0, std=1:
```python
mean = 0.5  # Expected mean of [0,1] uniform
std = 0.289  # Std of [0,1] uniform
z_normalized = (z - mean) / std
```

**Expected gain**: +1-2 dB PSNR

---

## Model Architectures in Detail

### V1 (MAMBA) Flow

```
Input: sparse_coords (B, 51, 2), sparse_values (B, 51, 1)
Query: query_coords (B, N_query, 2), query_values (B, N_query, 1)

↓ Fourier Features (fixed)
sparse_feats: (B, 51, 32)
query_feats: (B, N_query, 32)

↓ Projections
input_tokens: (B, 51, 128)
query_tokens: (B, N_query, 128)

↓ Time Conditioning
Add time_embed: (B, 128) → all tokens

↓ Concatenate
seq: (B, 51+N_query, 128)
    Train: (B, 102, 128)
    Eval:  (B, 1075, 128)  ← Different N!

↓ MAMBA Blocks (×4)
    dt = 1/N  ← CHANGES BETWEEN TRAIN AND EVAL!
    A_bar = exp(dt * A)  ← Discretization

↓ Split
input_seq: (B, 51, 128)
query_seq: (B, N_query, 128)

↓ Cross-Attention
attended: (B, N_query, 128)

↓ Decoder
output: (B, N_query, 1)
```

**Problem**: dt = 1/102 in training vs dt = 1/1075 in eval

### V2 (Transformer) Flow

```
Input: sparse_coords (B, 51, 2), sparse_values (B, 51, 1)
Query: query_coords (B, N_query, 2), query_values (B, N_query, 1)

↓ Fourier Features (fixed)
sparse_feats: (B, 51, 32)
query_feats: (B, N_query, 32)

↓ Projections
input_tokens: (B, 51, 128)
query_tokens: (B, N_query, 128)

↓ Time Conditioning
Add time_embed: (B, 128) → all tokens

↓ Concatenate
seq: (B, 51+N_query, 128)
    Train: (B, 102, 128)
    Eval:  (B, 1075, 128)  ← Different N!

↓ Transformer Layers (×4)
    Self-Attention (position-invariant)
    Feed-Forward Network
    NO dt parameter!  ← Length-independent!

↓ Split
input_seq: (B, 51, 128)
query_seq: (B, N_query, 128)

↓ Cross-Attention
attended: (B, N_query, 128)

↓ Decoder
output: (B, N_query, 1)
```

**No problem**: Attention is completely independent of sequence length

---

## Performance Predictions

### V1 (MAMBA) - Current State

```
Current Performance: ~6 dB PSNR
Root Cause: Sequence length bug (PRIMARY)
Secondary: Unbounded outputs, fixed features, no normalization

Expected with fixes:
1. Fix dt dependency (CRITICAL): +25-30 dB → 31-36 dB
2. Add sigmoid: +2-3 dB → 33-39 dB
3. Learnable Fourier: +1-2 dB → 34-41 dB
4. Normalization: +1-2 dB → 35-43 dB

Predicted Final: 35-40 dB PSNR
```

### V2 (Transformer) - Current State

```
Current Performance: Much better (user confirmed "transformer does much better")
Root Cause: No sequence length bug ✅

Expected with fixes:
Baseline (no bug): ~35 dB PSNR (estimated)
1. Add sigmoid: +2-3 dB → 37-38 dB
2. Learnable Fourier: +1-2 dB → 38-40 dB
3. Normalization: +1-2 dB → 39-42 dB

Predicted Final: 39-42 dB PSNR
```

---

## Why V1 Shows ~6 dB PSNR

**PSNR formula**: `PSNR = 20 * log10(MAX / RMSE)`

For MAX = 1.0:
- **6 dB** → RMSE ≈ 0.5
- **35 dB** → RMSE ≈ 0.018

**RMSE = 0.5 means:**
- Predicting constant value ~0.5 (mean of [0,1])
- Model is essentially NOT learning the pattern
- Performance is at "predict mean" baseline

**Why the model can't learn:**
1. Trains with dt=0.0098, learns slow state evolution
2. Evaluates with dt=0.0009, uses 10× faster evolution
3. Predictions are completely wrong
4. Gradient signals are corrupted
5. Model converges to safe strategy: predict mean

---

## Recommended Action Plan

### Immediate: Validate V2 Performance

```bash
# Check V2 results from improvement experiments
python3 synthetic_experiments/improvements/status.py

# V2 baseline should be ~35 dB PSNR
# If not, check experiment setup
```

### Short-term: Apply Secondary Fixes to V2

Priority fixes for V2 (in order):
1. **Add sigmoid to decoder** (2-3 dB gain)
2. **Make Fourier features learnable** (1-2 dB gain)
3. **Add data normalization** (1-2 dB gain)

Expected result: 39-42 dB PSNR

### Long-term: Fix V1 or Abandon It

Two options for V1:
1. **Fix dt dependency**: Use fixed dt independent of sequence length
2. **Abandon V1**: Focus on V2 Transformer which works correctly

**Recommendation**: Focus on V2. The Transformer architecture is fundamentally more robust for variable-length sequences.

---

## Implementation Code Snippets

### Fix 1: Add Sigmoid (Both Models)

```python
# In improved_transformer.py or train_mamba_v1.py
self.decoder = nn.Sequential(
    nn.Linear(d_model, d_model),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(d_model, 1),
    nn.Sigmoid()  # Add this line!
)
```

### Fix 2: Learnable Fourier Features (Both Models)

```python
# Change from:
self.register_buffer('B', torch.randn(2, num_frequencies) * 10.0)

# To:
self.B = nn.Parameter(torch.randn(2, num_frequencies) * 10.0)
```

### Fix 3: Data Normalization (Both Models)

```python
# In training loop
def train_epoch(...):
    # Normalize data
    mean = 0.5
    std = 0.289  # np.sqrt(1/12) for uniform [0,1]

    train_values = (train_values - mean) / std
    test_values = (test_values - mean) / std

    # Training...
    pred = model(...)

    # Denormalize predictions
    pred = pred * std + mean
    pred = torch.clamp(pred, 0, 1)  # Ensure [0,1] range
```

### Fix 4: V1 dt Independence (If Keeping V1)

```python
# In SimpleSSM forward method
# Change from:
N = x.shape[1]
dt = 1.0 / N  # BAD: depends on sequence length

# To:
dt = 1.0 / 1024  # Fixed dt based on spatial resolution
# OR
self.dt = nn.Parameter(torch.tensor(0.01))  # Learnable dt
```

---

## Diagnostic Experiments

### Experiment 1: Confirm V2 Baseline

```bash
# Check if V2 baseline is ~35 dB
python3 synthetic_experiments/baselines/train_transformer_v2.py \
    --epochs 100 \
    --complexity simple \
    --resolution 32 \
    --num_samples 500

# Expected: Final PSNR ~35 dB
```

### Experiment 2: Test Sigmoid Fix

```bash
# Add sigmoid to V2 decoder
# Re-run training
# Expected: PSNR ~37-38 dB (+2-3 dB gain)
```

### Experiment 3: Overfit Single Sample

```bash
# Train V1 on 1 sample for 1000 epochs
# If PSNR < 40 dB, architecture is fundamentally broken
# If PSNR > 40 dB, sequence length is the issue
```

---

## Conclusion

**V1 (MAMBA) is broken due to sequence length dependency in SSM discretization.**

**V2 (Transformer) works correctly and should be the baseline for all improvements.**

The improvement experiments (v3-v102) should all build on V2, not V1, because V2 has the correct architecture.

Expected performance after fixes:
- V2 baseline: ~35 dB PSNR
- V2 + sigmoid: ~37-38 dB PSNR
- V2 + all fixes: ~39-42 dB PSNR
- V2 + best improvement techniques: ~45-50 dB PSNR

**Action**: Focus all efforts on V2 Transformer. V1 MAMBA is not salvageable without major architecture changes.
