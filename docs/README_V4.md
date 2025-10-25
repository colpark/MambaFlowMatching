# MAMBA Diffusion V4 - Transformer Encoder Baseline

Standard Transformer architecture for comparison with MAMBA's linear complexity.

---

## 🎯 Purpose

V4 replaces MAMBA state space models with standard Transformer encoder to:
- **Benchmark MAMBA**: Compare linear O(N) vs quadratic O(N²) complexity
- **Evaluate Global Context**: Test if full attention helps sparse neural fields
- **Fair Comparison**: Same depth, dimension, and training setup as V1
- **Architectural Analysis**: Understand trade-offs between sequential state and global attention

---

## 🔬 Architecture Overview

### V1 (MAMBA) vs V4 (Transformer)

| Component | V1 (MAMBA) | V4 (Transformer) |
|-----------|-----------|------------------|
| **Sequence Processing** | State space model | Multi-head self-attention |
| **Complexity** | O(N) linear | O(N²) quadratic |
| **Context** | Sequential state propagation | Global attention to all tokens |
| **Memory** | Constant state size | Full sequence stored |
| **d_model** | 256 | 256 (same) |
| **Layers** | 6 | 6 (same) |
| **Parameters** | ~4M | ~4M (similar) |
| **Cross-Attention** | Yes | Yes (same) |

### Key Differences

**MAMBA (V1)**:
```
Input → MAMBA(state propagation, O(N)) → Cross-Attention → Output
- Sequential processing with hidden state
- Linear computational complexity
- Constant memory per layer
```

**Transformer (V4)**:
```
Input → Positional Encoding → Transformer(self-attention, O(N²)) → Cross-Attention → Output
- Parallel processing with full attention
- Quadratic computational complexity
- Memory scales with sequence length
```

---

## 📐 Mathematical Details

### MAMBA State Space (V1)

```
State update: h_t = A * h_{t-1} + B * x_t
Output:       y_t = C * h_t + D * x_t

Complexity: O(N * d_state)
```

### Transformer Self-Attention (V4)

```
Q, K, V = X @ W_Q, X @ W_K, X @ W_V
Attention = softmax(Q @ K^T / √d_k) @ V

Complexity: O(N² * d_model)
```

**For 32×32 images**:
- N = 1024 tokens (sparse + query)
- V1 MAMBA: O(1024 * d_state) ≈ O(1024)
- V4 Transformer: O(1024² * d_model) ≈ O(1M)

**Implication**: V4 is ~1000x more computationally expensive per layer for 32×32 images.

---

## 🚀 Usage

### Training V4

```bash
cd v4/training
./run_transformer_v4_training.sh
```

### Custom Configuration

```bash
# Match V1 exactly for fair comparison
D_MODEL=512 NUM_LAYERS=6 NUM_HEADS=8 BATCH_SIZE=64 ./run_transformer_v4_training.sh

# Experiment with more attention heads
NUM_HEADS=16 ./run_transformer_v4_training.sh

# Adjust FFN dimension
DIM_FEEDFORWARD=4096 ./run_transformer_v4_training.sh
```

### Monitor Training

```bash
tail -f training_v4_output.log
```

---

## 📊 Expected Comparison

### Computational Complexity

| Resolution | MAMBA (O(N)) | Transformer (O(N²)) | Ratio |
|-----------|-------------|---------------------|-------|
| 32×32 | 1,024 ops | 1,048,576 ops | 1024x |
| 64×64 | 4,096 ops | 16,777,216 ops | 4096x |
| 128×128 | 16,384 ops | 268,435,456 ops | 16384x |

### Training Speed (Expected)

| Model | Time per Epoch | Memory Usage |
|-------|---------------|--------------|
| V1 (MAMBA) | ~2 minutes | ~8 GB |
| V4 (Transformer) | ~10-20 minutes | ~12-16 GB |

**Note**: V4 is expected to be **5-10x slower** due to quadratic complexity.

### Quality (Hypothesis)

**Hypothesis 1: Global context helps**
- Transformer's full attention allows every query pixel to see all input pixels
- May improve spatial coherence and reduce artifacts
- Expected: +2-4 dB PSNR improvement over V1

**Hypothesis 2: Sequential state is sufficient**
- Sparse neural fields may not require global attention
- MAMBA's state propagation captures enough context
- Expected: Similar PSNR to V1, but much slower

**Experiment will determine which hypothesis is correct.**

---

## 🔧 Technical Implementation

### Positional Encoding

```python
class PositionalEncoding(nn.Module):
    """Sinusoidal position embeddings"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]
```

### Transformer Encoder Layer

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x
```

### TransformerDiffusion Model

```python
class TransformerDiffusion(nn.Module):
    def __init__(self, d_model=256, num_layers=6, num_heads=8,
                 dim_feedforward=1024, dropout=0.1, ...):
        super().__init__()

        # Same as V1: Fourier features
        self.fourier_features = FourierFeatures(...)
        self.query_proj = nn.Linear(...)
        self.input_proj = nn.Linear(...)
        self.time_embed = SinusoidalTimeEmbedding(...)

        # NEW: Positional encoding for sequence position
        self.pos_encoder = PositionalEncoding(d_model, max_len=5000, dropout=dropout)

        # NEW: Transformer encoder (replaces MAMBA)
        self.transformer_encoder = TransformerEncoder(
            d_model, num_layers, num_heads, dim_feedforward, dropout
        )

        # Same as V1: Cross-attention and decoder
        self.query_cross_attn = nn.MultiheadAttention(...)
        self.decoder = nn.Linear(...)

    def forward(self, noisy_values, query_coords, t, input_coords, input_values):
        # Encode features (same as V1)
        query_feats = self.fourier_features(query_coords)
        query_tokens = self.query_proj(torch.cat([query_feats, noisy_values], dim=-1))

        input_feats = self.fourier_features(input_coords)
        input_tokens = self.input_proj(torch.cat([input_feats, input_values], dim=-1))

        # Time conditioning (same as V1)
        t_embed = self.time_embed(t)
        query_tokens = query_tokens + t_embed.unsqueeze(1)
        input_tokens = input_tokens + t_embed.unsqueeze(1)

        # Concatenate sequences
        seq = torch.cat([input_tokens, query_tokens], dim=1)  # (B, N_in + N_out, D)

        # NEW: Add positional encoding
        seq = self.pos_encoder(seq)

        # NEW: Process with Transformer encoder (replaces MAMBA)
        seq = self.transformer_encoder(seq)

        # Split sequences
        N_in = input_tokens.size(1)
        input_seq = seq[:, :N_in, :]
        query_seq = seq[:, N_in:, :]

        # Cross-attention and decode (same as V1)
        output, _ = self.query_cross_attn(query_seq, input_seq, input_seq)
        return self.decoder(output)
```

---

## 📈 Training Configuration

### Default Parameters

```bash
d_model=256              # Same as V1
num_layers=6             # Same as V1
num_heads=8              # Standard for 256-dim
dim_feedforward=1024     # 4x expansion ratio
batch_size=64
lr=1e-4
epochs=1000
dropout=0.1
```

### Checkpoints

```
checkpoints_transformer_v4/
├── transformer_v4_best.pth      # Best validation loss
├── transformer_v4_latest.pth    # Latest epoch
└── transformer_v4_epoch_XXXX.pth # Periodic saves
```

---

## 🆚 Version Comparison

### All Architectures

| Feature | V1 | V2 | V3 | V4 |
|---------|----|----|----|----|
| **Encoder** | MAMBA | Bidirectional MAMBA | MAMBA | Transformer |
| **Ordering** | Row-major | Row-major | Morton | Row-major |
| **Attention** | Cross-attn | Perceiver | Cross-attn | Self + Cross |
| **Complexity** | O(N) | O(N) | O(N) | O(N²) |
| **d_model** | 256 | 256 | 256 | 256 |
| **Layers** | 6 | 8 | 6 | 6 |
| **Parameters** | 4M | 5M | 4M | 4M |
| **Compute Cost** | 1.0x | 1.7x | 1.0x | 10-20x |
| **Philosophy** | Baseline | Architecture | Ordering | Comparison |

### When to Use Each

**V1 (MAMBA baseline)**:
- Fast training and inference
- Good quality baseline
- Linear complexity scales well

**V2 (Bidirectional + Perceiver)**:
- When maximum quality needed
- Moderate computational budget
- Bidirectional context helpful

**V3 (Morton curves)**:
- Best efficiency/quality trade-off
- Same speed as V1, better quality
- Zero extra cost improvement

**V4 (Transformer)**:
- For comparison and analysis
- When compute is not constrained
- Research on attention vs state space
- Baseline for other Transformer variants

---

## 🔬 Research Questions

### Questions V4 Answers

1. **Does global attention help sparse neural fields?**
   - Transformer allows every query to see all inputs
   - MAMBA only propagates sequential state
   - Compare quality metrics to determine

2. **Is the complexity trade-off worth it?**
   - V4 is 10-20x slower than V1
   - If PSNR improvement is <3 dB, probably not worth it
   - If PSNR improvement is >5 dB, might be worth the cost

3. **How does sequence length affect the gap?**
   - At 32×32 (N=1024): Gap is manageable
   - At 128×128 (N=16K): Gap becomes prohibitive
   - Helps understand MAMBA's value proposition

4. **What about zero-shot super-resolution?**
   - Does Transformer generalize better to unseen resolutions?
   - Or does MAMBA's inductive bias help more?

---

## 📊 Evaluation

### Compare V1 vs V4

```bash
cd v4/evaluation
python eval_v1_vs_v4.py \
    --v1_checkpoint ../../v1/training/checkpoints_mamba/mamba_best.pth \
    --v4_checkpoint ../training/checkpoints_transformer_v4/transformer_v4_best.pth \
    --num_samples 20
```

### Metrics to Track

**Quality**:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity)
- MSE (Mean Squared Error)
- Visual artifact reduction

**Efficiency**:
- Training time per epoch
- Memory usage
- Inference speed
- FLOPs per forward pass

**Scalability**:
- Performance at 64×, 96×, 128× resolutions
- Zero-shot super-resolution quality
- Batch size limits

---

## 🎯 Expected Outcomes

### Scenario 1: Transformer Wins (Quality)
```
V4 PSNR: 32 dB (+4 dB over V1)
V4 SSIM: 0.92 (+0.07 over V1)
→ Conclusion: Global attention helps sparse neural fields
→ Trade-off: 10x slower but significantly better quality
→ Future: Investigate efficient attention variants (linear attention, etc.)
```

### Scenario 2: MAMBA Wins (Efficiency)
```
V4 PSNR: 28.5 dB (+0.5 dB over V1)
V4 SSIM: 0.86 (+0.01 over V1)
→ Conclusion: Sequential state is sufficient, global attention not needed
→ Trade-off: 10x slower for minimal improvement
→ Future: Focus on MAMBA improvements (V3, bidirectional, etc.)
```

### Scenario 3: Mixed Results
```
V4 PSNR at 32×32: +3 dB (good)
V4 PSNR at 128×128: +1 dB (poor scaling)
→ Conclusion: Transformer helps at low resolution, struggles at high resolution
→ Trade-off: Resolution-dependent performance
→ Future: Hierarchical architectures, sparse attention patterns
```

---

## 🧪 Experiments to Run

### Basic Training
```bash
# Train V4 with default settings
cd v4/training
./run_transformer_v4_training.sh

# Monitor and compare with V1
tail -f training_v4_output.log
```

### Ablation Studies

**Number of attention heads**:
```bash
NUM_HEADS=4 ./run_transformer_v4_training.sh   # Fewer heads
NUM_HEADS=16 ./run_transformer_v4_training.sh  # More heads
```

**FFN dimension**:
```bash
DIM_FEEDFORWARD=1024 ./run_transformer_v4_training.sh  # Smaller FFN
DIM_FEEDFORWARD=4096 ./run_transformer_v4_training.sh  # Larger FFN
```

**Depth**:
```bash
NUM_LAYERS=3 ./run_transformer_v4_training.sh   # Shallower
NUM_LAYERS=12 ./run_transformer_v4_training.sh  # Deeper
```

### Direct Comparison
```bash
# Train V1 and V4 side-by-side
cd v1/training && ./run_mamba_training.sh &
cd v4/training && ./run_transformer_v4_training.sh &

# Compare checkpoints
python compare_v1_v4.py
```

---

## 🎓 Learning Insights

### Why This Comparison Matters

1. **MAMBA Validation**: Tests if MAMBA's linear complexity claim holds in practice
2. **Attention Analysis**: Determines if global context is necessary for sparse fields
3. **Efficiency Study**: Quantifies the actual cost of quadratic attention
4. **Architecture Guide**: Informs future architecture choices

### Broader Context

- **Sparse Data**: Most work uses dense data, sparse is understudied
- **Flow Matching**: Newer than diffusion, benefits from architecture comparison
- **State Space Models**: MAMBA is recent (2023), needs empirical validation
- **Neural Fields**: Continuous representations may have different needs than images

---

## 📚 References

- **Transformer**: Vaswani et al. (2017) - "Attention Is All You Need"
- **MAMBA**: Gu & Dao (2023) - "Mamba: Linear-Time Sequence Modeling"
- **Flow Matching**: Lipman et al. (2023) - "Flow Matching for Generative Modeling"
- **Efficient Attention**: Survey of linear attention mechanisms for future work

---

## ✅ Success Indicators

After training V4, you should see:
- [ ] Successful training completion (all epochs)
- [ ] Checkpoint files created
- [ ] Training logs with loss curves
- [ ] Comparison metrics vs V1
- [ ] Clear conclusion about MAMBA vs Transformer trade-offs

---

## 🎉 Summary

**V4 = Standard Transformer Encoder**

- ✅ Fair comparison with V1 (same depth, dimension)
- ✅ Tests global attention vs sequential state
- ✅ Quantifies efficiency vs quality trade-off
- ✅ Informs future architecture decisions
- ⚠️ Expected 10-20x slower than V1

**Start training:**
```bash
cd v4/training
./run_transformer_v4_training.sh
```

**Compare results:**
```bash
cd v4/evaluation
python eval_v1_vs_v4.py --v1_checkpoint ... --v4_checkpoint ...
```

---

**Made with ❤️ for understanding MAMBA vs Transformer trade-offs**
