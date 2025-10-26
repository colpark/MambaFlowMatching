# Systematic Architecture Improvement Experiments

Automated framework for testing 100 combinations of 10 improvement techniques across 4 GPUs.

---

## 🎯 Quick Start

```bash
cd synthetic_experiments/improvements

# Generate combinations and run all 100 experiments
./run_all.sh

# Monitor progress
tail -f logs_improvements/orchestrator.log

# Check specific version
tail -f logs_improvements/v42_gpu2.log
```

**Expected Runtime**: ~30-40 hours for 100 versions × 100 epochs each

---

## 📋 10 Improvement Techniques

Ranked by projected performance impact:

| # | Technique | Projected Gain | Overhead | Type |
|---|-----------|----------------|----------|------|
| 1 | **Rectified Flow** | +5-8 dB | 1.2x | Flow Matching |
| 2 | **Multi-Scale PE** | +4-7 dB | 1.0x | Architecture |
| 3 | **Consistency Loss** | +3-6 dB | 1.1x | Loss Function |
| 4 | **Perceptual Loss** | +3-5 dB | 1.15x | Loss Function |
| 5 | **Adaptive LayerNorm** | +2-5 dB | 1.05x | Architecture |
| 6 | **Self-Conditioning** | +2-4 dB | 1.3x | Training |
| 7 | **EMA** | +1-3 dB | 1.02x | Training |
| 8 | **Noise Schedule** | +1-3 dB | 1.0x | Flow Matching |
| 9 | **Grad Clip + Warmup** | +0.5-2 dB | 1.0x | Training |
| 10 | **Stochastic Depth** | +0.5-2 dB | 1.0x | Regularization |

**See**: `IMPROVEMENT_TECHNIQUES.md` for detailed descriptions

---

## 🔧 Combinatorial Strategy

### Version Distribution (v3 → v102)

- **v3-v12** (10 versions): Single techniques
- **v13-v57** (45 versions): All 2-combinations (C(10,2))
- **v58-v92** (35 versions): Selected 3-combinations (top techniques + diversity)
- **v93-v102** (10 versions): 4+ combinations (kitchen sink)

**Total**: 100 versions testing different technique combinations

---

## 📁 Project Structure

```
improvements/
├── README.md                      # This file
├── IMPROVEMENT_TECHNIQUES.md      # Detailed technique descriptions
│
├── techniques.py                  # Technique implementations
├── improved_transformer.py        # Model with pluggable techniques
├── train_improved.py             # Training script
│
├── generate_combinations.py       # Generate v3-v102 combinations
├── orchestrator.py               # Multi-GPU orchestration
├── run_all.sh                    # Master execution script
│
├── combinations.json             # Generated: 100 version configs
├── execution_plan.json           # Generated: GPU assignment plan
│
├── checkpoints_improvements/     # Generated: Model checkpoints
│   └── v{N}/
│       ├── best_model.pth
│       └── results.json
│
├── logs_improvements/            # Generated: Training logs
│   ├── orchestrator.log
│   └── v{N}_gpu{G}.log
│
└── results_improvements/         # Generated: Final results
    ├── all_results.json          # All version results
    ├── performance_matrix.csv    # CSV for analysis
    ├── technique_ranking.json    # Technique impact ranking
    └── failed_jobs.json          # Failed runs (if any)
```

---

## 🚀 Usage

### 1. Generate Combinations

```bash
python3 generate_combinations.py
```

**Output**: `combinations.json` with 100 version configurations

### 2. Run Single Version (Testing)

```bash
python3 train_improved.py \
    --techniques "1,2,3" \
    --version 42 \
    --epochs 100 \
    --gpu_id 0
```

### 3. Run All Versions (Production)

```bash
# Default: 4 GPUs, 2 jobs per GPU, 100 epochs
./run_all.sh

# Custom configuration
EPOCHS=200 JOBS_PER_GPU=1 ./run_all.sh
```

**Environment Variables**:
- `COMPLEXITY`: Dataset complexity (default: `simple`)
- `EPOCHS`: Training epochs (default: `100`)
- `JOBS_PER_GPU`: Concurrent jobs per GPU (default: `2`)
- `D_MODEL`: Model dimension (default: `128`)
- `NUM_LAYERS`: Transformer layers (default: `4`)

### 4. Monitor Progress

```bash
# Overall progress
tail -f logs_improvements/orchestrator.log

# Specific version
tail -f logs_improvements/v42_gpu2.log

# GPU utilization
watch -n 1 nvidia-smi
```

### 5. Analyze Results

```bash
# View all results
cat results_improvements/all_results.json | jq

# Technique ranking
cat results_improvements/technique_ranking.json

# Performance matrix (CSV)
cat results_improvements/performance_matrix.csv

# Failed jobs (if any)
cat results_improvements/failed_jobs.json
```

---

## 📊 Output Files

### `all_results.json`
```json
{
  "v3": {
    "version": 3,
    "techniques": [1],
    "best_psnr": 42.5,
    "final_psnr": 41.8,
    "training_time": 320.5,
    "complexity": "simple"
  },
  ...
}
```

### `performance_matrix.csv`
```csv
Version,Techniques,Best_PSNR,Final_PSNR,Training_Time,Complexity
v3,1,42.50,41.80,5.3,simple
v4,2,43.20,42.90,5.1,simple
...
```

### `technique_ranking.json`
```json
[
  {
    "technique_id": 1,
    "technique_name": "Rectified Flow",
    "avg_improvement_db": 6.5,
    "num_samples": 10
  },
  ...
]
```

---

## 🔍 GPU Allocation

**4 GPUs × 2 jobs per GPU = 8 parallel experiments**

```
┌─────────┬──────────┬──────────┐
│  GPU 0  │   v3     │   v4     │
├─────────┼──────────┼──────────┤
│  GPU 1  │   v5     │   v6     │
├─────────┼──────────┼──────────┤
│  GPU 2  │   v7     │   v8     │
├─────────┼──────────┼──────────┤
│  GPU 3  │   v9     │   v10    │
└─────────┴──────────┴──────────┘

Jobs completed → Next jobs queued automatically
```

**Auto-scheduling**: Workers pull from queue when idle, ensuring full GPU utilization

---

## 🛡️ Error Handling

### Graceful Failure
- Failed jobs logged to `failed_jobs.json`
- Training continues for remaining versions
- No cascade failures

### Interrupt Handling
- `Ctrl+C`: Gracefully stops all workers
- Currently running jobs terminated
- Completed results saved

### Resume Capability
- Check `failed_jobs.json` for failures
- Re-run specific versions:
  ```bash
  python3 train_improved.py --version 42 --techniques "1,2,3" --gpu_id 0
  ```

---

## 📈 Expected Performance

### Baseline (V2 Transformer)
- **Simple**: ~35 dB PSNR
- **Multi-frequency**: ~30 dB PSNR
- **Radial**: ~32 dB PSNR

### Projected Best Combination
- **Techniques**: [1, 2, 3, 4, 5] (top 5)
- **Expected**: +10-15 dB improvement
- **Target**: 45-50 dB PSNR on simple dataset

### Success Criteria
- ✅ 90%+ completion rate (≥90/100 versions)
- ✅ ≥3 techniques show >2 dB improvement individually
- ✅ Best combination achieves >8 dB improvement

---

## 🔬 Analysis Workflows

### 1. Find Best Single Technique
```python
import json

with open('results_improvements/technique_ranking.json') as f:
    ranking = json.load(f)

print("Top 3 individual techniques:")
for i, tech in enumerate(ranking[:3], 1):
    print(f"{i}. Technique {tech['technique_id']}: +{tech['avg_improvement_db']:.2f} dB")
```

### 2. Find Best Combination
```python
import json

with open('results_improvements/all_results.json') as f:
    results = json.load(f)

# Sort by PSNR
sorted_results = sorted(results.items(), key=lambda x: x[1]['final_psnr'], reverse=True)

print("Top 5 combinations:")
for i, (version, result) in enumerate(sorted_results[:5], 1):
    print(f"{i}. {version}: {result['final_psnr']:.2f} dB | Techniques: {result['techniques']}")
```

### 3. Technique Interaction Analysis
```python
import json
import pandas as pd

# Load results
with open('results_improvements/all_results.json') as f:
    results = json.load(f)

# Convert to DataFrame
data = []
for version, result in results.items():
    row = {
        'version': version,
        'psnr': result['final_psnr'],
    }
    # One-hot encode techniques
    for i in range(1, 11):
        row[f'tech_{i}'] = 1 if i in result['techniques'] else 0
    data.append(row)

df = pd.DataFrame(data)

# Correlation matrix
corr = df[[f'tech_{i}' for i in range(1, 11)] + ['psnr']].corr()
print("Technique-PSNR correlations:")
print(corr['psnr'].sort_values(ascending=False))
```

---

## 🎯 Next Steps After Completion

### 1. Identify Top Performers
```bash
python3 -c "
import json
with open('results_improvements/all_results.json') as f:
    results = json.load(f)
sorted_r = sorted(results.items(), key=lambda x: x[1]['final_psnr'], reverse=True)
for v, r in sorted_r[:10]:
    print(f'{v}: {r[\"final_psnr\"]:.2f} dB | {r[\"techniques\"]}')"
```

### 2. Re-train Top 10 with More Epochs
```bash
# Example: Re-train v42 with 500 epochs
python3 train_improved.py \
    --version 42 \
    --techniques "1,2,3,5" \
    --epochs 500 \
    --save_dir checkpoints_final
```

### 3. Test Generalization
```bash
# Test best combination on all 6 complexities
for complexity in simple multi_frequency radial interference modulated composite; do
    python3 train_improved.py \
        --version 999 \
        --techniques "1,2,3,4,5" \
        --complexity $complexity \
        --epochs 200
done
```

### 4. Compare to V2 Baseline
```bash
cd ../evaluation
python3 compare_methods.py \
    --complexities simple \
    --methods_dir ../improvements/checkpoints_improvements
```

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: Reduce `JOBS_PER_GPU`:
```bash
JOBS_PER_GPU=1 ./run_all.sh
```

### Issue: Training Hangs
**Check GPU utilization**:
```bash
nvidia-smi
```
**Kill specific job**:
```bash
kill $(cat logs_improvements/v42_gpu2.log.pid)
```

### Issue: Import Errors
**Verify Python path**:
```bash
python3 -c "import sys; print(sys.path)"
python3 -c "from synthetic_experiments.improvements.techniques import TECHNIQUES"
```

### Issue: Many Failed Jobs
**Check failed_jobs.json**:
```bash
cat results_improvements/failed_jobs.json
```
**Common causes**:
- Incompatible technique combinations
- GPU memory issues
- Missing dependencies

---

## 📚 References

### Techniques
1. **Rectified Flow**: Liu et al., "Flow Straight and Fast" (2023)
2. **Multi-Scale PE**: Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions" (2020)
3. **Consistency Models**: Song et al., "Consistency Models" (2023)
4. **Perceptual Loss**: Johnson et al., "Perceptual Losses for Real-Time Style Transfer" (2016)
5. **AdaLN**: Peebles & Xie, "DiT: Scalable Diffusion Models with Transformers" (2023)
6. **Self-Conditioning**: Chen et al., "Analog Bits" (2023)
7. **EMA**: Standard in diffusion models
8. **Noise Schedules**: Nichol & Dhariwal, "Improved DDPM" (2021)
9. **Optimization**: Standard techniques
10. **Stochastic Depth**: Huang et al., "Deep Networks with Stochastic Depth" (2016)

---

## ✅ Checklist

Before running:
- [ ] GPU drivers installed and working (`nvidia-smi`)
- [ ] PyTorch with CUDA support installed
- [ ] Sufficient disk space (~10 GB for checkpoints)
- [ ] Review `IMPROVEMENT_TECHNIQUES.md`

After completion:
- [ ] Check success rate (`results_improvements/all_results.json`)
- [ ] Review technique ranking (`results_improvements/technique_ranking.json`)
- [ ] Identify top 10 combinations
- [ ] Re-train top combinations with 500 epochs
- [ ] Test generalization across all complexities
- [ ] Update main README with best results

---

**Status**: Ready for execution
**Estimated Time**: 30-40 hours for full suite
**GPUs Required**: 4 (or adjust `JOBS_PER_GPU`)
