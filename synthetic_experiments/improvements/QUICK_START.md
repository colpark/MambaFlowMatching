# Quick Start Guide - 100 Improvement Experiments

**Goal**: Systematically test 10 improvement techniques in 100 combinations to find the best configuration.

---

## 🚀 TL;DR

```bash
cd synthetic_experiments/improvements

# 1. Test setup (30 seconds)
./test_setup.sh

# 2. Run all experiments (30-40 hours)
./run_all.sh

# 3. Analyze results
python3 analyze_results.py
```

---

## 📋 What This Does

- **Generates 100 versions** (v3 to v102) with different technique combinations
- **Trains all 100** in parallel across 4 GPUs (2 jobs per GPU)
- **Tracks performance** and ranks techniques by impact
- **Produces visualizations** showing technique interactions

---

## 🎯 10 Techniques Being Tested

| ID | Technique | Expected Gain |
|----|-----------|---------------|
| 1 | Rectified Flow | +6.5 dB |
| 2 | Multi-Scale Positional Encoding | +5.5 dB |
| 3 | Consistency Loss | +4.5 dB |
| 4 | Perceptual Loss (Frequency) | +4.0 dB |
| 5 | Adaptive LayerNorm | +3.5 dB |
| 6 | Self-Conditioning | +3.0 dB |
| 7 | Exponential Moving Average | +2.0 dB |
| 8 | Noise Schedule (Cosine) | +2.0 dB |
| 9 | Gradient Clipping + Warmup | +1.5 dB |
| 10 | Stochastic Depth (DropPath) | +1.5 dB |

---

## 📊 Version Distribution

- **v3-v12**: Each technique individually (10 versions)
- **v13-v57**: All pairs of techniques (45 versions)
- **v58-v92**: Selected triplets (35 versions)
- **v93-v102**: 4+ technique combinations (10 versions)

---

## 🔧 Prerequisites

```bash
# Python packages
pip install torch torchvision matplotlib seaborn pandas

# GPU check
nvidia-smi  # Should show 4 GPUs
```

---

## 📁 Key Files

```
improvements/
├── run_all.sh                # Master script - START HERE
├── test_setup.sh             # Quick validation
│
├── techniques.py             # Technique implementations
├── improved_transformer.py   # Model with pluggable techniques
├── train_improved.py         # Training script
│
├── generate_combinations.py  # Generates v3-v102
├── orchestrator.py          # Multi-GPU manager
├── analyze_results.py       # Results visualization
│
└── README.md                # Full documentation
```

---

## 🎬 Step-by-Step

### Step 1: Verify Setup (30 seconds)

```bash
./test_setup.sh
```

**Expected output**:
```
✅ All Tests Passed!
✅ Found 4 GPU(s)
```

### Step 2: Run Experiments (30-40 hours)

```bash
./run_all.sh
```

**What happens**:
1. Generates `combinations.json` (100 configs)
2. Creates output directories
3. Launches 8 parallel workers (4 GPUs × 2 jobs each)
4. Trains all 100 versions
5. Saves results to `results_improvements/`

**Monitor progress**:
```bash
# Overall progress
tail -f logs_improvements/orchestrator.log

# Specific version
tail -f logs_improvements/v42_gpu2.log

# GPU usage
watch -n 1 nvidia-smi
```

### Step 3: Analyze Results

```bash
python3 analyze_results.py
```

**Generates**:
- `technique_heatmap.png`: Visual of all combinations
- `improvement_dist.png`: Distribution of improvements
- `technique_impact.png`: Individual technique rankings
- `combination_size.png`: Performance vs # of techniques
- `top_combinations.md`: Best 20 combinations

---

## 📈 Expected Results

**Baseline (V2 Transformer)**: ~35 dB PSNR on simple dataset

**Best Single Technique**: ~41-42 dB (+6-7 dB)

**Best Combination**: ~45-50 dB (+10-15 dB)

**Success Rate**: ≥90% (90+ versions complete)

---

## 🛑 Stopping/Resuming

**Stop gracefully**:
```bash
# Press Ctrl+C in run_all.sh terminal
# Workers will finish current jobs and stop
```

**Resume failed jobs**:
```bash
# Check which failed
cat results_improvements/failed_jobs.json

# Re-run specific version
python3 train_improved.py --version 42 --techniques "1,2,3" --gpu_id 0
```

---

## 🔍 Quick Analysis

**View top 10**:
```bash
python3 -c "
import json
with open('results_improvements/all_results.json') as f:
    r = json.load(f)
s = sorted(r.items(), key=lambda x: x[1]['final_psnr'], reverse=True)
for v, res in s[:10]:
    print(f'{v}: {res[\"final_psnr\"]:.2f} dB | {res[\"techniques\"]}')"
```

**Technique ranking**:
```bash
cat results_improvements/technique_ranking.json | jq
```

**Performance matrix**:
```bash
head -20 results_improvements/performance_matrix.csv
```

---

## 🎯 Customization

**Change dataset complexity**:
```bash
COMPLEXITY=radial ./run_all.sh
```

**More epochs for accuracy**:
```bash
EPOCHS=200 ./run_all.sh
```

**Fewer parallel jobs (memory)**:
```bash
JOBS_PER_GPU=1 ./run_all.sh
```

---

## ✅ Success Checklist

After completion:

- [ ] Check `all_results.json` - should have ~90-100 entries
- [ ] Review `technique_ranking.json` - identifies best individual techniques
- [ ] View `top_combinations.md` - shows best combinations
- [ ] Check `failed_jobs.json` - investigate any failures
- [ ] Re-train top 10 with 500 epochs for final validation
- [ ] Test best combination on all 6 complexity levels

---

## 🐛 Common Issues

**"CUDA out of memory"**:
```bash
JOBS_PER_GPU=1 ./run_all.sh
```

**"Import errors"**:
```bash
# Verify repository root in Python path
python3 -c "import sys; print(sys.path)"
```

**"Training hangs"**:
```bash
# Check GPU usage
nvidia-smi

# Kill specific job
kill $(pgrep -f "train_improved.py.*version 42")
```

---

## 📚 Further Reading

- `README.md`: Full documentation
- `IMPROVEMENT_TECHNIQUES.md`: Technique details
- `analyze_results.py`: Analysis code

---

**Status**: Ready to run
**Time Required**: ~30-40 hours
**Resources**: 4 GPUs, ~10 GB disk space

**Questions?** Check `README.md` or inspect the code!
