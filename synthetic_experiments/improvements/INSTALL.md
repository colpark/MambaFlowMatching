# Installation Guide

Complete setup instructions for the improvement experiment framework.

---

## Prerequisites

- **Python**: 3.8 or higher
- **GPUs**: 4 NVIDIA GPUs recommended (optional, works on CPU too)
- **Disk Space**: ~10 GB for checkpoints and results
- **Time**: ~30-40 hours for full 100-version suite

---

## 1. Install PyTorch

### For CUDA 11.8+ (Most Common)

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### For CUDA 12.1+

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### For CPU Only (Testing/Development)

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Verify Installation

```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

**Expected output** (with GPUs):
```
PyTorch 2.1.0+cu118
CUDA available: True
CUDA version: 11.8
GPU count: 4
```

---

## 2. Install Additional Dependencies

```bash
# Core dependencies
pip3 install numpy matplotlib seaborn pandas

# Optional but recommended
pip3 install jupyter tqdm
```

---

## 3. Verify GPU Drivers

```bash
nvidia-smi
```

**Expected output**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
|   0  NVIDIA A100-SXM...  Off  | 00000000:00:04.0 Off |                    0 |
|   1  NVIDIA A100-SXM...  Off  | 00000000:00:05.0 Off |                    0 |
|   2  NVIDIA A100-SXM...  Off  | 00000000:00:06.0 Off |                    0 |
|   3  NVIDIA A100-SXM...  Off  | 00000000:00:07.0 Off |                    0 |
+-----------------------------------------------------------------------------+
```

---

## 4. Clone Repository (If Not Already)

```bash
git clone https://github.com/colpark/MambaFlowMatching.git
cd MambaFlowMatching/synthetic_experiments/improvements
```

---

## 5. Run Tests

```bash
./test_setup.sh
```

**Expected output**:
```
============================================================
✅ All Tests Passed!
============================================================

Framework is ready for use. You can now run:
  ./run_all.sh

GPU Check:
  ✅ Found 4 GPU(s)
```

---

## 6. Quick Training Test (Optional)

Test a single version to verify everything works:

```bash
python3 train_improved.py \
    --techniques "1" \
    --version 999 \
    --epochs 10 \
    --num_samples 100 \
    --resolution 32 \
    --gpu_id 0
```

**Expected**: Training completes successfully, saves checkpoint to `checkpoints_improvements/v999/`

---

## 7. Ready to Run!

```bash
# Run all 100 experiments
./run_all.sh

# Or customize
EPOCHS=200 JOBS_PER_GPU=1 ./run_all.sh
```

---

## Troubleshooting

### Issue: "No module named 'torch'"

**Solution**: Install PyTorch (see step 1)

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "CUDA out of memory"

**Solution**: Reduce parallel jobs per GPU

```bash
JOBS_PER_GPU=1 ./run_all.sh
```

Or reduce model size:

```bash
D_MODEL=64 NUM_LAYERS=2 ./run_all.sh
```

### Issue: "RuntimeError: CUDA error: no kernel image is available"

**Solution**: PyTorch CUDA version mismatch. Check your CUDA version:

```bash
nvidia-smi  # Check "CUDA Version" in header
```

Then install matching PyTorch:
- CUDA 11.8: `--index-url https://download.pytorch.org/whl/cu118`
- CUDA 12.1: `--index-url https://download.pytorch.org/whl/cu121`

### Issue: "nvidia-smi: command not found"

**Solution**: Install NVIDIA drivers

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install nvidia-driver-525
sudo reboot
```

**CentOS/RHEL**:
```bash
sudo yum install nvidia-driver
sudo reboot
```

### Issue: Training very slow on GPU

**Check GPU utilization**:
```bash
watch -n 1 nvidia-smi
```

If GPU utilization is low (<50%), possible causes:
- Data loading bottleneck (increase `num_workers` if applicable)
- CPU preprocessing bottleneck (verify CPU usage)
- Small batch size (default 32 is usually fine)

### Issue: "ImportError: cannot import name 'SinusoidalDataset'"

**Solution**: Ensure you're in the correct directory and repository root is in Python path

```bash
cd MambaFlowMatching/synthetic_experiments/improvements
python3 -c "import sys; print(sys.path)"
```

Repository root should be in the path. The scripts handle this automatically.

---

## Performance Benchmarks

### Expected Training Times (per version)

| Configuration | Time per Epoch | 100 Epochs | Hardware |
|---------------|----------------|------------|----------|
| GPU (A100) | ~2 seconds | ~3-4 min | NVIDIA A100 40GB |
| GPU (V100) | ~4 seconds | ~6-7 min | NVIDIA V100 32GB |
| GPU (RTX 3090) | ~5 seconds | ~8-10 min | NVIDIA RTX 3090 |
| CPU (16 cores) | ~30 seconds | ~50 min | Intel Xeon |

### Full Suite (100 versions)

| Setup | Parallel Jobs | Wall-Clock Time |
|-------|---------------|-----------------|
| 4 GPUs × 2 jobs | 8 parallel | ~30-40 hours |
| 4 GPUs × 1 job | 4 parallel | ~50-60 hours |
| 1 GPU × 2 jobs | 2 parallel | ~150-200 hours |

---

## Environment Variables Reference

```bash
# Dataset configuration
COMPLEXITY=simple           # simple, multi_frequency, radial, interference, modulated, composite
RESOLUTION=32              # Image resolution (32, 64, 128)
NUM_SAMPLES=500            # Training samples per dataset

# Model architecture
D_MODEL=128                # Model dimension
NUM_LAYERS=4               # Number of transformer layers
NUM_HEADS=8                # Attention heads
DIM_FEEDFORWARD=512        # Feed-forward dimension

# Training
EPOCHS=100                 # Training epochs per version
LR=1e-3                    # Learning rate

# Hardware
JOBS_PER_GPU=2            # Concurrent jobs per GPU (1-3 recommended)
```

---

## Disk Space Management

### Estimate Required Space

```bash
# Checkpoints: ~50 MB per version
# 100 versions × 50 MB = ~5 GB

# Logs: ~5 MB per version
# 100 versions × 5 MB = ~500 MB

# Results: ~1 MB total

# Total: ~6 GB minimum
```

### Clean Up After Experiments

```bash
# Remove logs (keep checkpoints and results)
rm -rf logs_improvements/

# Remove test checkpoints
rm -rf checkpoints_test/

# Keep only best checkpoints (remove intermediate)
find checkpoints_improvements -name "epoch_*.pth" -delete
```

---

## Docker Installation (Optional)

If you prefer containerized setup:

```bash
# Create Dockerfile
cat > Dockerfile <<'EOF'
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace
COPY . .

RUN pip install matplotlib seaborn pandas jupyter tqdm

CMD ["bash"]
EOF

# Build image
docker build -t mamba-improvements .

# Run with GPU support
docker run --gpus all -it -v $(pwd):/workspace mamba-improvements

# Inside container
cd synthetic_experiments/improvements
./run_all.sh
```

---

## Cloud Platform Setup

### Google Colab

```python
# Install PyTorch (usually pre-installed)
!pip install torch torchvision

# Clone repository
!git clone https://github.com/colpark/MambaFlowMatching.git
%cd MambaFlowMatching/synthetic_experiments/improvements

# Run single experiment (Colab has 1 GPU)
!python3 train_improved.py --techniques "1,2,3" --version 42 --epochs 100
```

### AWS EC2 (p3.8xlarge - 4× V100)

```bash
# Launch instance with Deep Learning AMI
# SSH into instance

# PyTorch pre-installed, verify
python3 -c "import torch; print(torch.cuda.device_count())"

# Clone and run
git clone https://github.com/colpark/MambaFlowMatching.git
cd MambaFlowMatching/synthetic_experiments/improvements
./run_all.sh
```

### Paperspace Gradient

```bash
# PyTorch pre-installed in workspace
cd /notebooks

# Clone repository
git clone https://github.com/colpark/MambaFlowMatching.git
cd MambaFlowMatching/synthetic_experiments/improvements

# Run experiments
./run_all.sh
```

---

## Verification Checklist

Before starting full 100-version suite:

- [ ] PyTorch installed and CUDA working
- [ ] 4 GPUs detected via `nvidia-smi`
- [ ] `./test_setup.sh` passes all tests
- [ ] Quick training test (10 epochs) completes successfully
- [ ] Sufficient disk space (~10 GB free)
- [ ] Monitoring setup (tmux/screen session recommended)

---

## Next Steps After Installation

1. **Quick test**: `./test_setup.sh`
2. **Single version**: Train v42 for 10 epochs to verify
3. **Full suite**: `./run_all.sh`
4. **Monitor**: `tail -f logs_improvements/orchestrator.log`
5. **Analyze**: `python3 analyze_results.py` after completion

---

## Getting Help

- **Framework issues**: Check `README.md` and `QUICK_START.md`
- **PyTorch installation**: https://pytorch.org/get-started/locally/
- **CUDA setup**: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
- **General questions**: Review code comments and docstrings

---

**Installation complete!** You're ready to run the experiments. 🚀
