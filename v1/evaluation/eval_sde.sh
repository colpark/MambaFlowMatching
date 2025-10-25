#!/bin/bash

# ============================================================================
# SDE Multi-Scale Super-Resolution Evaluation Runner
#
# Evaluates MAMBA model with SDE sampling for smoother, less speckled results
# Compares ODE (Heun), SDE, and DDIM samplers at multiple resolutions
# ============================================================================

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default parameters (can be overridden via environment variables)
CHECKPOINT=${CHECKPOINT:-"checkpoints_mamba/mamba_best.pth"}
RESOLUTIONS=${RESOLUTIONS:-"32 64 96"}
SAMPLERS=${SAMPLERS:-"heun sde ddim"}
NUM_SAMPLES=${NUM_SAMPLES:-10}
NUM_STEPS=${NUM_STEPS:-50}
TEMPERATURE=${TEMPERATURE:-0.5}
ETA=${ETA:-0.3}
SAVE_DIR=${SAVE_DIR:-"eval_sde_multiscale"}
D_MODEL=${D_MODEL:-512}
NUM_LAYERS=${NUM_LAYERS:-6}
DEVICE=${DEVICE:-"auto"}

# Print configuration
echo "============================================================"
echo "SDE Multi-Scale Super-Resolution Evaluation"
echo "============================================================"
echo "Script directory: ${SCRIPT_DIR}"
echo ""
echo "Configuration:"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Resolutions: ${RESOLUTIONS}"
echo "  Samplers: ${SAMPLERS}"
echo "  Number of samples: ${NUM_SAMPLES}"
echo "  Sampling steps: ${NUM_STEPS}"
echo "  SDE temperature: ${TEMPERATURE}"
echo "  DDIM eta: ${ETA}"
echo "  Save directory: ${SAVE_DIR}"
echo "  Model dimension: ${D_MODEL}"
echo "  Number of layers: ${NUM_LAYERS}"
echo "  Device: ${DEVICE}"
echo "============================================================"
echo ""
echo "Sampler Details:"
echo "  🔹 HEUN: Standard ODE solver (deterministic, baseline)"
echo "  🌟 SDE: Stochastic sampler (reduces speckles, smoother)"
echo "  ⚡ DDIM: Non-uniform timesteps (faster, configurable)"
echo ""
echo "Key Benefits of SDE:"
echo "  ✓ Reduces background speckle artifacts"
echo "  ✓ Smoother, more coherent results"
echo "  ✓ Temperature controls exploration (0.5 recommended)"
echo "============================================================"
echo ""

# Check if checkpoint exists
if [ ! -f "${SCRIPT_DIR}/${CHECKPOINT}" ]; then
    echo "❌ Error: Checkpoint not found at ${CHECKPOINT}"
    echo ""
    echo "Available checkpoints:"
    ls -lh "${SCRIPT_DIR}/checkpoints_mamba/" 2>/dev/null || echo "  No checkpoints found"
    echo ""
    echo "Please specify checkpoint path with:"
    echo "  CHECKPOINT=path/to/checkpoint.pth ./eval_sde.sh"
    exit 1
fi

# Navigate to script directory
cd "${SCRIPT_DIR}"

# Run evaluation
echo "🚀 Starting SDE multi-scale evaluation..."
echo ""

python eval_sde_multiscale.py \
    --checkpoint ${CHECKPOINT} \
    --resolutions ${RESOLUTIONS} \
    --samplers ${SAMPLERS} \
    --num_samples ${NUM_SAMPLES} \
    --num_steps ${NUM_STEPS} \
    --temperature ${TEMPERATURE} \
    --eta ${ETA} \
    --save_dir ${SAVE_DIR} \
    --d_model ${D_MODEL} \
    --num_layers ${NUM_LAYERS} \
    --device ${DEVICE}

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Evaluation completed successfully!"
    echo "============================================================"
    echo ""
    echo "📊 Results saved to: ${SAVE_DIR}/"
    echo "   - metrics_comparison.txt (detailed metrics)"
    echo "   - multiscale_comparison.png (visual comparison)"
    echo "   - performance_chart.png (bar charts)"
    echo ""
    echo "📈 Key files:"
    echo "   - View metrics: cat ${SAVE_DIR}/metrics_comparison.txt"
    echo "   - View images: open ${SAVE_DIR}/*.png"
    echo ""
    echo "🔍 Analysis Tips:"
    echo "   - Check which sampler has highest PSNR/SSIM"
    echo "   - Compare visual quality of SDE vs Heun"
    echo "   - Look for reduced speckle artifacts in SDE"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Evaluation failed with exit code ${EXIT_CODE}"
    echo "============================================================"
    exit ${EXIT_CODE}
fi
