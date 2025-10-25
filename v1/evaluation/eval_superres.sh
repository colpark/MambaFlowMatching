#!/bin/bash

# ============================================================================
# Super-Resolution Evaluation Runner Script
#
# Evaluates the trained MAMBA model's zero-shot super-resolution capacity
# ============================================================================

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default parameters (can be overridden via environment variables)
CHECKPOINT=${CHECKPOINT:-"checkpoints_mamba/mamba_best.pth"}
RESOLUTIONS=${RESOLUTIONS:-"64 96 128 256"}
NUM_SAMPLES=${NUM_SAMPLES:-10}
NUM_STEPS=${NUM_STEPS:-50}
BATCH_SIZE=${BATCH_SIZE:-1}
SAVE_DIR=${SAVE_DIR:-"eval_superres"}
D_MODEL=${D_MODEL:-512}
NUM_LAYERS=${NUM_LAYERS:-6}
DEVICE=${DEVICE:-"auto"}

# Print configuration
echo "============================================================"
echo "Super-Resolution Evaluation"
echo "============================================================"
echo "Script directory: ${SCRIPT_DIR}"
echo ""
echo "Configuration:"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Target resolutions: ${RESOLUTIONS}"
echo "  Number of samples: ${NUM_SAMPLES}"
echo "  Sampling steps: ${NUM_STEPS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Save directory: ${SAVE_DIR}"
echo "  Model dimension: ${D_MODEL}"
echo "  Number of layers: ${NUM_LAYERS}"
echo "  Device: ${DEVICE}"
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
    echo "  CHECKPOINT=path/to/checkpoint.pth ./eval_superres.sh"
    exit 1
fi

# Navigate to script directory
cd "${SCRIPT_DIR}"

# Run evaluation
echo "🚀 Starting super-resolution evaluation..."
echo ""

python eval_superresolution.py \
    --checkpoint ${CHECKPOINT} \
    --resolutions ${RESOLUTIONS} \
    --num_samples ${NUM_SAMPLES} \
    --num_steps ${NUM_STEPS} \
    --batch_size ${BATCH_SIZE} \
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
    echo "   - metrics.txt (detailed metrics)"
    echo "   - superresolution_comparison.png (visual comparison)"
    echo "   - superres_64x64.png, superres_96x96.png, etc."
    echo ""
    echo "📈 Key files:"
    echo "   - View metrics: cat ${SAVE_DIR}/metrics.txt"
    echo "   - View images: open ${SAVE_DIR}/*.png"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Evaluation failed with exit code ${EXIT_CODE}"
    echo "============================================================"
    exit ${EXIT_CODE}
fi
