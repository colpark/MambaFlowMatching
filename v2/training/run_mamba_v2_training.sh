#!/bin/bash

# ============================================================================
# MAMBA V2 Training Runner Script
#
# Improved architecture with bidirectional MAMBA and lightweight perceiver
# Expected: 70-80% reduction in speckle artifacts
# ============================================================================

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_FILE="${SCRIPT_DIR}/training_v2_output.log"
PID_FILE="${SCRIPT_DIR}/training_v2.pid"

# Default parameters (can be overridden)
EPOCHS=${EPOCHS:-1000}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-1e-4}
SAVE_EVERY=${SAVE_EVERY:-10}
EVAL_EVERY=${EVAL_EVERY:-10}
VISUALIZE_EVERY=${VISUALIZE_EVERY:-50}
SAVE_DIR=${SAVE_DIR:-"checkpoints_mamba_v2"}
D_MODEL=${D_MODEL:-256}  # Changed from 512 to 256
NUM_LAYERS=${NUM_LAYERS:-8}  # Total layers: 4 forward + 4 backward
NUM_WORKERS=${NUM_WORKERS:-4}
DEVICE=${DEVICE:-"auto"}

# Print configuration
echo "============================================================"
echo "MAMBA Diffusion V2 Training Runner"
echo "============================================================"
echo "Script directory: ${SCRIPT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "PID file: ${PID_FILE}"
echo ""
echo "V2 Improvements:"
echo "  ✓ Bidirectional MAMBA (4 forward + 4 backward = 8 total layers)"
echo "  ✓ Lightweight Perceiver (2 iterations)"
echo "  ✓ Query self-attention for spatial coherence"
echo ""
echo "Training Configuration:"
echo "  Device: ${DEVICE}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LR}"
echo "  Save checkpoints every: ${SAVE_EVERY} epochs"
echo "  Evaluate every: ${EVAL_EVERY} epochs"
echo "  Visualize every: ${VISUALIZE_EVERY} epochs"
echo "  Save directory: ${SAVE_DIR}"
echo "  Model dimension: ${D_MODEL}"
echo "  Number of layers: ${NUM_LAYERS}"
echo "  DataLoader workers: ${NUM_WORKERS}"
echo "============================================================"
echo ""

# Check if training is already running
if [ -f "${PID_FILE}" ]; then
    OLD_PID=$(cat "${PID_FILE}")
    if ps -p "${OLD_PID}" > /dev/null 2>&1; then
        echo "⚠️  V2 training is already running (PID: ${OLD_PID})"
        echo "   To stop it, run: kill ${OLD_PID}"
        exit 1
    else
        echo "🧹 Cleaning up stale PID file..."
        rm "${PID_FILE}"
    fi
fi

# Navigate to script directory
cd "${SCRIPT_DIR}"

# Start training in background with nohup
echo "🚀 Starting V2 training in background..."
echo "   Logs will be written to: ${LOG_FILE}"
echo ""

nohup python train_mamba_v2.py \
    --device ${DEVICE} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --save_every ${SAVE_EVERY} \
    --eval_every ${EVAL_EVERY} \
    --visualize_every ${VISUALIZE_EVERY} \
    --save_dir ${SAVE_DIR} \
    --d_model ${D_MODEL} \
    --num_layers ${NUM_LAYERS} \
    --num_workers ${NUM_WORKERS} \
    > "${LOG_FILE}" 2>&1 &

# Save PID
TRAINING_PID=$!
echo ${TRAINING_PID} > "${PID_FILE}"

echo "✅ V2 training started successfully!"
echo "   PID: ${TRAINING_PID}"
echo ""
echo "📊 Monitor progress:"
echo "   tail -f ${LOG_FILE}"
echo ""
echo "🛑 Stop training:"
echo "   kill ${TRAINING_PID}"
echo ""
echo "📁 Checkpoints will be saved to: ${SAVE_DIR}/"
echo "   - mamba_v2_best.pth (best validation loss)"
echo "   - mamba_v2_latest.pth (latest epoch)"
echo "   - mamba_v2_epoch_XXXX.pth (every ${SAVE_EVERY} epochs)"
echo ""
echo "🖼️  Visualizations will be saved every ${VISUALIZE_EVERY} epochs"
echo ""
echo "Expected improvements over V1:"
echo "  - 70-80% reduction in speckle artifacts"
echo "  - +3-5 dB PSNR improvement"
echo "  - Smoother spatial fields"
echo ""
echo "============================================================"
