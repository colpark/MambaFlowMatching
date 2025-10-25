#!/bin/bash

# ============================================================================
# MAMBA V3 Training Runner Script
#
# V3: Space-filling curve ordering for better spatial locality
# Same architecture as V1, just better sequence ordering using Morton curves
# ============================================================================

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_FILE="${SCRIPT_DIR}/training_v3_output.log"
PID_FILE="${SCRIPT_DIR}/training_v3.pid"

# Default parameters (can be overridden)
EPOCHS=${EPOCHS:-1000}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-1e-4}
SAVE_EVERY=${SAVE_EVERY:-10}
EVAL_EVERY=${EVAL_EVERY:-10}
VISUALIZE_EVERY=${VISUALIZE_EVERY:-50}
SAVE_DIR=${SAVE_DIR:-"checkpoints_mamba_v3"}
D_MODEL=${D_MODEL:-512}  # Same as V1
NUM_LAYERS=${NUM_LAYERS:-6}  # Same as V1
NUM_WORKERS=${NUM_WORKERS:-4}
DEVICE=${DEVICE:-"auto"}

# Print configuration
echo "============================================================"
echo "MAMBA Diffusion V3 Training Runner"
echo "============================================================"
echo "Script directory: ${SCRIPT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "PID file: ${PID_FILE}"
echo ""
echo "V3 Improvement:"
echo "  ✓ Morton (Z-order) curve for spatial locality"
echo "  ✓ Neighboring pixels in 2D are neighbors in 1D"
echo "  ✓ Better MAMBA state propagation"
echo "  ✓ Same architecture as V1, just better ordering"
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
        echo "⚠️  V3 training is already running (PID: ${OLD_PID})"
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
echo "🚀 Starting V3 training in background..."
echo "   Logs will be written to: ${LOG_FILE}"
echo ""

nohup python train_mamba_v3_morton.py \
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

echo "✅ V3 training started successfully!"
echo "   PID: ${TRAINING_PID}"
echo ""
echo "📊 Monitor progress:"
echo "   tail -f ${LOG_FILE}"
echo ""
echo "🛑 Stop training:"
echo "   kill ${TRAINING_PID}"
echo ""
echo "📁 Checkpoints will be saved to: ${SAVE_DIR}/"
echo "   - mamba_v3_best.pth (best validation loss)"
echo "   - mamba_v3_latest.pth (latest epoch)"
echo "   - mamba_v3_epoch_XXXX.pth (every ${SAVE_EVERY} epochs)"
echo ""
echo "🖼️  Visualizations will be saved every ${VISUALIZE_EVERY} epochs"
echo ""
echo "Expected improvements over V1:"
echo "  - Better spatial coherence from Morton ordering"
echo "  - Reduced artifacts from spatially-aware MAMBA processing"
echo "  - Same computational cost as V1"
echo ""
echo "============================================================"
