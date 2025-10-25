#!/bin/bash

# ============================================================================
# MAMBA Training Runner Script
#
# This script runs the training in the background with nohup, so it continues
# even if the terminal closes or the screen turns off.
# ============================================================================

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_FILE="${SCRIPT_DIR}/training_output.log"
PID_FILE="${SCRIPT_DIR}/training.pid"

# Default parameters (can be overridden)
EPOCHS=${EPOCHS:-1000}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-1e-4}
SAVE_EVERY=${SAVE_EVERY:-10}
EVAL_EVERY=${EVAL_EVERY:-10}
VISUALIZE_EVERY=${VISUALIZE_EVERY:-50}
SAVE_DIR=${SAVE_DIR:-"checkpoints_mamba"}
D_MODEL=${D_MODEL:-512}
NUM_LAYERS=${NUM_LAYERS:-6}
NUM_WORKERS=${NUM_WORKERS:-4}
DEVICE=${DEVICE:-"auto"}

# Print configuration
echo "============================================================"
echo "MAMBA Diffusion Training Runner"
echo "============================================================"
echo "Script directory: ${SCRIPT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "PID file: ${PID_FILE}"
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
        echo "⚠️  Training is already running (PID: ${OLD_PID})"
        echo "   To stop it, run: kill ${OLD_PID}"
        echo "   Or use: ./stop_mamba_training.sh"
        exit 1
    else
        echo "🧹 Cleaning up stale PID file..."
        rm "${PID_FILE}"
    fi
fi

# Navigate to script directory
cd "${SCRIPT_DIR}"

# Start training in background with nohup
echo "🚀 Starting training in background..."
echo "   Logs will be written to: ${LOG_FILE}"
echo ""

nohup python train_mamba_standalone.py \
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

echo "✅ Training started successfully!"
echo "   PID: ${TRAINING_PID}"
echo ""
echo "📊 Monitor progress:"
echo "   tail -f ${LOG_FILE}"
echo ""
echo "🛑 Stop training:"
echo "   kill ${TRAINING_PID}"
echo "   Or use: ./stop_mamba_training.sh"
echo ""
echo "📁 Checkpoints will be saved to: ${SAVE_DIR}/"
echo "   - mamba_best.pth (best validation loss)"
echo "   - mamba_latest.pth (latest epoch)"
echo "   - mamba_epoch_XXXX.pth (every ${SAVE_EVERY} epochs)"
echo ""
echo "🖼️  Visualizations will be saved every ${VISUALIZE_EVERY} epochs"
echo ""
echo "The training will continue running even if you:"
echo "  - Close this terminal"
echo "  - Log out of SSH"
echo "  - Turn off your screen"
echo "  - Put your computer to sleep"
echo ""
echo "============================================================"
