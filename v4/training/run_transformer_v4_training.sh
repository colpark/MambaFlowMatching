#!/bin/bash

# ============================================================================
# Transformer V4 Training Runner Script
#
# V4: Standard Transformer encoder instead of MAMBA
# Compares quadratic O(N²) attention vs MAMBA's linear O(N) complexity
# ============================================================================

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_FILE="${SCRIPT_DIR}/training_v4_output.log"
PID_FILE="${SCRIPT_DIR}/training_v4.pid"

# Default parameters (can be overridden)
EPOCHS=${EPOCHS:-1000}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-1e-4}
SAVE_EVERY=${SAVE_EVERY:-10}
EVAL_EVERY=${EVAL_EVERY:-10}
VISUALIZE_EVERY=${VISUALIZE_EVERY:-50}
SAVE_DIR=${SAVE_DIR:-"checkpoints_transformer_v4"}
D_MODEL=${D_MODEL:-512}  # Same as V1
NUM_LAYERS=${NUM_LAYERS:-6}  # Same as V1
NUM_HEADS=${NUM_HEADS:-8}  # Multi-head attention
DIM_FEEDFORWARD=${DIM_FEEDFORWARD:-2048}  # FFN dimension
NUM_WORKERS=${NUM_WORKERS:-4}
DEVICE=${DEVICE:-"auto"}

# Print configuration
echo "============================================================"
echo "Transformer V4 Training Runner"
echo "============================================================"
echo "Script directory: ${SCRIPT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "PID file: ${PID_FILE}"
echo ""
echo "V4 Architecture:"
echo "  ✓ Standard Transformer encoder (replaces MAMBA)"
echo "  ✓ Multi-head self-attention: O(N²) complexity"
echo "  ✓ Global context vs MAMBA's sequential state"
echo "  ✓ Same number of layers as V1 for fair comparison"
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
echo "  Attention heads: ${NUM_HEADS}"
echo "  FFN dimension: ${DIM_FEEDFORWARD}"
echo "  DataLoader workers: ${NUM_WORKERS}"
echo "============================================================"
echo ""

# Check if training is already running
if [ -f "${PID_FILE}" ]; then
    OLD_PID=$(cat "${PID_FILE}")
    if ps -p "${OLD_PID}" > /dev/null 2>&1; then
        echo "⚠️  V4 training is already running (PID: ${OLD_PID})"
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
echo "🚀 Starting V4 training in background..."
echo "   Logs will be written to: ${LOG_FILE}"
echo ""

nohup python train_transformer_v4.py \
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
    --num_heads ${NUM_HEADS} \
    --dim_feedforward ${DIM_FEEDFORWARD} \
    --num_workers ${NUM_WORKERS} \
    > "${LOG_FILE}" 2>&1 &

# Save PID
TRAINING_PID=$!
echo ${TRAINING_PID} > "${PID_FILE}"

echo "✅ V4 training started successfully!"
echo "   PID: ${TRAINING_PID}"
echo ""
echo "📊 Monitor progress:"
echo "   tail -f ${LOG_FILE}"
echo ""
echo "🛑 Stop training:"
echo "   kill ${TRAINING_PID}"
echo ""
echo "📁 Checkpoints will be saved to: ${SAVE_DIR}/"
echo "   - transformer_v4_best.pth (best validation loss)"
echo "   - transformer_v4_latest.pth (latest epoch)"
echo "   - transformer_v4_epoch_XXXX.pth (every ${SAVE_EVERY} epochs)"
echo ""
echo "🖼️  Visualizations will be saved every ${VISUALIZE_EVERY} epochs"
echo ""
echo "Comparison with MAMBA V1:"
echo "  - Same architecture depth (${NUM_LAYERS} layers)"
echo "  - Same model dimension (${D_MODEL})"
echo "  - Transformer: O(N²) global attention"
echo "  - MAMBA: O(N) linear state space"
echo "  - This comparison will show if global context helps sparse neural fields"
echo ""
echo "============================================================"
