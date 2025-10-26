#!/bin/bash

# ============================================================================
# Synthetic MAMBA V1 Training Runner Script
#
# Trains baseline MAMBA flow matching on synthetic sinusoidal datasets
# with 5%+5% disjoint sampling strategy
# ============================================================================

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Default parameters (can be overridden via environment variables)
COMPLEXITY=${COMPLEXITY:-"simple"}
RESOLUTION=${RESOLUTION:-32}
NUM_SAMPLES=${NUM_SAMPLES:-500}
NOISE_LEVEL=${NOISE_LEVEL:-0.0}
TRAIN_SPARSITY=${TRAIN_SPARSITY:-0.05}
TEST_SPARSITY=${TEST_SPARSITY:-0.05}

D_MODEL=${D_MODEL:-128}
NUM_LAYERS=${NUM_LAYERS:-4}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-32}
LR=${LR:-1e-3}

DEVICE=${DEVICE:-"auto"}
SAVE_DIR=${SAVE_DIR:-"${SCRIPT_DIR}/checkpoints"}

# Log file with timestamp and complexity
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${COMPLEXITY}_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/training_${COMPLEXITY}.pid"

# Print configuration
echo "============================================================"
echo "Synthetic MAMBA V1 Training Runner"
echo "============================================================"
echo "Script directory: ${SCRIPT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "PID file: ${PID_FILE}"
echo ""
echo "Dataset Configuration:"
echo "  Complexity: ${COMPLEXITY}"
echo "  Resolution: ${RESOLUTION}x${RESOLUTION}"
echo "  Number of samples: ${NUM_SAMPLES}"
echo "  Noise level: ${NOISE_LEVEL}"
echo "  Train sparsity: ${TRAIN_SPARSITY} ($(python3 -c "print(int(${TRAIN_SPARSITY} * ${RESOLUTION} * ${RESOLUTION}))")  pixels)"
echo "  Test sparsity: ${TEST_SPARSITY} ($(python3 -c "print(int(${TEST_SPARSITY} * ${RESOLUTION} * ${RESOLUTION}))")  pixels, disjoint)"
echo ""
echo "Model Configuration:"
echo "  Device: ${DEVICE}"
echo "  Model dimension (d_model): ${D_MODEL}"
echo "  Number of layers: ${NUM_LAYERS}"
echo ""
echo "Training Configuration:"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LR}"
echo "  Save directory: ${SAVE_DIR}"
echo "============================================================"
echo ""

# Check if already running
if [ -f "${PID_FILE}" ]; then
    OLD_PID=$(cat "${PID_FILE}")
    if ps -p ${OLD_PID} > /dev/null 2>&1; then
        echo "⚠️  Training already running with PID ${OLD_PID}"
        echo "To stop it, run: kill ${OLD_PID}"
        exit 1
    else
        echo "⚠️  Removing stale PID file"
        rm "${PID_FILE}"
    fi
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Training interrupted"
    if [ -f "${PID_FILE}" ]; then
        rm "${PID_FILE}"
    fi
}

trap cleanup EXIT INT TERM

# Run training
echo "🚀 Starting training..."
echo "📝 Output will be logged to: ${LOG_FILE}"
echo "💡 To monitor progress: tail -f ${LOG_FILE}"
echo ""

# Run in background with nohup
nohup python3 "${SCRIPT_DIR}/train_mamba_v1.py" \
    --complexity "${COMPLEXITY}" \
    --resolution ${RESOLUTION} \
    --num_samples ${NUM_SAMPLES} \
    --noise_level ${NOISE_LEVEL} \
    --train_sparsity ${TRAIN_SPARSITY} \
    --test_sparsity ${TEST_SPARSITY} \
    --d_model ${D_MODEL} \
    --num_layers ${NUM_LAYERS} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --device "${DEVICE}" \
    --save_dir "${SAVE_DIR}" \
    > "${LOG_FILE}" 2>&1 &

# Save PID
TRAINING_PID=$!
echo ${TRAINING_PID} > "${PID_FILE}"

echo "✅ Training started with PID: ${TRAINING_PID}"
echo ""
echo "📊 Commands:"
echo "  Monitor: tail -f ${LOG_FILE}"
echo "  Stop: kill ${TRAINING_PID}"
echo "  Check status: ps -p ${TRAINING_PID}"
echo ""

# Wait a moment and check if process is still running
sleep 2
if ! ps -p ${TRAINING_PID} > /dev/null 2>&1; then
    echo "❌ Training process failed to start. Check log file:"
    echo "   ${LOG_FILE}"
    tail -20 "${LOG_FILE}"
    exit 1
fi

echo "✅ Training is running in background"
echo ""
echo "============================================================"
echo "Quick Reference - Training Complexities"
echo "============================================================"
echo "COMPLEXITY=simple              # Single frequency (PSNR: 35-40 dB)"
echo "COMPLEXITY=multi_frequency     # 2-3 frequencies (PSNR: 30-35 dB)"
echo "COMPLEXITY=radial              # Circular patterns (PSNR: 32-37 dB)"
echo "COMPLEXITY=interference        # Wave beating (PSNR: 28-33 dB)"
echo "COMPLEXITY=modulated           # AM/FM modulation (PSNR: 25-30 dB)"
echo "COMPLEXITY=composite           # Complex multi-component (PSNR: 22-28 dB)"
echo "============================================================"
echo ""
echo "Example usage:"
echo "  COMPLEXITY=radial EPOCHS=200 ./run_mamba_v1_training.sh"
echo "  TRAIN_SPARSITY=0.1 TEST_SPARSITY=0.1 ./run_mamba_v1_training.sh"
echo ""
