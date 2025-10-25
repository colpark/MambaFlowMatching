#!/bin/bash

# ============================================================================
# Stop MAMBA Training Script
# ============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PID_FILE="${SCRIPT_DIR}/training.pid"

echo "============================================================"
echo "Stopping MAMBA Training"
echo "============================================================"

if [ ! -f "${PID_FILE}" ]; then
    echo "❌ No training process found (no PID file)"
    exit 1
fi

TRAINING_PID=$(cat "${PID_FILE}")

if ps -p "${TRAINING_PID}" > /dev/null 2>&1; then
    echo "🛑 Stopping training process (PID: ${TRAINING_PID})..."
    kill ${TRAINING_PID}

    # Wait for process to stop
    sleep 2

    if ps -p "${TRAINING_PID}" > /dev/null 2>&1; then
        echo "⚠️  Process still running, force killing..."
        kill -9 ${TRAINING_PID}
    fi

    echo "✅ Training stopped successfully"
else
    echo "⚠️  Process ${TRAINING_PID} is not running"
fi

rm "${PID_FILE}"
echo "🧹 Cleaned up PID file"
echo "============================================================"
