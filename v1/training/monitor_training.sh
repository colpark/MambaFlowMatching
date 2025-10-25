#!/bin/bash

# ============================================================================
# Monitor MAMBA Training Script
# ============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_FILE="${SCRIPT_DIR}/training_output.log"
PID_FILE="${SCRIPT_DIR}/training.pid"

echo "============================================================"
echo "MAMBA Training Monitor"
echo "============================================================"

# Check if training is running
if [ -f "${PID_FILE}" ]; then
    TRAINING_PID=$(cat "${PID_FILE}")
    if ps -p "${TRAINING_PID}" > /dev/null 2>&1; then
        echo "✅ Training is running (PID: ${TRAINING_PID})"

        # Get process info
        echo ""
        echo "Process Info:"
        ps -p ${TRAINING_PID} -o pid,ppid,user,%cpu,%mem,etime,cmd

    else
        echo "❌ Training process (PID: ${TRAINING_PID}) is not running"
        echo "   (PID file exists but process is dead)"
    fi
else
    echo "❌ No training process found (no PID file)"
fi

echo ""
echo "============================================================"

# Check if log file exists
if [ -f "${LOG_FILE}" ]; then
    echo "📊 Latest training output:"
    echo "============================================================"
    tail -n 30 "${LOG_FILE}"
    echo "============================================================"
    echo ""
    echo "📈 To continuously monitor training:"
    echo "   tail -f ${LOG_FILE}"
else
    echo "⚠️  No log file found at: ${LOG_FILE}"
fi

echo ""
echo "============================================================"
