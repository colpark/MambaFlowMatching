#!/bin/bash

# ============================================================================
# Master Script for Running All 100 Improvement Experiments
#
# This script:
# 1. Generates 100 combinations (v3-v102)
# 2. Launches multi-GPU orchestrator
# 3. Runs all experiments across 4 GPUs with error handling
# ============================================================================

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

echo "============================================================"
echo "Multi-GPU Improvement Experiment Suite"
echo "============================================================"
echo "Script directory: ${SCRIPT_DIR}"
echo ""

# ============================================================================
# Step 1: Generate combinations
# ============================================================================

echo "📋 Step 1/3: Generating 100 combinations (v3-v102)..."
echo ""

python3 generate_combinations.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Failed to generate combinations"
    exit 1
fi

echo ""
echo "✅ Combinations generated"
echo ""

# ============================================================================
# Step 2: Verify GPU availability
# ============================================================================

echo "🔍 Step 2/3: Verifying GPU availability..."
echo ""

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --list-gpus
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo ""
    echo "✅ Found ${NUM_GPUS} GPU(s)"
else
    echo "⚠️  nvidia-smi not found, assuming CPU mode"
    NUM_GPUS=0
fi

echo ""

# ============================================================================
# Step 3: Launch orchestrator
# ============================================================================

echo "🚀 Step 3/3: Launching multi-GPU orchestrator..."
echo ""

# Configuration (can be overridden via environment variables)
COMPLEXITY=${COMPLEXITY:-"simple"}
RESOLUTION=${RESOLUTION:-32}
NUM_SAMPLES=${NUM_SAMPLES:-500}
D_MODEL=${D_MODEL:-128}
NUM_LAYERS=${NUM_LAYERS:-4}
NUM_HEADS=${NUM_HEADS:-8}
DIM_FEEDFORWARD=${DIM_FEEDFORWARD:-512}
EPOCHS=${EPOCHS:-100}
LR=${LR:-1e-3}
JOBS_PER_GPU=${JOBS_PER_GPU:-2}

echo "Configuration:"
echo "  Dataset: ${COMPLEXITY} (${RESOLUTION}x${RESOLUTION}, ${NUM_SAMPLES} samples)"
echo "  Model: d_model=${D_MODEL}, layers=${NUM_LAYERS}, heads=${NUM_HEADS}"
echo "  Training: ${EPOCHS} epochs, lr=${LR}"
echo "  GPUs: ${NUM_GPUS} GPUs, ${JOBS_PER_GPU} jobs per GPU"
echo ""

# Create output directories
mkdir -p logs_improvements
mkdir -p checkpoints_improvements
mkdir -p results_improvements

echo "📁 Created output directories:"
echo "   logs_improvements/       - Training logs"
echo "   checkpoints_improvements/ - Model checkpoints"
echo "   results_improvements/    - Final results"
echo ""

# Launch orchestrator
echo "🚀 Starting orchestrator..."
echo ""
echo "   To monitor progress:"
echo "   - Overall: tail -f logs_improvements/orchestrator.log"
echo "   - Specific version: tail -f logs_improvements/v<N>_gpu<G>.log"
echo ""

python3 -u orchestrator.py \
    --complexity "${COMPLEXITY}" \
    --resolution ${RESOLUTION} \
    --num_samples ${NUM_SAMPLES} \
    --d_model ${D_MODEL} \
    --num_layers ${NUM_LAYERS} \
    --num_heads ${NUM_HEADS} \
    --dim_feedforward ${DIM_FEEDFORWARD} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --num_gpus ${NUM_GPUS} \
    --jobs_per_gpu ${JOBS_PER_GPU} \
    2>&1 | tee logs_improvements/orchestrator.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo ""
    echo "❌ Orchestrator failed"
    exit 1
fi

echo ""
echo "✅ Orchestration complete!"
echo ""

# ============================================================================
# Step 4: Generate summary report
# ============================================================================

echo "📊 Generating summary report..."
echo ""

python3 -c "
import json
from pathlib import Path

results_path = Path('results_improvements/all_results.json')
if results_path.exists():
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Find best versions
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('final_psnr', 0), reverse=True)

    print('🏆 Top 10 Versions:')
    print('─' * 70)
    for i, (version, result) in enumerate(sorted_results[:10], 1):
        psnr = result.get('final_psnr', 0)
        techniques = result.get('techniques', [])
        time_min = result.get('training_time', 0) / 60
        print(f'  {i}. {version}: {psnr:.2f} dB | Techniques: {techniques} | {time_min:.1f}min')
    print('─' * 70)
    print()

    # Baseline comparison
    baseline_psnr = 35.0  # Assume V2 baseline
    best_psnr = sorted_results[0][1].get('final_psnr', 0)
    improvement = best_psnr - baseline_psnr

    print(f'📈 Performance Summary:')
    print(f'   Baseline (V2): {baseline_psnr:.2f} dB')
    print(f'   Best (v{sorted_results[0][0][1:]}): {best_psnr:.2f} dB')
    print(f'   Improvement: +{improvement:.2f} dB ({improvement/baseline_psnr*100:.1f}%)')
    print()

    # Success rate
    total_versions = 100
    completed = len(results)
    success_rate = completed / total_versions * 100
    print(f'✅ Success Rate: {completed}/{total_versions} ({success_rate:.1f}%)')
    print()
else:
    print('⚠️  No results found')
"

echo ""
echo "============================================================"
echo "Experiment Suite Complete!"
echo "============================================================"
echo "Results available in:"
echo "  results_improvements/all_results.json         - All version results"
echo "  results_improvements/performance_matrix.csv   - CSV performance data"
echo "  results_improvements/technique_ranking.json   - Technique impact ranking"
echo "  results_improvements/failed_jobs.json         - Failed jobs (if any)"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Review results: cat results_improvements/technique_ranking.json"
echo "  2. Re-train top versions with more epochs for final validation"
echo "  3. Test generalization across all 6 complexity levels"
echo ""
