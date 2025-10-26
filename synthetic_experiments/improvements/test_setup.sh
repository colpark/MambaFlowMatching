#!/bin/bash

# ============================================================================
# Quick Test Script
#
# Verifies the improvement experiment framework is properly set up
# ============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

echo "============================================================"
echo "Testing Improvement Experiment Framework"
echo "============================================================"
echo ""

# Test 1: Generate combinations
echo "✓ Test 1: Generating combinations..."
python3 generate_combinations.py > /dev/null 2>&1
if [ $? -eq 0 ] && [ -f "combinations.json" ]; then
    echo "  ✅ combinations.json generated"
else
    echo "  ❌ Failed to generate combinations"
    exit 1
fi

# Test 2: Verify combinations
echo ""
echo "✓ Test 2: Verifying combinations..."
NUM_COMBOS=$(python3 -c "import json; print(len(json.load(open('combinations.json'))))")
if [ "$NUM_COMBOS" -eq 100 ]; then
    echo "  ✅ 100 combinations found"
else
    echo "  ❌ Expected 100 combinations, found $NUM_COMBOS"
    exit 1
fi

# Test 3: Test single version training (1 epoch, CPU)
echo ""
echo "✓ Test 3: Testing single version training (v3, 1 epoch, CPU)..."
python3 train_improved.py \
    --techniques "1" \
    --version 3 \
    --epochs 1 \
    --num_samples 50 \
    --resolution 16 \
    --device cpu \
    --save_dir checkpoints_test \
    > /dev/null 2>&1

if [ $? -eq 0 ] && [ -f "checkpoints_test/v3/best_model.pth" ]; then
    echo "  ✅ Training successful"
    rm -rf checkpoints_test
else
    echo "  ❌ Training failed"
    exit 1
fi

# Test 4: Verify imports
echo ""
echo "✓ Test 4: Verifying Python imports..."
python3 -c "
from synthetic_experiments.improvements.techniques import TECHNIQUES
from synthetic_experiments.improvements.improved_transformer import build_model_from_techniques
from synthetic_experiments.improvements.train_improved import conditional_flow
print('  ✅ All imports successful')
" 2>&1

if [ $? -ne 0 ]; then
    echo "  ❌ Import errors detected"
    exit 1
fi

# Test 5: Check required directories
echo ""
echo "✓ Test 5: Checking directory structure..."
for dir in "." "../datasets" "../evaluation"; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir exists"
    else
        echo "  ❌ $dir missing"
        exit 1
    fi
done

# Test 6: Verify scripts are executable
echo ""
echo "✓ Test 6: Verifying script permissions..."
for script in run_all.sh train_improved.py orchestrator.py generate_combinations.py analyze_results.py; do
    if [ -x "$script" ]; then
        echo "  ✅ $script is executable"
    else
        echo "  ⚠️  $script not executable (fixing...)"
        chmod +x "$script"
    fi
done

# Summary
echo ""
echo "============================================================"
echo "✅ All Tests Passed!"
echo "============================================================"
echo ""
echo "Framework is ready for use. You can now run:"
echo "  ./run_all.sh                  # Run all 100 experiments"
echo "  python3 train_improved.py ... # Run single experiment"
echo ""
echo "GPU Check:"
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "  ✅ Found ${NUM_GPUS} GPU(s)"
    nvidia-smi --list-gpus
else
    echo "  ⚠️  No GPUs detected (CPU mode only)"
fi
echo ""
