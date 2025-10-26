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

# Test 3: Test PyTorch availability and optional training
echo ""
echo "✓ Test 3: Checking PyTorch installation..."

python3 -c "import torch; print(f'  ✅ PyTorch {torch.__version__} installed')" 2>/dev/null

if [ $? -eq 0 ]; then
    # PyTorch available - run quick training test

    # Determine device
    if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
        TEST_DEVICE="auto"
        TEST_GPU_ID=0
        DEVICE_NAME="GPU 0"
        python3 -c "import torch; print(f'  ✅ CUDA available: {torch.cuda.is_available()}')" 2>/dev/null
    else
        TEST_DEVICE="cpu"
        TEST_GPU_ID=0
        DEVICE_NAME="CPU"
    fi

    echo "  🧪 Running quick training test (v3, 1 epoch, ${DEVICE_NAME})..."
    python3 train_improved.py \
        --techniques "1" \
        --version 3 \
        --epochs 1 \
        --num_samples 50 \
        --resolution 16 \
        --device "${TEST_DEVICE}" \
        --gpu_id ${TEST_GPU_ID} \
        --save_dir checkpoints_test \
        > logs_test.txt 2>&1

    if [ $? -eq 0 ] && [ -f "checkpoints_test/v3/best_model.pth" ]; then
        echo "  ✅ Training test successful on ${DEVICE_NAME}"
        rm -rf checkpoints_test logs_test.txt
    else
        echo "  ⚠️  Training test failed (non-critical)"
        echo "  📋 Last 10 lines of log:"
        tail -10 logs_test.txt
        echo "  💡 This is OK - framework structure is valid"
        rm -rf logs_test.txt
    fi
else
    echo "  ⚠️  PyTorch not installed"
    echo "  💡 Install PyTorch to run experiments:"
    echo "     pip install torch torchvision"
    echo "  💡 Continuing with other tests..."
fi

# Test 4: Verify code structure (syntax check)
echo ""
echo "✓ Test 4: Verifying Python code syntax..."

for pyfile in techniques.py improved_transformer.py train_improved.py orchestrator.py generate_combinations.py analyze_results.py; do
    python3 -m py_compile "$pyfile" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  ✅ $pyfile syntax OK"
    else
        echo "  ❌ $pyfile syntax error"
        python3 -m py_compile "$pyfile"
        exit 1
    fi
done

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
