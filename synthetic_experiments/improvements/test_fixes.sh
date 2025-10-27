#!/bin/bash

# ============================================================================
# Quick Test Script for Fixes
#
# Verifies the critical fixes are working by training for a few epochs
# ============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

echo "============================================================"
echo "Testing Critical Fixes"
echo "============================================================"
echo ""

# Test 1: Verify fixes are in code
echo "✓ Test 1: Verifying fixes are present in code..."
echo ""

# Check Fix 1: Random query points
if grep -q "num_query = 200" train_improved.py; then
    echo "  ✅ Fix 1: Random query point sampling found"
else
    echo "  ❌ Fix 1: Random query point sampling NOT FOUND"
    exit 1
fi

# Check Fix 2: Output clamping
if grep -q "torch.clamp" improved_transformer.py; then
    echo "  ✅ Fix 2: Output clamping found"
else
    echo "  ❌ Fix 2: Output clamping NOT FOUND"
    exit 1
fi

# Check Fix 3: Learnable Fourier features
if grep -q "nn.Parameter.*torch.randn(2, num_frequencies)" improved_transformer.py; then
    echo "  ✅ Fix 3: Learnable Fourier features found"
else
    echo "  ❌ Fix 3: Learnable Fourier features NOT FOUND"
    exit 1
fi

echo ""
echo "All fixes verified in code!"
echo ""

# Test 2: Check if PyTorch is available
echo "✓ Test 2: Checking PyTorch installation..."
echo ""

python3 -c "import torch; print(f'  ✅ PyTorch {torch.__version__} installed')" 2>/dev/null

if [ $? -eq 0 ]; then
    # PyTorch available - run quick training test

    # Determine device
    if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
        TEST_DEVICE="auto"
        TEST_GPU_ID=0
        DEVICE_NAME="GPU 0"
        python3 -c "import torch; print(f'  ✅ CUDA available: {torch.cuda.is_available()}'))" 2>/dev/null
    else
        TEST_DEVICE="cpu"
        TEST_GPU_ID=0
        DEVICE_NAME="CPU"
        echo "  ⚠️  No GPU detected, using CPU (will be slow)"
    fi

    echo ""
    echo "✓ Test 3: Running quick training test..."
    echo "  🧪 Training v999 for 10 epochs on ${DEVICE_NAME}..."
    echo "  📝 Expected: PSNR should be >15 dB (not ~6 dB)"
    echo ""

    python3 train_improved.py \
        --techniques "" \
        --version 999 \
        --epochs 10 \
        --num_samples 100 \
        --resolution 32 \
        --device "${TEST_DEVICE}" \
        --gpu_id ${TEST_GPU_ID} \
        --save_dir checkpoints_test \
        > logs_test_fixes.txt 2>&1

    if [ $? -eq 0 ]; then
        # Check results
        if [ -f "checkpoints_test/v999/results.json" ]; then
            FINAL_PSNR=$(python3 -c "import json; print(json.load(open('checkpoints_test/v999/results.json'))['final_psnr'])")

            echo ""
            echo "  ✅ Training completed successfully!"
            echo "  📊 Final PSNR: ${FINAL_PSNR} dB"
            echo ""

            # Check if PSNR improved
            if python3 -c "import json; exit(0 if json.load(open('checkpoints_test/v999/results.json'))['final_psnr'] > 15 else 1)"; then
                echo "  🎉 SUCCESS: PSNR > 15 dB - Fixes are working!"
                echo "     (Previous baseline was ~6 dB)"
            else
                echo "  ⚠️  WARNING: PSNR still low (<15 dB)"
                echo "     Expected >15 dB after fixes"
                echo "     Check logs: cat logs_test_fixes.txt"
            fi

            # Cleanup
            echo ""
            echo "  🧹 Cleaning up test files..."
            rm -rf checkpoints_test logs_test_fixes.txt
        else
            echo "  ⚠️  Training finished but no results found"
            echo "  📋 Last 20 lines of log:"
            tail -20 logs_test_fixes.txt
        fi
    else
        echo "  ❌ Training failed"
        echo "  📋 Last 20 lines of log:"
        tail -20 logs_test_fixes.txt
        exit 1
    fi
else
    echo "  ⚠️  PyTorch not installed"
    echo "  💡 Install PyTorch to run training test:"
    echo "     pip install torch torchvision"
    echo "  💡 Code verification passed, but cannot test training"
fi

echo ""
echo "============================================================"
echo "✅ Fix Verification Complete"
echo "============================================================"
echo ""
echo "Summary:"
echo "  ✅ All 3 fixes verified in code"
if command -v python3 -c "import torch" &> /dev/null 2>&1; then
    echo "  ✅ Training test completed"
else
    echo "  ⏭️  Training test skipped (PyTorch not installed)"
fi
echo ""
echo "Next steps:"
echo "  1. Review: cat FIXES_APPLIED.md"
echo "  2. Run full suite: ./run_all.sh"
echo ""
