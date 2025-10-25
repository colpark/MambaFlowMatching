#!/bin/bash
# ============================================================================
# Remote Server Setup Script for SDE Sampling
#
# This script safely pulls the latest SDE sampling code on a remote server
# ============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="${PROJECT_DIR:-${HOME}/NFDiffusion}"
BRANCH="${BRANCH:-v2_local_implicit_unet}"
REPO_URL="https://github.com/colpark/NFDiffusion.git"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}NFDiffusion SDE Sampling - Remote Setup${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "Configuration:"
echo "  Project directory: ${PROJECT_DIR}"
echo "  Branch: ${BRANCH}"
echo "  Repository: ${REPO_URL}"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Error: git is not installed${NC}"
    echo "Install git first: sudo apt-get install git"
    exit 1
fi

# Check if directory exists
if [ -d "$PROJECT_DIR" ]; then
    echo -e "${GREEN}📁 Project directory exists, updating...${NC}"
    cd "$PROJECT_DIR"

    # Check current branch
    CURRENT_BRANCH=$(git branch --show-current)
    echo "Current branch: ${CURRENT_BRANCH}"

    # Check for uncommitted changes
    if ! git diff-index --quiet HEAD -- 2>/dev/null; then
        echo -e "${YELLOW}⚠️  Uncommitted changes detected${NC}"
        echo "Options:"
        echo "  1) Stash changes and update"
        echo "  2) Discard changes and update"
        echo "  3) Cancel"
        read -p "Choose [1-3]: " choice

        case $choice in
            1)
                echo "Stashing changes..."
                git stash save "backup-$(date +%Y%m%d-%H%M%S)"
                STASHED=true
                ;;
            2)
                echo "Discarding changes..."
                git reset --hard HEAD
                ;;
            3)
                echo "Cancelled"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid choice${NC}"
                exit 1
                ;;
        esac
    fi

    # Fetch and pull
    echo ""
    echo -e "${BLUE}📥 Fetching latest changes...${NC}"
    git fetch origin

    # Show what will be pulled
    NEW_COMMITS=$(git rev-list HEAD..origin/$BRANCH --count 2>/dev/null || echo "0")
    if [ "$NEW_COMMITS" -gt 0 ]; then
        echo -e "${GREEN}Found ${NEW_COMMITS} new commits:${NC}"
        git log HEAD..origin/$BRANCH --oneline --graph --decorate --color
        echo ""
    else
        echo -e "${GREEN}Already up to date${NC}"
    fi

    echo -e "${BLUE}🔄 Pulling branch: ${BRANCH}${NC}"
    git pull origin "$BRANCH"

    # Restore stashed changes if any
    if [ "$STASHED" = true ]; then
        echo ""
        echo -e "${BLUE}♻️  Restoring stashed changes...${NC}"
        if git stash pop; then
            echo -e "${GREEN}✓ Changes restored${NC}"
        else
            echo -e "${YELLOW}⚠️  Conflicts detected, resolve manually${NC}"
        fi
    fi

else
    echo -e "${YELLOW}📦 Project directory not found, cloning...${NC}"

    # Create parent directory if needed
    mkdir -p "$(dirname "$PROJECT_DIR")"

    # Clone repository
    git clone "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"

    # Checkout correct branch
    git checkout "$BRANCH"
    echo -e "${GREEN}✓ Repository cloned${NC}"
fi

# Verify we're on the correct branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
    echo -e "${YELLOW}⚠️  Not on ${BRANCH}, switching...${NC}"
    git checkout "$BRANCH"
fi

# Verify new files exist
echo ""
echo -e "${BLUE}✅ Verifying SDE files...${NC}"

REQUIRED_FILES=(
    "ASF/eval_sde.sh"
    "ASF/eval_sde_multiscale.py"
    "ASF/train_mamba_standalone.py"
    "ASF/README_SDE.md"
)

ALL_FOUND=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${RED}✗${NC} $file ${RED}(missing)${NC}"
        ALL_FOUND=false
    fi
done

if [ "$ALL_FOUND" = false ]; then
    echo -e "${RED}❌ Error: Some required files are missing${NC}"
    echo "Try: git pull origin $BRANCH --force"
    exit 1
fi

# Make scripts executable
echo ""
echo -e "${BLUE}🔧 Setting file permissions...${NC}"
chmod +x ASF/eval_sde.sh
chmod +x ASF/eval_superres.sh 2>/dev/null || true
chmod +x ASF/run_mamba_training.sh 2>/dev/null || true
echo -e "${GREEN}✓ Scripts are executable${NC}"

# Check Python environment
echo ""
echo -e "${BLUE}🐍 Checking Python environment...${NC}"

PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_CMD="python"
    if ! command -v $PYTHON_CMD &> /dev/null; then
        echo -e "${RED}❌ Python not found${NC}"
        exit 1
    fi
fi

echo "Python: $($PYTHON_CMD --version)"

# Check dependencies
echo "Checking dependencies..."
DEPS_OK=true

for dep in torch numpy matplotlib sklearn; do
    if $PYTHON_CMD -c "import $dep" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $dep"
    else
        echo -e "${RED}✗${NC} $dep ${YELLOW}(missing)${NC}"
        DEPS_OK=false
    fi
done

if [ "$DEPS_OK" = false ]; then
    echo ""
    echo -e "${YELLOW}⚠️  Missing dependencies detected${NC}"
    echo "Install with:"
    echo "  pip install torch numpy matplotlib scikit-image"
    echo "Or:"
    echo "  pip install -r requirements.txt"
fi

# Test imports
echo ""
echo -e "${BLUE}🧪 Testing SDE imports...${NC}"
cd ASF
if $PYTHON_CMD -c "from train_mamba_standalone import sde_sample, ddim_sample; print('OK')" &>/dev/null; then
    echo -e "${GREEN}✓ SDE sampler imports successfully${NC}"
    echo -e "${GREEN}✓ DDIM sampler imports successfully${NC}"
else
    echo -e "${RED}✗ Import test failed${NC}"
    echo "Debug with: cd ASF && python3 -c 'from train_mamba_standalone import sde_sample'"
fi

# Check for trained checkpoint
echo ""
echo -e "${BLUE}📦 Checking for trained checkpoints...${NC}"
if [ -d "checkpoints_mamba" ] && [ "$(ls -A checkpoints_mamba/*.pth 2>/dev/null)" ]; then
    NUM_CHECKPOINTS=$(ls checkpoints_mamba/*.pth 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ Found ${NUM_CHECKPOINTS} checkpoint(s)${NC}"
    ls -lh checkpoints_mamba/*.pth | awk '{print "  - " $9 " (" $5 ")"}'
    READY_TO_EVAL=true
else
    echo -e "${YELLOW}⚠️  No checkpoints found${NC}"
    echo "Train a model first:"
    echo "  cd ASF"
    echo "  ./run_mamba_training.sh"
    READY_TO_EVAL=false
fi

# Display summary
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Project: ${PROJECT_DIR}"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git log -1 --oneline)"
echo ""
echo -e "${BLUE}📊 New SDE Sampling Features:${NC}"
echo "  ✓ SDE sampler (stochastic, reduces speckles)"
echo "  ✓ DDIM sampler (fast, non-uniform timesteps)"
echo "  ✓ Multi-scale evaluation (32x32, 64x64, 96x96)"
echo "  ✓ Sampler comparison (Heun vs SDE vs DDIM)"
echo ""

if [ "$READY_TO_EVAL" = true ]; then
    echo -e "${GREEN}🚀 Ready to run evaluation:${NC}"
    echo "  cd ${PROJECT_DIR}/ASF"
    echo "  ./eval_sde.sh"
    echo ""
    echo -e "${BLUE}Quick start:${NC}"
    echo "  cat QUICKSTART_SDE.md        # Quick reference"
    echo "  cat README_SDE.md            # Full documentation"
    echo ""
    echo -e "${BLUE}Example commands:${NC}"
    echo "  ./eval_sde.sh                              # Compare all samplers"
    echo "  SAMPLERS=\"sde\" ./eval_sde.sh              # SDE only"
    echo "  TEMPERATURE=0.7 ./eval_sde.sh              # Smoother results"
    echo "  NUM_SAMPLES=50 NUM_STEPS=100 ./eval_sde.sh  # High quality"
else
    echo -e "${YELLOW}📝 Next steps:${NC}"
    echo "  1. Train a model:"
    echo "     cd ${PROJECT_DIR}/ASF"
    echo "     ./run_mamba_training.sh"
    echo ""
    echo "  2. Then run SDE evaluation:"
    echo "     ./eval_sde.sh"
fi

echo -e "${GREEN}============================================================${NC}"
