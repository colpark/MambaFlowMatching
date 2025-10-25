# GitHub Setup Instructions

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `MambaFlowMatching` (or your preferred name)
3. Description: `MAMBA state space models with flow matching for sparse neural field generation`
4. Choose: **Public** or **Private**
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## Step 2: Connect Local Repository to GitHub

After creating the repository on GitHub, you'll see a page with instructions. Use these commands:

```bash
cd /Users/davidpark/Documents/Claude/MambaFlowMatching

# Add remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/MambaFlowMatching.git

# Verify remote was added
git remote -v

# Push to GitHub
git push -u origin main
```

## Step 3: Verify Upload

Go to your GitHub repository URL:
`https://github.com/YOUR_USERNAME/MambaFlowMatching`

You should see:
- ✅ README.md displayed on main page
- ✅ All directories: core/, v1/, v2/, docs/, scripts/
- ✅ 37 files committed
- ✅ Initial commit message visible

## Alternative: Use GitHub CLI

If you have GitHub CLI installed (`gh`):

```bash
cd /Users/davidpark/Documents/Claude/MambaFlowMatching

# Create repository and push in one command
gh repo create MambaFlowMatching --public --source=. --remote=origin --push

# Or for private repository
gh repo create MambaFlowMatching --private --source=. --remote=origin --push
```

## Repository Information

- **Local path**: `/Users/davidpark/Documents/Claude/MambaFlowMatching`
- **Branch**: `main`
- **Commit**: `3059876` - "Initial commit: MAMBA Flow Matching for Sparse Neural Fields"
- **Files**: 37 files, 7528 lines of code
- **Structure**: Organized with V1, V2, core modules, docs, and scripts

## What's Included

### Core Modules
- `core/neural_fields/` - Fourier features, perceiver components
- `core/sparse/` - Sparse CIFAR-10 dataset, metrics tracking
- `core/diffusion/` - Flow matching utilities

### V1 Architecture
- `v1/training/` - Baseline MAMBA training (6 unidirectional layers)
- `v1/evaluation/` - Super-resolution and SDE sampling evaluation

### V2 Architecture
- `v2/training/` - Improved bidirectional MAMBA (4+4 layers)
- `v2/evaluation/` - V1 vs V2 comparison tools

### Documentation
- Main README.md with quick start guide
- docs/ directory with detailed guides for V2, super-resolution, SDE sampling

### Utilities
- requirements.txt with all dependencies
- .gitignore configured for ML projects
- Remote setup scripts and verification tools

## Next Steps After Push

1. Add topics/tags on GitHub: `mamba`, `flow-matching`, `neural-fields`, `sparse-learning`, `deep-learning`
2. Consider adding a LICENSE file (MIT recommended)
3. Set up GitHub Actions for CI/CD (optional)
4. Add project wiki for detailed documentation (optional)

## Troubleshooting

### Authentication Issues
If you get authentication errors:

**Using HTTPS:**
```bash
# Set up credential helper
git config --global credential.helper store
```

**Using SSH:**
```bash
# Generate SSH key if you don't have one
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to GitHub: Settings → SSH and GPG keys → New SSH key
cat ~/.ssh/id_ed25519.pub

# Change remote to SSH
git remote set-url origin git@github.com:YOUR_USERNAME/MambaFlowMatching.git
```

### Push Fails
```bash
# Check remote
git remote -v

# Try force push (only for initial push)
git push -u origin main --force
```
