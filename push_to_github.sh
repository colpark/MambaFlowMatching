#!/bin/bash

# ============================================================================
# Push MambaFlowMatching to GitHub
# ============================================================================

echo "=========================================="
echo "MambaFlowMatching - GitHub Push Script"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d ".git" ]; then
    echo "❌ Error: Must be run from MambaFlowMatching directory"
    exit 1
fi

echo "✅ Repository ready to push"
echo ""
echo "Current status:"
git log --oneline -1
echo ""
git status --short
echo ""

# Prompt for GitHub username
read -p "Enter your GitHub username: " GITHUB_USER

if [ -z "$GITHUB_USER" ]; then
    echo "❌ Error: GitHub username required"
    exit 1
fi

# Set repository name (can be customized)
REPO_NAME="MambaFlowMatching"

echo ""
echo "=========================================="
echo "Repository Configuration:"
echo "  GitHub User: $GITHUB_USER"
echo "  Repository: $REPO_NAME"
echo "  URL: https://github.com/$GITHUB_USER/$REPO_NAME"
echo "=========================================="
echo ""

# Ask for confirmation
read -p "Have you created the repository '$REPO_NAME' on GitHub? (y/n): " CREATED

if [ "$CREATED" != "y" ] && [ "$CREATED" != "Y" ]; then
    echo ""
    echo "Please create the repository first:"
    echo "  1. Go to: https://github.com/new"
    echo "  2. Repository name: $REPO_NAME"
    echo "  3. Description: MAMBA state space models with flow matching for sparse neural field generation"
    echo "  4. Choose Public or Private"
    echo "  5. DO NOT initialize with README, .gitignore, or license"
    echo "  6. Click 'Create repository'"
    echo ""
    echo "Then run this script again."
    exit 0
fi

echo ""
echo "🔄 Adding GitHub remote..."

# Check if remote already exists
if git remote | grep -q "origin"; then
    echo "⚠️  Remote 'origin' already exists. Removing..."
    git remote remove origin
fi

# Add remote
git remote add origin "https://github.com/$GITHUB_USER/$REPO_NAME.git"

echo "✅ Remote added"
echo ""

# Verify remote
echo "Remote configuration:"
git remote -v
echo ""

# Push to GitHub
echo "🚀 Pushing to GitHub..."
echo ""

git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ SUCCESS! Repository pushed to GitHub"
    echo "=========================================="
    echo ""
    echo "View your repository at:"
    echo "  https://github.com/$GITHUB_USER/$REPO_NAME"
    echo ""
    echo "Next steps:"
    echo "  1. Add topics: mamba, flow-matching, neural-fields, sparse-learning"
    echo "  2. Review README.md on GitHub"
    echo "  3. Consider adding a LICENSE file"
    echo "  4. Start training models!"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ Push failed"
    echo "=========================================="
    echo ""
    echo "Common issues:"
    echo "  1. Repository doesn't exist on GitHub"
    echo "  2. Authentication failed (see below)"
    echo "  3. Wrong username or repository name"
    echo ""
    echo "Authentication options:"
    echo ""
    echo "A) Personal Access Token (Recommended):"
    echo "   1. Go to: https://github.com/settings/tokens"
    echo "   2. Generate new token (classic)"
    echo "   3. Select 'repo' scope"
    echo "   4. Copy token"
    echo "   5. Use token as password when prompted"
    echo ""
    echo "B) SSH (Alternative):"
    echo "   1. Generate SSH key: ssh-keygen -t ed25519 -C 'your_email@example.com'"
    echo "   2. Add to GitHub: https://github.com/settings/keys"
    echo "   3. Change remote: git remote set-url origin git@github.com:$GITHUB_USER/$REPO_NAME.git"
    echo "   4. Push again: git push -u origin main"
    echo ""
    exit 1
fi
