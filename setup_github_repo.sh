#!/bin/bash

# GitHub Repository Setup Script
# This script will guide you through setting up your repository

set -e

echo "=========================================="
echo "GitHub Repository Setup"
echo "=========================================="
echo ""

# Get GitHub username
read -p "Enter your GitHub username [octaviodegodoy]: " GITHUB_USER
GITHUB_USER=${GITHUB_USER:-octaviodegodoy}

echo ""
echo "✅ Using GitHub username: $GITHUB_USER"
echo ""

# Repository details
REPO_NAME="kalman-pairs-trading"
REPO_URL="https://github.com/$GITHUB_USER/$REPO_NAME.git"

echo "Repository:  $REPO_NAME"
echo "URL: $REPO_URL"
echo ""

# Check if gh CLI is installed
if command -v gh &> /dev/null; then
    echo "✅ GitHub CLI detected"
    echo ""
    read -p "Create repository using GitHub CLI? (y/n) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Creating repository..."
        gh repo create $REPO_NAME \
            --public \
            --description "Production-ready Kalman Filter pairs trading system with dynamic hedge ratio estimation" \
            --homepage "https://$GITHUB_USER.github. io/$REPO_NAME" \
            --enable-issues \
            --enable-wiki=false
        
        echo "✅ Repository created!"
    fi
else
    echo "⚠️  GitHub CLI not found"
    echo ""
    echo "Please create the repository manually:"
    echo "1. Go to: https://github.com/new"
    echo "2. Repository name: $REPO_NAME"
    echo "3. Description: Production-ready Kalman Filter pairs trading system"
    echo "4. Visibility: Public"
    echo "5. Do NOT initialize with README (we have one)"
    echo "6. Click 'Create repository'"
    echo ""
    read -p "Press Enter when repository is created..."
fi

# Initialize git if needed
if [ ! -d ". git" ]; then
    echo ""
    echo "Initializing git repository..."
    git init
    echo "✅ Git initialized"
fi

# Configure git
echo ""
echo "Configuring git..."
git config user.name "$GITHUB_USER"
read -p "Enter your email: " GIT_EMAIL
git config user. email "$GIT_EMAIL"
echo "✅ Git configured"

# Add all files
echo ""
echo "Adding files..."
git add .
echo "✅ Files added"

# Create initial commit
echo ""
echo "Creating initial commit..."
git commit -m "Initial commit: Complete Kalman Filter pairs trading system

Features:
- Kalman Filter for dynamic hedge ratio estimation
- Comprehensive backtesting engine
- Multi-pair portfolio optimization
- Automated pair selection
- Parameter optimization (Grid, Random, Bayesian)
- Machine learning integration (LSTM, HMM)
- Real-time trading with broker integration
- Interactive Streamlit dashboard
- Complete CI/CD pipeline
- Comprehensive test suite
- Docker deployment ready

Tech Stack:
- Python 3.8+
- NumPy, Pandas, SciPy
- Scikit-learn, TensorFlow
- Streamlit, Plotly
- GitHub Actions
- Docker
" || echo "Already committed"

echo "✅ Initial commit created"

# Add remote
echo ""
echo "Adding remote..."
if git remote | grep -q origin; then
    echo "Remote 'origin' already exists, removing..."
    git remote remove origin
fi

git remote add origin "$REPO_URL"
echo "✅ Remote added:  $REPO_URL"

# Create and push main branch
echo ""
echo "Setting up main branch..."
git branch -M main
echo "✅ Main branch configured"

echo ""
echo "Pushing to GitHub..."
git push -u origin main
echo "✅ Pushed to main branch"

# Create develop branch
echo ""
echo "Creating develop branch..."
git checkout -b develop
git push -u origin develop
echo "✅ Develop branch created and pushed"

# Go back to main
git checkout main

echo ""
echo "=========================================="
echo "✅ Repository Setup Complete!"
echo "=========================================="
echo ""
echo "📝 Repository URL: https://github.com/$GITHUB_USER/$REPO_NAME"
echo ""

# Configure branch protection
if command -v gh &> /dev/null; then
    read -p "Configure branch protection rules? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Setting up branch protection..."
        
        # Note: This requires admin access
        gh api repos/$GITHUB_USER/$REPO_NAME/branches/main/protection \
            --method PUT \
            --field required_status_checks='{"strict": true,"contexts":["test"]}' \
            --field enforce_admins=true \
            --field required_pull_request_reviews='{"required_approving_review_count": 1}' \
            --field restrictions=null \
            2>/dev/null && echo "✅ Branch protection enabled" || echo "⚠️  Branch protection requires admin access.  Set up manually."
    fi
fi

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. 🔐 Set up GitHub Secrets:"
echo "   Go to: https://github.com/$GITHUB_USER/$REPO_NAME/settings/secrets/actions"
echo "   Add these secrets:"
echo "   - DOCKER_USERNAME"
echo "   - DOCKER_PASSWORD"
echo "   - PYPI_API_TOKEN (optional)"
echo "   - ALPACA_API_KEY (optional)"
echo "   - ALPACA_API_SECRET (optional)"
echo ""
echo "2. 🔄 Enable GitHub Actions:"
echo "   Go to: https://github.com/$GITHUB_USER/$REPO_NAME/actions"
echo "   (Should be enabled automatically)"
echo ""
echo "3. 📊 Set up GitHub Pages:"
echo "   Go to:  https://github.com/$GITHUB_USER/$REPO_NAME/settings/pages"
echo "   Source: Deploy from a branch"
echo "   Branch: gh-pages / root"
echo ""
echo "4. 🏷️ Add topics to your repo:"
echo "   Go to: https://github.com/$GITHUB_USER/$REPO_NAME"
echo "   Add:  pairs-trading, kalman-filter, algorithmic-trading,"
echo "        quantitative-finance, python, machine-learning"
echo ""
echo "5. ⭐ Make your first release:"
echo "   git tag -a v1.0.0 -m 'Release v1.0.0'"
echo "   git push origin v1.0.0"
echo ""

# Ask about secrets setup
read -p "Do you want to set up GitHub secrets now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Setting up secrets..."
    
    if command -v gh &> /dev/null; then
        echo ""
        read -p "Enter Docker Hub username (or skip): " DOCKER_USER
        if [ !  -z "$DOCKER_USER" ]; then
            read -sp "Enter Docker Hub password: " DOCKER_PASS
            echo ""
            echo "$DOCKER_PASS" | gh secret set DOCKER_PASSWORD --repo=$GITHUB_USER/$REPO_NAME
            echo "$DOCKER_USER" | gh secret set DOCKER_USERNAME --repo=$GITHUB_USER/$REPO_NAME
            echo "✅ Docker secrets added"
        fi
        
        echo ""
        read -p "Enter PyPI API token (or skip): " PYPI_TOKEN
        if [ !  -z "$PYPI_TOKEN" ]; then
            echo "$PYPI_TOKEN" | gh secret set PYPI_API_TOKEN --repo=$GITHUB_USER/$REPO_NAME
            echo "✅ PyPI token added"
        fi
        
        echo ""
        read -p "Enter Alpaca API key (or skip): " ALPACA_KEY
        if [ ! -z "$ALPACA_KEY" ]; then
            echo "$ALPACA_KEY" | gh secret set ALPACA_API_KEY --repo=$GITHUB_USER/$REPO_NAME
            read -sp "Enter Alpaca API secret: " ALPACA_SECRET
            echo ""
            echo "$ALPACA_SECRET" | gh secret set ALPACA_API_SECRET --repo=$GITHUB_USER/$REPO_NAME
            echo "✅ Alpaca credentials added"
        fi
    else
        echo "GitHub CLI not available.  Please add secrets manually at:"
        echo "https://github.com/$GITHUB_USER/$REPO_NAME/settings/secrets/actions"
    fi
fi

echo ""
echo "=========================================="
echo "🎉 Setup Complete!"
echo "=========================================="
echo ""
echo "Your repository is live at:"
echo "🔗 https://github.com/$GITHUB_USER/$REPO_NAME"
echo ""
echo "GitHub Actions will run automatically on your next push!"
echo ""