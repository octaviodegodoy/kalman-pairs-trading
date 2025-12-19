#!/bin/bash

# Kalman Pairs Trading - Deployment Script

set -e

echo "=================================================="
echo "Kalman Pairs Trading - Automated Deployment"
echo "=================================================="

# Check prerequisites
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required but not installed.  Aborting." >&2; exit 1; }
command -v git >/dev/null 2>&1 || { echo "Git is required but not installed.  Aborting." >&2; exit 1; }

# Step 1: Create GitHub repository
echo ""
echo "Step 1: GitHub Repository Setup"
echo "================================"
echo "Please create a GitHub repository named 'kalman-pairs-trading' at:"
echo "https://github.com/new"
echo ""
read -p "Have you created the repository? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

# Step 2: Initialize git
echo ""
echo "Step 2:  Initializing Git Repository"
echo "===================================="

if [ !  -d ". git" ]; then
    git init
    echo "Git repository initialized"
else
    echo "Git repository already exists"
fi

# Step 3: Add files
echo ""
echo "Step 3: Adding Files"
echo "==================="
git add .
git commit -m "Initial commit: Complete Kalman pairs trading system" || echo "No changes to commit"

# Step 4: Add remote
echo ""
echo "Step 4: Connecting to GitHub"
echo "============================="
read -p "Enter your GitHub username:  " github_user

if !  git remote | grep -q origin; then
    git remote add origin "https://github.com/$github_user/kalman-pairs-trading.git"
    echo "Remote 'origin' added"
else
    echo "Remote 'origin' already exists"
fi

# Step 5: Create main branch
echo ""
echo "Step 5: Setting Up Branches"
echo "==========================="
git branch -M main
echo "Main branch created"

# Step 6: Push to GitHub
echo ""
echo "Step 6: Pushing to GitHub"
echo "========================="
git push -u origin main

# Step 7: Create develop branch
echo ""
echo "Step 7: Creating Develop Branch"
echo "==============================="
git checkout -b develop
git push -u origin develop
git checkout main

# Step 8: Set up GitHub Secrets
echo ""
echo "Step 8: GitHub Secrets Setup"
echo "============================"
echo "Please add the following secrets to your GitHub repository:"
echo ""
echo "1. Go to:  https://github.com/$github_user/kalman-pairs-trading/settings/secrets/actions"
echo "2. Add these secrets:"
echo "   - DOCKER_USERNAME (your Docker Hub username)"
echo "   - DOCKER_PASSWORD (your Docker Hub password)"
echo "   - PYPI_API_TOKEN (optional, for PyPI publishing)"
echo "   - ALPACA_API_KEY (for live trading)"
echo "   - ALPACA_API_SECRET (for live trading)"
echo ""
read -p "Press Enter when secrets are configured..."

# Step 9: Enable GitHub Actions
echo ""
echo "Step 9: GitHub Actions"
echo "====================="
echo "GitHub Actions are automatically enabled."
echo "View workflow runs at: https://github.com/$github_user/kalman-pairs-trading/actions"

# Step 10: Docker setup
echo ""
echo "Step 10: Docker Setup (Optional)"
echo "================================"
read -p "Do you want to build Docker image now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    if command -v docker >/dev/null 2>&1; then
        docker build -t $github_user/kalman-pairs-trading:latest .
        echo "Docker image built successfully"
        
        read -p "Push to Docker Hub? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]
        then
            docker login
            docker push $github_user/kalman-pairs-trading:latest
            echo "Docker image pushed to Docker Hub"
        fi
    else
        echo "Docker not installed. Skipping Docker build."
    fi
fi

# Step 11: Install locally
echo ""
echo "Step 11: Local Installation"
echo "==========================="
read -p "Install package locally? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python3 -m pip install -e ".[dev]"
    echo "Package installed in development mode"
fi

# Step 12: Run tests
echo ""
echo "Step 12: Running Tests"
echo "======================"
read -p "Run tests? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    pytest tests/ -v
fi

# Step 13: Launch dashboard
echo ""
echo "Step 13: Dashboard"
echo "=================="
read -p "Launch dashboard? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Starting dashboard at http://localhost:8501"
    streamlit run dashboard/app.py
fi

echo ""
echo "=================================================="
echo "Deployment Complete!  🎉"
echo "=================================================="
echo ""
echo "📝 Next Steps:"
echo "  1. View your repository:  https://github.com/$github_user/kalman-pairs-trading"
echo "  2. Check Actions:  https://github.com/$github_user/kalman-pairs-trading/actions"
echo "  3. Deploy dashboard: https://share.streamlit.io/"
echo "  4. Star the repo!  ⭐"
echo ""
echo "📚 Documentation:"
echo "  - README.md - Project overview"
echo "  - DEPLOYMENT.md - Detailed deployment guide"
echo "  - CONTRIBUTING.md - Contribution guidelines"
echo ""
echo "🚀 Quick Commands:"
echo "  make test       - Run tests"
echo "  make run        - Start dashboard"
echo "  make docker     - Build Docker image"
echo ""
echo "Happy Trading! 📈"