#!/bin/bash

# Complete GitHub Repository Setup
# This script runs all setup steps automatically

set -e

echo "=========================================="
echo "🚀 Complete GitHub Setup"
echo "=========================================="
echo ""

# Make scripts executable
chmod +x setup_github_repo.sh
chmod +x configure_repo_settings.sh
chmod +x create_initial_issues. sh
chmod +x deploy. sh

# Run setup
echo "Step 1: Setting up repository..."
./setup_github_repo.sh

echo ""
echo "Step 2: Configuring repository settings..."
./configure_repo_settings.sh

echo ""
echo "Step 3: Creating initial issues..."
./create_initial_issues.sh

echo ""
echo "=========================================="
echo "✅ Complete Setup Finished!"
echo "=========================================="
echo ""
echo "📊 Your Repository:"
echo "   https://github.com/octaviodegodoy/kalman-pairs-trading"
echo ""
echo "🔄 Actions:"
echo "   https://github.com/octaviodegodoy/kalman-pairs-trading/actions"
echo ""
echo "📝 Issues:"
echo "   https://github.com/octaviodegodoy/kalman-pairs-trading/issues"
echo ""
echo "📚 Documentation (after first workflow):"
echo "   https://octaviodegodoy.github.io/kalman-pairs-trading"
echo ""
echo "🐳 Docker Hub (after pushing image):"
echo "   https://hub.docker.com/r/octaviodegodoy/kalman-pairs-trading"
echo ""
echo "🎯 Next Actions:"
echo "   1. Wait for GitHub Actions to complete"
echo "   2. Check that all workflows passed"
echo "   3. Set up Streamlit Cloud deployment"
echo "   4. Share your project!"
echo ""
echo "Happy Trading! 📈"