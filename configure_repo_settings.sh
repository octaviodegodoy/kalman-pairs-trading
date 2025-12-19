#!/bin/bash

# Configure repository settings via GitHub CLI

GITHUB_USER="octaviodegodoy"
REPO_NAME="kalman-pairs-trading"

echo "Configuring repository settings..."

if !  command -v gh &> /dev/null; then
    echo "GitHub CLI required. Install from: https://cli.github.com/"
    exit 1
fi

# Enable features
echo "Enabling repository features..."

gh repo edit $GITHUB_USER/$REPO_NAME \
    --enable-issues \
    --enable-projects \
    --enable-wiki=false \
    --enable-discussions

# Add topics
echo "Adding repository topics..."

gh repo edit $GITHUB_USER/$REPO_NAME \
    --add-topic pairs-trading \
    --add-topic kalman-filter \
    --add-topic algorithmic-trading \
    --add-topic quantitative-finance \
    --add-topic python \
    --add-topic machine-learning \
    --add-topic backtesting \
    --add-topic trading-strategies \
    --add-topic financial-analysis \
    --add-topic portfolio-optimization

echo "✅ Repository settings configured"

# Set up labels
echo "Creating issue labels..."

gh label create "bug" --color d73a4a --description "Something isn't working" --repo $GITHUB_USER/$REPO_NAME || true
gh label create "enhancement" --color a2eeef --description "New feature or request" --repo $GITHUB_USER/$REPO_NAME || true
gh label create "documentation" --color 0075ca --description "Improvements or additions to documentation" --repo $GITHUB_USER/$REPO_NAME || true
gh label create "good first issue" --color 7057ff --description "Good for newcomers" --repo $GITHUB_USER/$REPO_NAME || true
gh label create "help wanted" --color 008672 --description "Extra attention is needed" --repo $GITHUB_USER/$REPO_NAME || true
gh label create "question" --color d876e3 --description "Further information is requested" --repo $GITHUB_USER/$REPO_NAME || true
gh label create "wontfix" --color ffffff --description "This will not be worked on" --repo $GITHUB_USER/$REPO_NAME || true
gh label create "dependencies" --color 0366d6 --description "Pull requests that update a dependency file" --repo $GITHUB_USER/$REPO_NAME || true
gh label create "ci" --color 00ff00 --description "Continuous integration" --repo $GITHUB_USER/$REPO_NAME || true
gh label create "performance" --color ff9800 --description "Performance improvements" --repo $GITHUB_USER/$REPO_NAME || true

echo "✅ Labels created"

echo ""
echo "Repository fully configured!"