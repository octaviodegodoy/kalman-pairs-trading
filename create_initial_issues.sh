#!/bin/bash

# Create initial issues for project tracking

GITHUB_USER="octaviodegodoy"
REPO_NAME="kalman-pairs-trading"

echo "Creating initial project issues..."

if ! command -v gh &> /dev/null; then
    echo "GitHub CLI required."
    exit 1
fi

# Issue 1: Documentation
gh issue create \
    --repo $GITHUB_USER/$REPO_NAME \
    --title "📚 Complete API documentation" \
    --body "Add comprehensive API documentation for all modules using Sphinx or MkDocs. 

Tasks:
- [ ] Set up documentation framework
- [ ] Document all classes and functions
- [ ] Add usage examples
- [ ] Create tutorial section
- [ ] Deploy to GitHub Pages" \
    --label documentation

# Issue 2: Testing
gh issue create \
    --repo $GITHUB_USER/$REPO_NAME \
    --title "🧪 Increase test coverage to 90%+" \
    --body "Expand test suite to achieve comprehensive code coverage. 

Tasks:
- [ ] Add edge case tests
- [ ] Add integration tests
- [ ] Add performance benchmarks
- [ ] Set up coverage reporting
- [ ] Add mutation testing" \
    --label enhancement

# Issue 3: Features
gh issue create \
    --repo $GITHUB_USER/$REPO_NAME \
    --title "✨ Add support for cryptocurrency pairs" \
    --body "Extend system to support cryptocurrency trading pairs.

Tasks:
- [ ] Add crypto data sources (Binance, Coinbase)
- [ ] Handle 24/7 trading
- [ ] Adjust parameters for crypto volatility
- [ ] Add crypto-specific risk management
- [ ] Test with major pairs (BTC/ETH, etc. )" \
    --label enhancement

# Issue 4: Performance
gh issue create \
    --repo $GITHUB_USER/$REPO_NAME \
    --title "⚡ Optimize Kalman Filter performance" \
    --body "Improve Kalman Filter update speed using Numba or Cython.

Target: 50,000+ updates/second

Tasks:
- [ ] Profile current performance
- [ ] Implement Numba JIT compilation
- [ ] Consider Cython alternative
- [ ] Benchmark improvements
- [ ] Update documentation" \
    --label performance

echo "✅ Initial issues created"
echo ""
echo "View issues at:  https://github.com/$GITHUB_USER/$REPO_NAME/issues"