# Deployment Guide

Complete guide for deploying the Kalman Pairs Trading system. 

## 📋 Prerequisites

- Python 3.8+
- Git
- Docker (optional)
- GitHub account
- Cloud platform account (optional:  Heroku, AWS, GCP)

## 🚀 Deployment Options

### Option 1: Local Deployment

```bash
# Clone repository
git clone https://github.com/octaviodegodoy/kalman-pairs-trading.git
cd kalman-pairs-trading

# Install dependencies
make install

# Run tests
make test

# Start dashboard
make run