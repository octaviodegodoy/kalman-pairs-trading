"""
Kalman Filter Pairs Trading System

A comprehensive pairs trading framework with dynamic hedge ratio estimation. 
"""

__version__ = "1.0.0"
__author__ = "Octavio de Godoy"

import sys
from pathlib import Path

# Add parent directory to path to import from root modules
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from kalman_pairs_trader import (
    KalmanFilterRegression,
    PairsTradingStrategy,
    KalmanPairsBacktest,
    VisualizationTools
)
from risk_management import RiskManager

__all__ = [
    'KalmanFilterRegression',
    'PairsTradingStrategy',
    'KalmanPairsBacktest',
    'RiskManager',
    'VisualizationTools',
]