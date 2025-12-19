"""
Kalman Filter Pairs Trading System

A comprehensive pairs trading framework with dynamic hedge ratio estimation. 
"""

__version__ = "1.0.0"
__author__ = "Octavio de Godoy"

from .kalman_filter import KalmanFilterRegression
from .pairs_strategy import PairsTradingStrategy
from .backtester import KalmanPairsBacktest
from .risk_management import RiskManager
from .visualization import VisualizationTools

__all__ = [
    'KalmanFilterRegression',
    'PairsTradingStrategy',
    'KalmanPairsBacktest',
    'RiskManager',
    'VisualizationTools',
]