"""
Kalman Filter Pairs Trading System

A comprehensive pairs trading framework with dynamic hedge ratio estimation. 
"""

__version__ = "1.0.0"
__author__ = "Octavio de Godoy"

# Import existing modules
from .cointegration_tests import CointegrationTester
from .data_manager import DataManager
from .pair_selection import PairSelector, PairScore
from .parameter_optimizer import ParameterOptimizer
from .realtime_trader import RealtimeTrader, BrokerInterface, AlpacaBroker
from .ml_features import LSTMBetaPredictor, RegimeDetector, FeatureEngineer

__all__ = [
    'CointegrationTester',
    'DataManager',
    'PairSelector',
    'PairScore',
    'ParameterOptimizer',
    'RealtimeTrader',
    'BrokerInterface',
    'AlpacaBroker',
    'LSTMBetaPredictor',
    'RegimeDetector',
    'FeatureEngineer',
]