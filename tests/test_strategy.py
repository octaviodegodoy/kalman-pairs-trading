"""
test_strategy.py

Unit tests for pairs trading strategy. 
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from kalman_pairs_trader import PairsTradingStrategy


class TestPairsTradingStrategy(unittest.TestCase):
    """Test cases for PairsTradingStrategy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = PairsTradingStrategy(
            entry_zscore=2.0,
            exit_zscore=0.5,
            stop_loss_zscore=4.0,
            lookback_window=20
        )
        
        # Generate synthetic price data
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        
        self.price_a = pd.Series(np.cumsum(np.random.randn(n)) + 100, index=dates)
        self.price_b = pd.Series(np.cumsum(np.random.randn(n)) + 50, index=dates)
    
    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(self.strategy.entry_zscore, 2.0)
        self.assertEqual(self.strategy.exit_zscore, 0.5)
        self.assertEqual(self.strategy.position, 0)
        self.assertIsNone(self.strategy. entry_price_a)
    
    def test_zscore_calculation(self):
        """Test z-score calculation."""
        # Create spread series with known statistics
        spread = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 10)
        
        zscore = self.strategy.calculate_zscore(spread)
        
        self.assertIsInstance(zscore, float)
        self.assertFalse(np.isnan(zscore))
    
    def test_signal_generation(self):
        """Test signal generation."""
        spread_series = pd.Series(dtype=float)
        
        for i in range(50):
            signal, info = self.strategy.generate_signal(
                self.price_a. iloc[i],
                self.price_b. iloc[i],
                spread_series
            )
            
            spread_series[self.price_a.index[i]] = info['spread']
            
            # Signal should be -1, 0, or 1
            self.assertIn(signal, [-1, 0, 1])
            
            # Info should contain required keys
            self.assertIn('beta', info)
            self.assertIn('spread', info)
            self.assertIn('zscore', info)
            self.assertIn('action', info)
    
    def test_position_tracking(self):
        """Test that positions are tracked correctly."""
        spread_series = pd.Series(dtype=float)
        
        # Initial position should be 0
        self.assertEqual(self.strategy. position, 0)
        
        # Generate signals
        for i in range(50):
            signal, info = self.strategy.generate_signal(
                self.price_a. iloc[i],
                self. price_b.iloc[i],
                spread_series
            )
            
            spread_series[self.price_a.index[i]] = info['spread']
        
        # Position should be updated
        self.assertIn(self.strategy.position, [-1, 0, 1])
    
    def test_entry_signals(self):
        """Test entry signal generation."""
        # Create extreme z-score scenario
        spread = pd.Series([0.0] * 30)  # Stable spread
        spread = pd.concat([spread, pd.Series([10.0])])  # Sudden jump
        
        # Mock z-score calculation to return high value
        original_calc = self.strategy.calculate_zscore
        self.strategy.calculate_zscore = lambda x: 2.5  # Above entry threshold
        
        signal, info = self.strategy.generate_signal(100, 50, spread)
        
        # Should generate entry signal
        self. assertNotEqual(signal, 0)
        self.assertIn(info['action'], ['LONG_SPREAD', 'SHORT_SPREAD'])
        
        # Restore original method
        self.strategy. calculate_zscore = original_calc
    
    def test_exit_signals(self):
        """Test exit signal generation."""
        # Set up a position
        self.strategy.position = 1
        self.strategy.entry_price_a = 100
        self.strategy.entry_price_b = 50
        
        # Create spread that reverts to mean
        spread = pd.Series([1.0] * 30)
        
        # Mock z-score to return low value
        original_calc = self.strategy.calculate_zscore
        self.strategy.calculate_zscore = lambda x: 0.3  # Below exit threshold
        
        signal, info = self.strategy.generate_signal(100, 50, spread)
        
        # Should generate exit signal
        self.assertEqual(signal, 0)
        self.assertIn('EXIT', info['action'])
        
        # Restore
        self.strategy.calculate_zscore = original_calc
    
    def test_stop_loss(self):
        """Test stop loss functionality."""
        self.strategy.position = 1
        
        # Mock extreme z-score
        original_calc = self.strategy.calculate_zscore
        self.strategy.calculate_zscore = lambda x: 5.0  # Above stop loss
        
        spread = pd.Series([1.0] * 30)
        signal, info = self.strategy.generate_signal(100, 50, spread)
        
        # Should trigger stop loss
        self.assertEqual(signal, 0)
        self.assertEqual(info['action'], 'EXIT_STOPLOSS')
        
        # Restore
        self.strategy. calculate_zscore = original_calc


class TestStrategyEdgeCases(unittest.TestCase):
    """Test edge cases for strategy."""
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        strategy = PairsTradingStrategy(lookback_window=20)
        
        # Only 5 observations
        spread = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        
        zscore = strategy.calculate_zscore(spread)
        
        # Should return 0 (or handle gracefully)
        self.assertEqual(zscore, 0.0)
    
    def test_zero_volatility(self):
        """Test behavior with zero volatility."""
        strategy = PairsTradingStrategy()
        
        # Constant spread (no volatility)
        spread = pd.Series([5.0] * 30)
        
        zscore = strategy.calculate_zscore(spread)
        
        # Should handle division by zero
        self.assertFalse(np.isnan(zscore))
        self.assertEqual(zscore, 0.0)


if __name__ == '__main__':
    unittest.main()