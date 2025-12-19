"""
test_risk_management.py

Unit tests for risk management. 
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

sys. path.append(os.path. dirname(os.path.dirname(os.path.abspath(__file__))))

from src.risk_management import RiskManager


class TestRiskManager(unittest.TestCase):
    """Test cases for RiskManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.risk_manager = RiskManager(
            max_position_size=1. 0,
            max_drawdown_limit=0.15
        )
        
        # Generate synthetic data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.prices_a = pd.Series(np. cumsum(np.random.randn(100)) + 100, index=dates)
        self.prices_b = pd.Series(np.cumsum(np.random.randn(100)) + 50, index=dates)
    
    def test_position_size_calculation(self):
        """Test position size calculation."""
        spread_vol = 0.02
        
        position_size = self.risk_manager.calculate_position_size(
            spread_volatility=spread_vol,
            target_risk=0.01
        )
        
        self. assertIsInstance(position_size, float)
        self.assertGreater(position_size, 0)
        self.assertLessEqual(position_size, self.risk_manager.max_position_size)
    
    def test_correlation_check(self):
        """Test correlation checking."""
        is_valid, correlation = self.risk_manager. check_correlation(
            self. prices_a,
            self. prices_b,
            lookback=60
        )
        
        self. assertIsInstance(is_valid, bool)
        self.assertIsInstance(correlation, float)
        self.assertGreaterEqual(abs(correlation), -1.0)
        self.assertLessEqual(abs(correlation), 1.0)
    
    def test_drawdown_check(self):
        """Test drawdown checking."""
        # Create returns series with drawdown
        returns = pd.Series(np.random.randn(100) * 0.01)
        returns[50:60] = -0.05  # Simulate drawdown
        
        cumulative = (1 + returns).cumprod()
        
        is_safe, current_dd = self.risk_manager.check_drawdown(cumulative)
        
        self. assertIsInstance(is_safe, bool)
        self.assertIsInstance(current_dd, float)
        self.assertLessEqual(current_dd, 0.0)
    
    def test_kelly_criterion(self):
        """Test Kelly criterion calculation."""
        kelly = self.risk_manager.calculate_kelly_criterion(
            win_rate=0.6,
            avg_win=0.02,
            avg_loss=0.01
        )
        
        self.assertIsInstance(kelly, float)
        self.assertGreaterEqual(kelly, 0.0)
        self.assertLessEqual(kelly, self.risk_manager.max_position_size)
    
    def test_risk_limits_approval(self):
        """Test risk limits with good conditions."""
        signal = 1
        spread = pd.Series(np.random.randn(100))
        cumulative = pd.Series(np.linspace(1. 0, 1.2, 100))
        
        adjusted_signal, risk_info = self.risk_manager.apply_risk_limits(
            signal,
            spread,
            self.prices_a,
            self.prices_b,
            cumulative
        )
        
        self.assertIn('correlation', risk_info)
        self.assertIn('drawdown', risk_info)
        self.assertIn('risk_override', risk_info)
    
    def test_risk_limits_rejection(self):
        """Test risk limits with violation."""
        # Create scenario with excessive drawdown
        signal = 1
        spread = pd.Series(np.random.randn(100))
        
        # Large drawdown
        cumulative = pd.Series(np.linspace(1.0, 0.7, 100))
        
        adjusted_signal, risk_info = self.risk_manager.apply_risk_limits(
            signal,
            spread,
            self. prices_a,
            self. prices_b,
            cumulative
        )
        
        # Should be blocked
        self.assertEqual(adjusted_signal, 0)
        self.assertEqual(risk_info['risk_override'], 'DRAWDOWN_LIMIT_EXCEEDED')


if __name__ == '__main__': 
    unittest.main()