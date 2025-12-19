"""
risk_management.py

Advanced risk management for Kalman Filter pairs trading.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

class RiskManager:
    """
    Risk management system for pairs trading.
    """
    
    def __init__(
        self,
        max_position_size: float = 1.0,
        max_portfolio_risk: float = 0.02,
        max_correlation_threshold: float = 0.7,
        min_correlation_threshold: float = 0.3,
        max_drawdown_limit: float = 0.15,
        volatility_lookback:  int = 20
    ):
        """
        Initialize risk manager.
        
        Args:
            max_position_size: Maximum position size as fraction of capital
            max_portfolio_risk:  Maximum portfolio risk (volatility)
            max_correlation_threshold: Maximum correlation for pair viability
            min_correlation_threshold: Minimum correlation for pair viability
            max_drawdown_limit: Maximum drawdown before stopping
            volatility_lookback:  Lookback period for volatility calculation
        """
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlation_threshold = max_correlation_threshold
        self.min_correlation_threshold = min_correlation_threshold
        self.max_drawdown_limit = max_drawdown_limit
        self.volatility_lookback = volatility_lookback
        
    def calculate_position_size(
        self,
        spread_volatility: float,
        target_risk: float = 0.01
    ) -> float:
        """
        Calculate optimal position size based on volatility. 
        
        Args:
            spread_volatility: Volatility of spread
            target_risk:  Target risk per trade
            
        Returns:
            Position size as fraction
        """
        if spread_volatility == 0:
            return 0. 0
        
        position_size = target_risk / spread_volatility
        position_size = min(position_size, self.max_position_size)
        
        return position_size
    
    def check_correlation(
        self,
        asset_a_prices: pd.Series,
        asset_b_prices: pd.Series,
        lookback:  int = 60
    ) -> Tuple[bool, float]:
        """
        Check if correlation is within acceptable range.
        
        Args:
            asset_a_prices: Price series for asset A
            asset_b_prices: Price series for asset B
            lookback: Lookback period for correlation
            
        Returns:
            is_valid: Whether correlation is acceptable
            correlation: Correlation value
        """
        if len(asset_a_prices) < lookback:
            return False, 0.0
        
        recent_a = asset_a_prices. iloc[-lookback:]
        recent_b = asset_b_prices.iloc[-lookback:]
        
        correlation = recent_a. corr(recent_b)
        
        is_valid = (
            self.min_correlation_threshold <= abs(correlation) <= self.max_correlation_threshold
        )
        
        return is_valid, correlation
    
    def calculate_spread_volatility(
        self,
        spread_series: pd.Series
    ) -> float:
        """
        Calculate spread volatility.
        
        Args:
            spread_series:  Spread time series
            
        Returns: 
            Annualized volatility
        """
        if len(spread_series) < self.volatility_lookback:
            return 0.0
        
        recent_spread = spread_series.iloc[-self.volatility_lookback:]
        returns = recent_spread.pct_change().dropna()
        
        volatility = returns.std() * np.sqrt(252)
        
        return volatility
    
    def check_drawdown(
        self,
        cumulative_returns: pd.Series
    ) -> Tuple[bool, float]:
        """
        Check if drawdown exceeds limit.
        
        Args:
            cumulative_returns: Cumulative returns series
            
        Returns:
            is_safe: Whether drawdown is within limit
            current_drawdown: Current drawdown value
        """
        if len(cumulative_returns) == 0:
            return True, 0.0
        
        cumulative_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - cumulative_max) / cumulative_max
        current_drawdown = drawdown.iloc[-1]
        
        is_safe = current_drawdown > -self.max_drawdown_limit
        
        return is_safe, current_drawdown
    
    def calculate_kelly_criterion(
        self,
        win_rate: float,
        avg_win:  float,
        avg_loss:  float
    ) -> float:
        """
        Calculate Kelly criterion for position sizing.
        
        Args:
            win_rate: Historical win rate
            avg_win:  Average winning trade return
            avg_loss: Average losing trade return (absolute value)
            
        Returns: 
            Kelly fraction
        """
        if avg_loss == 0 or win_rate == 0:
            return 0.0
        
        win_loss_ratio = avg_win / avg_loss
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Use half-Kelly for safety
        kelly = max(0, min(kelly * 0.5, self.max_position_size))
        
        return kelly
    
    def apply_risk_limits(
        self,
        signal:  int,
        spread_series: pd.Series,
        asset_a_prices: pd.Series,
        asset_b_prices: pd.Series,
        cumulative_returns: pd.Series
    ) -> Tuple[int, Dict]: 
        """
        Apply risk management rules to trading signal.
        
        Args:
            signal: Raw trading signal
            spread_series:  Spread history
            asset_a_prices:  Asset A price history
            asset_b_prices: Asset B price history
            cumulative_returns: Cumulative returns history
            
        Returns: 
            adjusted_signal: Risk-adjusted signal
            risk_info: Dictionary with risk information
        """
        risk_info = {}
        
        # Check correlation
        is_corr_valid, correlation = self.check_correlation(
            asset_a_prices, asset_b_prices
        )
        risk_info['correlation'] = correlation
        risk_info['correlation_valid'] = is_corr_valid
        
        # Check drawdown
        is_dd_safe, current_dd = self.check_drawdown(cumulative_returns)
        risk_info['drawdown'] = current_dd
        risk_info['drawdown_safe'] = is_dd_safe
        
        # Calculate spread volatility
        spread_vol = self.calculate_spread_volatility(spread_series)
        risk_info['spread_volatility'] = spread_vol
        
        # Calculate position size
        position_size = self.calculate_position_size(spread_vol)
        risk_info['position_size'] = position_size
        
        # Adjust signal based on risk checks
        adjusted_signal = signal
        
        if not is_corr_valid: 
            adjusted_signal = 0
            risk_info['risk_override'] = 'CORRELATION_OUT_OF_RANGE'
        elif not is_dd_safe:
            adjusted_signal = 0
            risk_info['risk_override'] = 'DRAWDOWN_LIMIT_EXCEEDED'
        elif spread_vol > self.max_portfolio_risk * 2:
            adjusted_signal = 0
            risk_info['risk_override'] = 'EXCESSIVE_VOLATILITY'
        else:
            risk_info['risk_override'] = 'NONE'
        
        return adjusted_signal, risk_info


# Integration example
class RiskAwarePairsStrategy(PairsTradingStrategy):
    """
    Pairs trading strategy with integrated risk management.
    """
    
    def __init__(self, *args, risk_manager: Optional[RiskManager] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.risk_manager = risk_manager or RiskManager()
        
    def generate_signal_with_risk(
        self,
        price_a: float,
        price_b: float,
        spread_series: pd.Series,
        price_a_series: pd.Series,
        price_b_series: pd.Series,
        cumulative_returns:  pd.Series
    ) -> Tuple[int, Dict]:
        """Generate signal with risk management overlay."""
        
        # Get base signal
        signal, info = self.generate_signal(price_a, price_b, spread_series)
        
        # Apply risk management
        adjusted_signal, risk_info = self.risk_manager. apply_risk_limits(
            signal,
            spread_series,
            price_a_series,
            price_b_series,
            cumulative_returns
        )
        
        # Merge info
        info. update(risk_info)
        info['original_signal'] = signal
        info['adjusted_signal'] = adjusted_signal
        
        return adjusted_signal, info