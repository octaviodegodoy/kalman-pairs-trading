"""
kalman_pairs_trader.py

Complete Kalman Filter implementation for pairs trading with dynamic hedge ratio estimation. 
Includes backtesting, risk management, and performance analytics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class KalmanFilterRegression:
    """
    Kalman Filter for dynamic linear regression in pairs trading.
    
    State Space Model:
        State equation: β(t) = β(t-1) + w(t), where w(t) ~ N(0, Q)
        Observation equation: y(t) = H(t) * β(t) + v(t), where v(t) ~ N(0, R)
    
    State vector: β(t) = [hedge_ratio, intercept]'
    Observation matrix: H(t) = [x(t), 1]
    """
    
    def __init__(self, delta:  float = 1e-4, Ve: float = 1e-3):
        """
        Initialize Kalman Filter.
        
        Args:
            delta: Process noise parameter (controls beta adaptation speed)
            Ve: Observation noise variance
        """
        self.delta = delta
        self. Ve = Ve
        
        # State estimates
        self.beta = None  # Hedge ratio
        self.alpha = None  # Intercept
        self.P = None  # State covariance matrix
        
        # History
        self.beta_history = []
        self.alpha_history = []
        self.P_history = []
        
    def initialize(self, initial_beta: float = 0.0, initial_alpha: float = 0.0):
        """Initialize state estimates."""
        self.beta = initial_beta
        self. alpha = initial_alpha
        self.P = np.array([[1.0, 0.0], 
                           [0.0, 1.0]])
        
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step:  Project state and covariance forward.
        
        Returns:
            state_prior: Prior state estimate
            P_prior: Prior covariance estimate
        """
        # State prediction: β(t|t-1) = β(t-1)
        state_prior = np.array([[self.beta], [self.alpha]])
        
        # Covariance prediction: P(t|t-1) = P(t-1) + Q
        Q = self.delta * np.eye(2)
        P_prior = self.P + Q
        
        return state_prior, P_prior
    
    def update(self, x: float, y: float) -> Tuple[float, float, float]:
        """
        Update step: Incorporate new observation.
        
        Args:
            x: Independent variable (Asset B price)
            y: Dependent variable (Asset A price)
            
        Returns:
            beta:  Updated hedge ratio
            alpha: Updated intercept
            spread: Current spread value
        """
        if self. beta is None:
            self. initialize()
        
        # Prediction
        state_prior, P_prior = self.predict()
        
        # Observation matrix: H = [x, 1]
        H = np.array([[x, 1.0]])
        
        # Innovation (measurement residual)
        y_pred = (H @ state_prior)[0, 0]
        innovation = y - y_pred
        
        # Innovation covariance:  S = H * P * H' + R
        S = (H @ P_prior @ H.T)[0, 0] + self. Ve
        
        # Kalman gain: K = P * H' / S
        K = (P_prior @ H.T) / S
        
        # State update
        state_update = state_prior + K * innovation
        self. beta = state_update[0, 0]
        self.alpha = state_update[1, 0]
        
        # Covariance update:  P = (I - K * H) * P_prior
        I = np.eye(2)
        self.P = (I - K @ H) @ P_prior
        
        # Calculate spread
        spread = y - (self.beta * x + self.alpha)
        
        # Store history
        self.beta_history.append(self.beta)
        self.alpha_history.append(self.alpha)
        self.P_history.append(self.P. copy())
        
        return self.beta, self.alpha, spread
    
    def get_spread(self, x: float, y: float) -> float:
        """Calculate spread given current prices."""
        if self.beta is None:
            return 0.0
        return y - (self.beta * x + self.alpha)
    
    def get_confidence_interval(self, confidence:  float = 0.95) -> Tuple[float, float]: 
        """
        Get confidence interval for hedge ratio.
        
        Args:
            confidence:  Confidence level (default 95%)
            
        Returns: 
            lower_bound, upper_bound for hedge ratio
        """
        if self.P is None:
            return (0.0, 0.0)
        
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)
        std_error = np.sqrt(self.P[0, 0])
        
        lower = self.beta - z_score * std_error
        upper = self.beta + z_score * std_error
        
        return lower, upper


class PairsTradingStrategy:
    """
    Complete pairs trading strategy using Kalman Filter.
    """
    
    def __init__(
        self,
        entry_zscore: float = 2.0,
        exit_zscore:  float = 0.5,
        stop_loss_zscore: float = 4.0,
        lookback_window: int = 20,
        delta: float = 1e-4,
        Ve: float = 1e-3,
        position_size: float = 1.0,
        transaction_cost: float = 0.001
    ):
        """
        Initialize pairs trading strategy.
        
        Args:
            entry_zscore: Z-score threshold for entry
            exit_zscore: Z-score threshold for exit
            stop_loss_zscore: Z-score threshold for stop loss
            lookback_window: Rolling window for spread statistics
            delta:  Kalman filter process noise
            Ve: Kalman filter observation noise
            position_size: Position sizing factor
            transaction_cost: Transaction cost (as fraction of trade value)
        """
        self. entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_loss_zscore = stop_loss_zscore
        self.lookback_window = lookback_window
        self. position_size = position_size
        self.transaction_cost = transaction_cost
        
        self.kf = KalmanFilterRegression(delta=delta, Ve=Ve)
        
        # Trading state
        self.position = 0  # 1:  long spread, -1: short spread, 0: flat
        self. entry_price_a = None
        self.entry_price_b = None
        self.entry_spread = None
        
    def calculate_zscore(self, spread_series: pd.Series) -> float:
        """Calculate z-score of current spread."""
        if len(spread_series) < self.lookback_window:
            return 0.0
        
        recent_spread = spread_series.iloc[-self.lookback_window:]
        mean = recent_spread.mean()
        std = recent_spread.std()
        
        if std == 0 or np.isnan(std):
            return 0.0
        
        current_spread = spread_series.iloc[-1]
        zscore = (current_spread - mean) / std
        
        return zscore
    
    def generate_signal(
        self,
        price_a: float,
        price_b: float,
        spread_series: pd.Series
    ) -> Tuple[int, Dict]: 
        """
        Generate trading signal. 
        
        Args:
            price_a: Current price of asset A
            price_b: Current price of asset B
            spread_series: Historical spread series
            
        Returns: 
            signal: 1 (long), -1 (short), 0 (flat)
            info: Dictionary with signal information
        """
        # Update Kalman filter
        beta, alpha, spread = self.kf.update(price_b, price_a)
        
        # Calculate z-score
        zscore = self.calculate_zscore(spread_series)
        
        signal = 0
        action = "HOLD"
        
        # Check for entry signals
        if self.position == 0:
            if zscore > self.entry_zscore:
                # Spread is too high:  short spread (sell A, buy B)
                signal = -1
                action = "SHORT_SPREAD"
                self.entry_price_a = price_a
                self.entry_price_b = price_b
                self.entry_spread = spread
                
            elif zscore < -self.entry_zscore:
                # Spread is too low: long spread (buy A, sell B)
                signal = 1
                action = "LONG_SPREAD"
                self.entry_price_a = price_a
                self. entry_price_b = price_b
                self.entry_spread = spread
        
        # Check for exit signals
        elif self.position != 0:
            # Normal exit
            if abs(zscore) < self.exit_zscore:
                signal = 0
                action = "EXIT_NORMAL"
                
            # Stop loss
            elif abs(zscore) > self.stop_loss_zscore:
                signal = 0
                action = "EXIT_STOPLOSS"
                
            # Mean reversion reversal
            elif (self.position == 1 and zscore > self.entry_zscore) or \
                 (self.position == -1 and zscore < -self.entry_zscore):
                signal = 0
                action = "EXIT_REVERSAL"
            
            else:
                signal = self.position
        
        info = {
            'beta': beta,
            'alpha':  alpha,
            'spread': spread,
            'zscore': zscore,
            'action': action,
            'position': self.position
        }
        
        # Update position
        if signal != self.position:
            self.position = signal
        
        return signal, info
    
    def calculate_trade_pnl(
        self,
        entry_a: float,
        entry_b: float,
        exit_a: float,
        exit_b: float,
        beta: float,
        position: int
    ) -> float:
        """
        Calculate P&L for a trade.
        
        Args:
            entry_a, entry_b: Entry prices
            exit_a, exit_b:  Exit prices
            beta: Hedge ratio
            position: 1 for long spread, -1 for short spread
            
        Returns:
            pnl:  Profit/loss as fraction
        """
        if position == 1:
            # Long spread:  buy A, sell B
            pnl_a = (exit_a - entry_a) / entry_a
            pnl_b = -(exit_b - entry_b) / entry_b * beta
        else:
            # Short spread:  sell A, buy B
            pnl_a = -(exit_a - entry_a) / entry_a
            pnl_b = (exit_b - entry_b) / entry_b * beta
        
        # Account for transaction costs
        total_pnl = pnl_a + pnl_b
        costs = 2 * self.transaction_cost  # Entry and exit
        
        return total_pnl - costs


class KalmanPairsBacktest:
    """
    Backtesting engine for Kalman Filter pairs trading. 
    """
    
    def __init__(self, strategy: PairsTradingStrategy):
        """
        Initialize backtester.
        
        Args:
            strategy: PairsTradingStrategy instance
        """
        self.strategy = strategy
        
    def run(
        self,
        df: pd.DataFrame,
        asset_a_col: str = 'asset1',
        asset_b_col: str = 'asset2'
    ) -> pd.DataFrame:
        """
        Run backtest on historical data.
        
        Args:
            df: DataFrame with asset prices (columns: asset1, asset2)
            asset_a_col: Column name for asset A
            asset_b_col: Column name for asset B
            
        Returns:
            DataFrame with backtest results
        """
        results = []
        spread_series = pd.Series(dtype=float)
        
        for idx, row in df.iterrows():
            price_a = row[asset_a_col]
            price_b = row[asset_b_col]
            
            # Update spread series
            if len(spread_series) > 0:
                spread_series = pd.concat([
                    spread_series,
                    pd.Series([self.strategy.kf.get_spread(price_b, price_a)], index=[idx])
                ])
            else:
                spread_series = pd.Series([0.0], index=[idx])
            
            # Generate signal
            signal, info = self. strategy.generate_signal(
                price_a, price_b, spread_series
            )
            
            results.append({
                'date':  idx,
                'price_a': price_a,
                'price_b': price_b,
                'beta': info['beta'],
                'alpha': info['alpha'],
                'spread': info['spread'],
                'zscore': info['zscore'],
                'signal': signal,
                'action':  info['action'],
                'position': info['position']
            })
        
        results_df = pd.DataFrame(results)
        results_df. set_index('date', inplace=True)
        
        # Calculate returns
        results_df = self._calculate_returns(results_df, df, asset_a_col, asset_b_col)
        
        return results_df
    
    def _calculate_returns(
        self,
        results_df: pd.DataFrame,
        price_df: pd.DataFrame,
        asset_a_col: str,
        asset_b_col: str
    ) -> pd.DataFrame:
        """Calculate strategy returns."""
        
        # Calculate spread returns
        results_df['spread_return'] = results_df['spread'].pct_change()
        
        # Calculate strategy returns
        results_df['position_lagged'] = results_df['position'].shift(1).fillna(0)
        results_df['strategy_return'] = (
            results_df['position_lagged'] * results_df['spread_return']
        )
        
        # Account for transaction costs
        position_changes = results_df['position']. diff().abs()
        transaction_costs = position_changes * self.strategy.transaction_cost
        results_df['strategy_return'] -= transaction_costs
        
        # Cumulative returns
        results_df['cumulative_return'] = (
            1 + results_df['strategy_return']. fillna(0)
        ).cumprod()
        
        # Drawdown
        cumulative_max = results_df['cumulative_return'].cummax()
        results_df['drawdown'] = (
            results_df['cumulative_return'] - cumulative_max
        ) / cumulative_max
        
        return results_df
    
    def get_performance_metrics(self, results_df: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics.
        
        Args:
            results_df:  Backtest results DataFrame
            
        Returns:
            Dictionary of performance metrics
        """
        returns = results_df['strategy_return'].dropna()
        
        if len(returns) == 0:
            return {}
        
        total_return = results_df['cumulative_return'].iloc[-1] - 1
        
        # Annualized metrics (assuming daily data)
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Drawdown
        max_drawdown = results_df['drawdown'].min()
        
        # Win rate
        trades = results_df[results_df['action']. str.contains('EXIT')].copy()
        if len(trades) > 0:
            winning_trades = len(trades[trades['strategy_return'] > 0])
            win_rate = winning_trades / len(trades)
        else:
            win_rate = 0
        
        # Number of trades
        num_trades = len(trades)
        
        metrics = {
            'Total Return': f"{total_return * 100:.2f}%",
            'Annual Return': f"{annual_return * 100:.2f}%",
            'Annual Volatility': f"{annual_vol * 100:.2f}%",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Max Drawdown': f"{max_drawdown * 100:.2f}%",
            'Win Rate': f"{win_rate * 100:.2f}%",
            'Number of Trades': num_trades,
            'Avg Return per Trade': f"{returns.mean() * 100:.4f}%"
        }
        
        return metrics


class VisualizationTools:
    """Visualization tools for pairs trading analysis."""
    
    @staticmethod
    def plot_complete_analysis(
        results_df: pd.DataFrame,
        metrics: Dict,
        title: str = "Kalman Filter Pairs Trading Analysis"
    ):
        """
        Create comprehensive visualization of backtest results.
        
        Args:
            results_df:  Backtest results DataFrame
            metrics:  Performance metrics dictionary
            title: Plot title
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(5, 2, hspace=0.3, wspace=0.3)
        
        # 1. Asset prices
        ax1 = fig. add_subplot(gs[0, : ])
        ax1.plot(results_df. index, results_df['price_a'], label='Asset A', alpha=0.7)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(results_df.index, results_df['price_b'], 
                      label='Asset B', color='orange', alpha=0.7)
        ax1.set_title('Asset Prices', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Dynamic hedge ratio (beta)
        ax2 = fig.add_subplot(gs[1, : ])
        ax2.plot(results_df.index, results_df['beta'], color='green', linewidth=1. 5)
        ax2.set_title('Dynamic Hedge Ratio (Beta)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Beta')
        ax2.grid(True, alpha=0.3)
        
        # 3. Spread and z-score
        ax3 = fig.add_subplot(gs[2, :])
        ax3.plot(results_df. index, results_df['spread'], label='Spread', color='purple')
        ax3_twin = ax3.twinx()
        ax3_twin.plot(results_df.index, results_df['zscore'], 
                      label='Z-Score', color='red', alpha=0.7, linewidth=1)
        ax3_twin.axhline(y=2, color='r', linestyle='--', alpha=0.5, label='Entry threshold')
        ax3_twin.axhline(y=-2, color='r', linestyle='--', alpha=0.5)
        ax3_twin.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('Spread and Z-Score', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # 4. Trading signals
        ax4 = fig.add_subplot(gs[3, :])
        # Plot position changes
        long_entries = results_df[results_df['action'] == 'LONG_SPREAD']
        short_entries = results_df[results_df['action'] == 'SHORT_SPREAD']
        exits = results_df[results_df['action']. str.contains('EXIT')]
        
        ax4.scatter(long_entries. index, long_entries['zscore'], 
                   color='green', marker='^', s=100, label='Long Entry', zorder=5)
        ax4.scatter(short_entries.index, short_entries['zscore'], 
                   color='red', marker='v', s=100, label='Short Entry', zorder=5)
        ax4.scatter(exits.index, exits['zscore'], 
                   color='blue', marker='x', s=100, label='Exit', zorder=5)
        ax4.plot(results_df.index, results_df['zscore'], 
                color='gray', alpha=0.5, linewidth=1)
        ax4.axhline(y=2, color='r', linestyle='--', alpha=0.5)
        ax4.axhline(y=-2, color='r', linestyle='--', alpha=0.5)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('Trading Signals', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Cumulative returns
        ax5 = fig.add_subplot(gs[4, 0])
        ax5.plot(results_df.index, results_df['cumulative_return'], 
                color='darkblue', linewidth=2)
        ax5.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Cumulative Return')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        
        # 6. Drawdown
        ax6 = fig.add_subplot(gs[4, 1])
        ax6.fill_between(results_df.index, results_df['drawdown'] * 100, 0, 
                        color='red', alpha=0.3)
        ax6.plot(results_df.index, results_df['drawdown'] * 100, 
                color='darkred', linewidth=1)
        ax6.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Drawdown (%)')
        ax6.grid(True, alpha=0.3)
        
        # Add title and metrics
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        
        # Add metrics text box
        metrics_text = '\n'.join([f"{k}: {v}" for k, v in metrics.items()])
        fig.text(0.02, 0.98, metrics_text, transform=fig.transFigure,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.show()
    
    @staticmethod
    def plot_beta_distribution(results_df: pd.DataFrame):
        """Plot distribution of hedge ratio estimates."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Time series
        axes[0].plot(results_df.index, results_df['beta'], color='green')
        axes[0].set_title('Beta Over Time')
        axes[0].set_ylabel('Beta')
        axes[0].grid(True, alpha=0.3)
        
        # Distribution
        axes[1].hist(results_df['beta']. dropna(), bins=50, 
                    color='green', alpha=0.7, edgecolor='black')
        axes[1].axvline(results_df['beta'].mean(), color='red', 
                       linestyle='--', label=f"Mean: {results_df['beta'].mean():.4f}")
        axes[1].set_title('Beta Distribution')
        axes[1].set_xlabel('Beta')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def download_data(
    ticker1: str,
    ticker2: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Download price data for two assets.
    
    Args:
        ticker1: Ticker for asset 1
        ticker2: Ticker for asset 2
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with adjusted close prices
    """
    try:
        import yfinance as yf
        
        print(f"Downloading data for {ticker1} and {ticker2}...")
        data1 = yf.download(ticker1, start=start_date, end=end_date, progress=False)['Adj Close']
        data2 = yf.download(ticker2, start=start_date, end=end_date, progress=False)['Adj Close']
        
        df = pd.DataFrame({
            'asset1': data1,
            'asset2': data2
        }).dropna()
        
        print(f"Downloaded {len(df)} data points.")
        return df
        
    except ImportError:
        print("yfinance not installed. Install with: pip install yfinance")
        return None
    except Exception as e: 
        print(f"Error downloading data: {e}")
        return None


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("KALMAN FILTER PAIRS TRADING - COMPLETE IMPLEMENTATION")
    print("=" * 80)
    
    # Configuration
    TICKER_A = 'GLD'  # Gold ETF
    TICKER_B = 'GDX'  # Gold Miners ETF
    START_DATE = '2020-01-01'
    END_DATE = '2024-01-01'
    
    # Download data
    df = download_data(TICKER_A, TICKER_B, START_DATE, END_DATE)
    
    if df is not None:
        # Initialize strategy
        strategy = PairsTradingStrategy(
            entry_zscore=2.0,
            exit_zscore=0.5,
            stop_loss_zscore=4.0,
            lookback_window=20,
            delta=1e-4,
            Ve=1e-3,
            transaction_cost=0.001
        )
        
        # Run backtest
        print("\nRunning backtest...")
        backtester = KalmanPairsBacktest(strategy)
        results = backtester.run(df)
        
        # Get performance metrics
        metrics = backtester.get_performance_metrics(results)
        
        # Print metrics
        print("\n" + "=" * 80)
        print("PERFORMANCE METRICS")
        print("=" * 80)
        for key, value in metrics.items():
            print(f"{key: . <30} {value: >20}")
        
        # Visualize results
        print("\nGenerating visualizations...")
        viz = VisualizationTools()
        viz.plot_complete_analysis(
            results,
            metrics,
            title=f"Kalman Filter Pairs Trading:  {TICKER_A} vs {TICKER_B}"
        )
        
        viz.plot_beta_distribution(results)
        
        print("\n" + "=" * 80)
        print("Analysis complete!")
        print("=" * 80)