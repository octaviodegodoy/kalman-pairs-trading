"""
basic_backtest.py

Simple example of running a pairs trading backtest. 
"""

import sys
import os
sys.path. append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pairs_strategy import PairsTradingStrategy
from src.backtester import KalmanPairsBacktest
from src.data_manager import download_data
from src. visualization import VisualizationTools


def main():
    """Run basic backtest example."""
    
    print("=" * 80)
    print("KALMAN FILTER PAIRS TRADING - BASIC BACKTEST")
    print("=" * 80)
    
    # 1. Download data
    print("\n1. Downloading data...")
    df = download_data('GLD', 'GDX', '2020-01-01', '2024-01-01')
    print(f"   Downloaded {len(df)} observations")
    
    # 2. Initialize strategy
    print("\n2. Initializing strategy...")
    strategy = PairsTradingStrategy(
        entry_zscore=2.0,
        exit_zscore=0.5,
        stop_loss_zscore=4.0,
        lookback_window=20,
        delta=1e-4,
        transaction_cost=0.001
    )
    print("   Strategy initialized")
    
    # 3. Run backtest
    print("\n3. Running backtest...")
    backtester = KalmanPairsBacktest(strategy)
    results = backtester.run(df)
    print("   Backtest complete")
    
    # 4. Calculate metrics
    print("\n4. Calculating performance metrics...")
    metrics = backtester.get_performance_metrics(results)
    
    # 5. Print results
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    for key, value in metrics.items():
        print(f"{key: . <30} {value: >20}")
    print("=" * 80)
    
    # 6. Visualize
    print("\n5. Generating visualizations...")
    viz = VisualizationTools()
    viz.plot_complete_analysis(
        results,
        metrics,
        title="GLD/GDX Pairs Trading Backtest"
    )
    
    viz.plot_beta_distribution(results)
    
    print("\nBacktest complete!")


if __name__ == "__main__":
    main()