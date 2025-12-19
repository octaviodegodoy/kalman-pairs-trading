"""
cli.py

Command-line interface for pairs trading system.
"""

import argparse
import sys
from datetime import datetime

from . pairs_strategy import PairsTradingStrategy
from .backtester import KalmanPairsBacktest
from .data_manager import download_data
from .parameter_optimizer import ParameterOptimizer
from .visualization import VisualizationTools


def run_backtest():
    """CLI for running backtests."""
    parser = argparse.ArgumentParser(description='Run pairs trading backtest')
    
    parser.add_argument('asset_a', help='First asset ticker')
    parser.add_argument('asset_b', help='Second asset ticker')
    parser.add_argument('--start', default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default=datetime.now().strftime('%Y-%m-%d'), help='End date')
    parser.add_argument('--entry-zscore', type=float, default=2.0, help='Entry z-score threshold')
    parser.add_argument('--exit-zscore', type=float, default=0.5, help='Exit z-score threshold')
    parser.add_argument('--delta', type=float, default=1e-4, help='Kalman filter delta')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    
    args = parser.parse_args()
    
    print(f"Running backtest for {args.asset_a}/{args.asset_b}")
    print(f"Date range: {args.start} to {args.end}")
    
    # Download data
    print("Downloading data...")
    df = download_data(args.asset_a, args.asset_b, args.start, args.end)
    
    # Create strategy
    strategy = PairsTradingStrategy(
        entry_zscore=args.entry_zscore,
        exit_zscore=args.exit_zscore,
        delta=args.delta
    )
    
    # Run backtest
    print("Running backtest...")
    backtester = KalmanPairsBacktest(strategy)
    results = backtester.run(df)
    
    # Get metrics
    metrics = backtester.get_performance_metrics(results)
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    for key, value in metrics.items():
        print(f"{key: . <30} {value: >20}")
    print("=" * 80)
    
    # Plot
    if not args.no_plot:
        viz = VisualizationTools()
        viz.plot_complete_analysis(results, metrics)


def optimize_parameters():
    """CLI for parameter optimization."""
    parser = argparse.ArgumentParser(description='Optimize strategy parameters')
    
    parser.add_argument('asset_a', help='First asset ticker')
    parser.add_argument('asset_b', help='Second asset ticker')
    parser.add_argument('--start', default='2020-01-01', help='Start date')
    parser.add_argument('--end', default=datetime.now().strftime('%Y-%m-%d'), help='End date')
    parser.add_argument('--method', default='grid', choices=['grid', 'random', 'bayesian'])
    parser.add_argument('--objective', default='sharpe_ratio', help='Objective function')
    
    args = parser.parse_args()
    
    print(f"Optimizing parameters for {args.asset_a}/{args.asset_b}")
    
    # Download data
    df = download_data(args.asset_a, args.asset_b, args.start, args.end)
    
    # Initialize optimizer
    optimizer = ParameterOptimizer(PairsTradingStrategy, df, args.objective)
    
    # Run optimization
    if args.method == 'grid': 
        param_grid = {
            'entry_zscore':  [1.5, 2.0, 2.5],
            'exit_zscore': [0.3, 0.5, 0.7],
            'delta': [1e-5, 1e-4, 1e-3]
        }
        best = optimizer.grid_search(param_grid)
    
    elif args.method == 'random':
        param_dist = {
            'entry_zscore': (1.5, 3.0),
            'exit_zscore': (0.2, 1.0),
            'delta':  (1e-5, 1e-3)
        }
        best = optimizer.random_search(param_dist, n_iter=100)
    
    print("\n" + "=" * 80)
    print("BEST PARAMETERS")
    print("=" * 80)
    for param, value in best['params'].items():
        print(f"{param:.<30} {value}")
    print(f"{'Score':. <30} {best['score']:. 4f}")
    print("=" * 80)


def start_trading():
    """CLI for starting live trading."""
    parser = argparse.ArgumentParser(description='Start live trading')
    
    parser.add_argument('pairs', nargs='+', help='Pairs to trade (format: ASSET_A/ASSET_B)')
    parser.add_argument('--broker', default='alpaca', choices=['alpaca'])
    parser.add_argument('--paper', action='store_true', help='Use paper trading')
    
    args = parser.parse_args()
    
    print("⚠️  Live trading is not fully implemented in CLI mode.")
    print("Please use the dashboard or Python API for live trading.")
    print(f"\nPairs to trade: {args.pairs}")
    print(f"Broker: {args. broker}")
    print(f"Mode: {'Paper' if args.paper else 'Live'}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python -m src.cli {backtest|optimize|trade} [args]")
        sys.exit(1)