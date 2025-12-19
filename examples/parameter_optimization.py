"""
parameter_optimization.py

Example of parameter optimization using various methods.
"""

import sys
import os
sys.path. append(os.path. dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pairs_strategy import PairsTradingStrategy
from src.parameter_optimizer import ParameterOptimizer
from src.data_manager import download_data
import numpy as np


def main():
    """Run parameter optimization example."""
    
    print("=" * 80)
    print("PARAMETER OPTIMIZATION EXAMPLE")
    print("=" * 80)
    
    # 1. Download data
    print("\n1. Downloading data...")
    df = download_data('GLD', 'GDX', '2020-01-01', '2024-01-01')
    print(f"   Downloaded {len(df)} observations")
    
    # 2. Initialize optimizer
    print("\n2. Initializing optimizer...")
    optimizer = ParameterOptimizer(
        PairsTradingStrategy,
        df,
        objective_function='sharpe_ratio'
    )
    
    # 3. Grid Search
    print("\n3. Running Grid Search...")
    param_grid = {
        'entry_zscore': [1.5, 2.0, 2.5],
        'exit_zscore': [0.3, 0.5, 0.7],
        'delta': [1e-5, 1e-4, 1e-3],
        'lookback_window':  [15, 20, 30]
    }
    
    best_grid = optimizer.grid_search(param_grid, verbose=True)
    print("\n   Best parameters from Grid Search:")
    for param, value in best_grid['params'].items():
        print(f"   {param}: {value}")
    print(f"   Best Sharpe Ratio: {best_grid['score']:.4f}")
    
    # 4. Random Search
    print("\n4. Running Random Search...")
    param_distributions = {
        'entry_zscore': (1.5, 3.0),
        'exit_zscore': (0.2, 1.0),
        'delta': (1e-5, 1e-3),
        'lookback_window': (10, 40)
    }
    
    best_random = optimizer.random_search(
        param_distributions,
        n_iter=50,
        random_state=42,
        verbose=True
    )
    print("\n   Best parameters from Random Search:")
    for param, value in best_random['params'].items():
        print(f"   {param}: {value}")
    print(f"   Best Sharpe Ratio: {best_random['score']:.4f}")
    
    # 5. Sensitivity Analysis
    print("\n5. Running Sensitivity Analysis...")
    base_params = best_grid['params']
    
    print("\n   Analyzing sensitivity to entry_zscore...")
    sensitivity_df = optimizer.sensitivity_analysis(
        base_params,
        'entry_zscore',
        np.linspace(1.0, 3.0, 20)
    )
    
    # 6. Walk-Forward Analysis
    print("\n6. Running Walk-Forward Analysis...")
    wf_results = optimizer.walk_forward_analysis(
        params=best_grid['params'],
        train_window=252,
        test_window=63,
        reoptimize=True,
        param_grid={
            'entry_zscore':  [1.5, 2.0, 2.5],
            'exit_zscore': [0.3, 0.5, 0.7]
        }
    )
    
    print("\n   Walk-Forward Results:")
    print(wf_results[['window', 'test_score']])
    
    # 7. Monte Carlo Simulation
    print("\n7. Running Monte Carlo Simulation...")
    mc_results = optimizer. monte_carlo_simulation(
        params=best_grid['params'],
        n_simulations=500,
        noise_level=0.1
    )
    
    # 8. Plot optimization results
    print("\n8. Generating visualization...")
    optimizer.plot_optimization_results('entry_zscore', 'exit_zscore')
    
    print("\nOptimization complete!")


if __name__ == "__main__":
    main()