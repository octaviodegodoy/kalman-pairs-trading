"""
parameter_optimizer.py

Hyperparameter optimization for pairs trading strategies including
grid search, Bayesian optimization, and walk-forward analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional
from itertools import product
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
warnings.filterwarnings('ignore')


class ParameterOptimizer: 
    """
    Parameter optimization for pairs trading strategies.
    
    Supports:
    - Grid search
    - Random search
    - Bayesian optimization
    - Walk-forward analysis
    - Monte Carlo simulation
    - Sensitivity analysis
    """
    
    def __init__(
        self,
        strategy_class,
        data:  pd.DataFrame,
        objective_function: str = 'sharpe_ratio'
    ):
        """
        Initialize parameter optimizer.
        
        Args:
            strategy_class:  Strategy class to optimize
            data:  Historical data
            objective_function:  Metric to optimize ('sharpe_ratio', 'total_return', 'calmar_ratio')
        """
        self.strategy_class = strategy_class
        self.data = data
        self.objective_function = objective_function
        self.results = []
        
    def grid_search(
        self,
        param_grid: Dict[str, List],
        n_jobs: int = 1,
        verbose: bool = True
    ) -> Dict:
        """
        Exhaustive grid search over parameter space.
        
        Args:
            param_grid: Dictionary of parameter names and values to try
            n_jobs: Number of parallel jobs
            verbose: Whether to show progress
            
        Returns: 
            Dictionary with best parameters and results
        """
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        if verbose:
            print(f"Testing {len(combinations)} parameter combinations...")
        
        results = []
        
        if n_jobs > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = []
                
                for combo in combinations:
                    params = dict(zip(param_names, combo))
                    future = executor.submit(self._evaluate_params, params)
                    futures. append((future, params))
                
                for future, params in tqdm(futures, disable=not verbose):
                    try:
                        score = future.result()
                        results.append({
                            'params': params,
                            'score': score
                        })
                    except Exception as e:
                        if verbose:
                            print(f"Error with params {params}: {e}")
        else:
            # Sequential execution
            iterator = tqdm(combinations, disable=not verbose)
            for combo in iterator:
                params = dict(zip(param_names, combo))
                try:
                    score = self._evaluate_params(params)
                    results.append({
                        'params': params,
                        'score': score
                    })
                except Exception as e: 
                    if verbose:
                        print(f"Error with params {params}: {e}")
        
        # Find best parameters
        results.sort(key=lambda x: x['score'], reverse=True)
        self.results = results
        
        best_result = results[0]
        
        if verbose:
            print(f"\nBest parameters found:")
            for param, value in best_result['params'].items():
                print(f"  {param}: {value}")
            print(f"Best {self.objective_function}:  {best_result['score']:. 4f}")
        
        return best_result
    
    def random_search(
        self,
        param_distributions: Dict[str, Tuple],
        n_iter: int = 100,
        random_state: Optional[int] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Random search over parameter space.
        
        Args:
            param_distributions: Dictionary with parameter ranges (min, max)
            n_iter: Number of iterations
            random_state: Random seed
            verbose: Whether to show progress
            
        Returns:
            Dictionary with best parameters
        """
        if random_state is not None:
            np.random. seed(random_state)
        
        results = []
        
        iterator = tqdm(range(n_iter), disable=not verbose)
        for _ in iterator:
            # Sample random parameters
            params = {}
            for param_name, (min_val, max_val) in param_distributions. items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = np.random. randint(min_val, max_val + 1)
                else:
                    params[param_name] = np.random. uniform(min_val, max_val)
            
            try: 
                score = self._evaluate_params(params)
                results. append({
                    'params':  params,
                    'score':  score
                })
            except Exception as e:
                if verbose: 
                    print(f"Error with params {params}: {e}")
        
        results.sort(key=lambda x: x['score'], reverse=True)
        self.results = results
        
        best_result = results[0]
        
        if verbose:
            print(f"\nBest parameters found:")
            for param, value in best_result['params'].items():
                print(f"  {param}: {value}")
            print(f"Best {self.objective_function}:  {best_result['score']:. 4f}")
        
        return best_result
    
    def bayesian_optimize(
        self,
        param_bounds: Dict[str, Tuple],
        n_iter: int = 50,
        init_points: int = 5,
        verbose: bool = True
    ) -> Dict:
        """
        Bayesian optimization using Gaussian Processes.
        
        Args:
            param_bounds: Dictionary with parameter bounds
            n_iter: Number of iterations
            init_points: Number of initial random points
            verbose: Whether to show progress
            
        Returns: 
            Dictionary with best parameters
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer
            from skopt.utils import use_named_args
        except ImportError: 
            raise ImportError("scikit-optimize not installed. Install with: pip install scikit-optimize")
        
        # Define search space
        space = []
        param_names = []
        
        for param_name, (min_val, max_val) in param_bounds.items():
            param_names.append(param_name)
            if isinstance(min_val, int) and isinstance(max_val, int):
                space.append(Integer(min_val, max_val, name=param_name))
            else:
                space.append(Real(min_val, max_val, name=param_name))
        
        # Define objective function
        @use_named_args(space)
        def objective(**params):
            try:
                score = self._evaluate_params(params)
                return -score  # Minimize negative score
            except: 
                return 1e10  # Large penalty for failed evaluations
        
        # Run optimization
        if verbose:
            print(f"Running Bayesian optimization with {n_iter} iterations...")
        
        result = gp_minimize(
            objective,
            space,
            n_calls=n_iter,
            n_initial_points=init_points,
            random_state=42,
            verbose=verbose
        )
        
        # Extract best parameters
        best_params = dict(zip(param_names, result.x))
        best_score = -result.fun
        
        best_result = {
            'params': best_params,
            'score': best_score
        }
        
        if verbose: 
            print(f"\nBest parameters found:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")
            print(f"Best {self.objective_function}: {best_score:. 4f}")
        
        return best_result
    
    def walk_forward_analysis(
        self,
        params: Dict,
        train_window: int = 252,
        test_window: int = 63,
        reoptimize: bool = True,
        param_grid: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Walk-forward analysis for out-of-sample testing.
        
        Args:
            params:  Initial parameters
            train_window: Training window size (days)
            test_window: Testing window size (days)
            reoptimize: Whether to reoptimize in each window
            param_grid: Parameter grid for reoptimization
            
        Returns:
            DataFrame with walk-forward results
        """
        results = []
        
        n = len(self.data)
        n_windows = (n - train_window) // test_window
        
        print(f"Running walk-forward analysis with {n_windows} windows...")
        
        for i in range(n_windows):
            train_start = i * test_window
            train_end = train_start + train_window
            test_start = train_end
            test_end = test_start + test_window
            
            if test_end > n:
                break
            
            train_data = self.data.iloc[train_start:train_end]
            test_data = self.data.iloc[test_start:test_end]
            
            # Optimize on training data
            if reoptimize and param_grid is not None:
                optimizer = ParameterOptimizer(
                    self.strategy_class,
                    train_data,
                    self.objective_function
                )
                best_result = optimizer.grid_search(param_grid, verbose=False)
                current_params = best_result['params']
            else:
                current_params = params
            
            # Test on out-of-sample data
            test_score = self._evaluate_params(current_params, test_data)
            
            results.append({
                'window': i + 1,
                'train_start': train_data.index[0],
                'train_end': train_data. index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'params': current_params,
                'test_score': test_score
            })
            
            print(f"Window {i+1}/{n_windows}: Test {self.objective_function} = {test_score:.4f}")
        
        results_df = pd.DataFrame(results)
        
        print(f"\nWalk-Forward Summary:")
        print(f"Average Test Score: {results_df['test_score'].mean():.4f}")
        print(f"Std Test Score: {results_df['test_score'].std():.4f}")
        print(f"Min Test Score: {results_df['test_score'].min():.4f}")
        print(f"Max Test Score: {results_df['test_score'].max():.4f}")
        
        return results_df
    
    def sensitivity_analysis(
        self,
        base_params: Dict,
        param_to_vary: str,
        param_range: np.ndarray
    ) -> pd.DataFrame:
        """
        Analyze sensitivity to a single parameter.
        
        Args:
            base_params: Base parameter set
            param_to_vary:  Parameter to vary
            param_range: Range of values to test
            
        Returns:
            DataFrame with sensitivity results
        """
        results = []
        
        print(f"Analyzing sensitivity to {param_to_vary}...")
        
        for value in tqdm(param_range):
            params = base_params.copy()
            params[param_to_vary] = value
            
            try: 
                score = self._evaluate_params(params)
                results.append({
                    param_to_vary: value,
                    'score': score
                })
            except: 
                continue
        
        results_df = pd.DataFrame(results)
        
        # Plot results
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(results_df[param_to_vary], results_df['score'], 
                linewidth=2, marker='o')
        ax.set_xlabel(param_to_vary, fontsize=12)
        ax.set_ylabel(self.objective_function, fontsize=12)
        ax.set_title(f'Sensitivity Analysis: {param_to_vary}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return results_df
    
    def monte_carlo_simulation(
        self,
        params: Dict,
        n_simulations: int = 1000,
        noise_level: float = 0.1
    ) -> Dict:
        """
        Monte Carlo simulation with parameter perturbations.
        
        Args:
            params: Base parameters
            n_simulations:  Number of simulations
            noise_level: Level of noise to add to parameters
            
        Returns: 
            Dictionary with simulation results
        """
        scores = []
        
        print(f"Running {n_simulations} Monte Carlo simulations...")
        
        for _ in tqdm(range(n_simulations)):
            # Perturb parameters
            perturbed_params = {}
            for param, value in params.items():
                if isinstance(value, (int, float)):
                    noise = np.random.normal(0, abs(value) * noise_level)
                    perturbed_params[param] = value + noise
                else: 
                    perturbed_params[param] = value
            
            try:
                score = self._evaluate_params(perturbed_params)
                scores.append(score)
            except:
                continue
        
        scores = np.array(scores)
        
        results = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'min_score': scores.min(),
            'max_score':  scores.max(),
            'percentile_5': np.percentile(scores, 5),
            'percentile_95': np.percentile(scores, 95),
            'scores': scores
        }
        
        print(f"\nMonte Carlo Results:")
        print(f"Mean Score: {results['mean_score']:.4f}")
        print(f"Std Score: {results['std_score']:.4f}")
        print(f"5th Percentile: {results['percentile_5']:.4f}")
        print(f"95th Percentile: {results['percentile_95']:.4f}")
        
        # Plot distribution
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(scores, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(results['mean_score'], color='red', linestyle='--', 
                   linewidth=2, label=f"Mean: {results['mean_score']:.4f}")
        ax.set_xlabel(self.objective_function, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Monte Carlo Simulation Results', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return results
    
    def _evaluate_params(
        self,
        params: Dict,
        data: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Evaluate parameters on data.
        
        Args:
            params: Parameters to evaluate
            data: Data to use (uses self.data if None)
            
        Returns:
            Objective function value
        """
        if data is None:
            data = self.data
        
        # Import necessary classes
        from . backtester import KalmanPairsBacktest
        
        # Create strategy with parameters
        strategy = self.strategy_class(**params)
        
        # Run backtest
        backtester = KalmanPairsBacktest(strategy)
        results = backtester.run(data)
        
        # Calculate objective
        returns = results['strategy_return']. dropna()
        
        if len(returns) == 0:
            return -np.inf
        
        if self.objective_function == 'sharpe_ratio':
            score = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        elif self.objective_function == 'total_return':
            score = results['cumulative_return'].iloc[-1] - 1
        
        elif self.objective_function == 'calmar_ratio':
            total_return = results['cumulative_return'].iloc[-1] - 1
            cumulative = results['cumulative_return']
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            score = total_return / max_drawdown if max_drawdown > 0 else 0
        
        elif self.objective_function == 'sortino_ratio':
            downside = returns[returns < 0]
            downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-10
            score = returns.mean() / downside_std * np.sqrt(252)
        
        else:
            raise ValueError(f"Unknown objective function:  {self.objective_function}")
        
        return score
    
    def plot_optimization_results(self, param_x: str, param_y: str):
        """
        Plot 2D heatmap of optimization results.
        
        Args:
            param_x: Parameter for x-axis
            param_y:  Parameter for y-axis
        """
        if not self.results:
            print("No results to plot.  Run optimization first.")
            return
        
        # Extract data
        x_values = []
        y_values = []
        scores = []
        
        for result in self.results:
            params = result['params']
            if param_x in params and param_y in params:
                x_values.append(params[param_x])
                y_values. append(params[param_y])
                scores.append(result['score'])
        
        # Create pivot table
        df = pd.DataFrame({
            param_x: x_values,
            param_y: y_values,
            'score': scores
        })
        
        pivot = df.pivot_table(values='score', index=param_y, columns=param_x)
        
        # Plot heatmap
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax)
        ax.set_title(f'Parameter Optimization:  {param_x} vs {param_y}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    from .pairs_strategy import PairsTradingStrategy
    from .data_manager import download_data
    
    # Download sample data
    df = download_data('GLD', 'GDX', '2020-01-01', '2024-01-01')
    
    # Define parameter grid
    param_grid = {
        'entry_zscore': [1.5, 2.0, 2.5],
        'exit_zscore': [0.3, 0.5, 0.7],
        'delta':  [1e-5, 1e-4, 1e-3],
        'lookback_window': [15, 20, 30]
    }
    
    # Initialize optimizer
    optimizer = ParameterOptimizer(
        PairsTradingStrategy,
        df,
        objective_function='sharpe_ratio'
    )
    
    # Run grid search
    best_params = optimizer.grid_search(param_grid)
    
    # Plot results
    optimizer.plot_optimization_results('entry_zscore', 'exit_zscore')