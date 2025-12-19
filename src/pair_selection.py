"""
pair_selection.py

Automated pair selection using cointegration tests and other statistical measures.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import warnings
from statsmodels.tsa.stattools import adfuller, coint
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')


@dataclass
class PairScore:
    """Score for a trading pair."""
    asset_a: str
    asset_b: str
    cointegration_pvalue: float
    correlation:  float
    half_life: float
    hurst_exponent: float
    total_score: float
    
    def __repr__(self):
        return (f"PairScore({self.asset_a}/{self.asset_b}:  "
                f"score={self.total_score:.4f}, "
                f"pval={self.cointegration_pvalue:.4f})")


class PairSelector:
    """
    Automated pair selection using statistical tests.
    
    Evaluates pairs based on:
    - Cointegration (Engle-Granger test)
    - Correlation
    - Half-life of mean reversion
    - Hurst exponent
    - Distance measures
    """
    
    def __init__(
        self,
        min_correlation: float = 0.5,
        max_correlation: float = 0.95,
        max_cointegration_pvalue: float = 0.05,
        max_half_life: float = 60,
        min_hurst:  float = 0.0,
        max_hurst:  float = 0.5
    ):
        """
        Initialize pair selector.
        
        Args:
            min_correlation: Minimum acceptable correlation
            max_correlation: Maximum acceptable correlation
            max_cointegration_pvalue: Maximum p-value for cointegration test
            max_half_life: Maximum half-life (days) for mean reversion
            min_hurst: Minimum Hurst exponent
            max_hurst: Maximum Hurst exponent
        """
        self.min_correlation = min_correlation
        self.max_correlation = max_correlation
        self.max_cointegration_pvalue = max_cointegration_pvalue
        self.max_half_life = max_half_life
        self. min_hurst = min_hurst
        self.max_hurst = max_hurst
        
        self.pair_scores: List[PairScore] = []
        
    def scan_universe(
        self,
        tickers: List[str],
        price_data: Dict[str, pd.Series] = None,
        start_date: str = None,
        end_date: str = None,
        top_n: int = 10
    ) -> List[PairScore]:
        """
        Scan universe of assets for tradeable pairs.
        
        Args:
            tickers: List of asset tickers
            price_data:  Dictionary of price series (optional, will download if None)
            start_date: Start date for data
            end_date: End date for data
            top_n: Number of top pairs to return
            
        Returns:
            List of top-scoring pairs
        """
        # Download data if not provided
        if price_data is None:
            price_data = self._download_prices(tickers, start_date, end_date)
        
        # Generate all possible pairs
        pair_combinations = list(combinations(tickers, 2))
        
        print(f"Scanning {len(pair_combinations)} pairs...")
        
        self.pair_scores = []
        
        for i, (asset_a, asset_b) in enumerate(pair_combinations):
            if (i + 1) % 50 == 0:
                print(f"Progress: {i+1}/{len(pair_combinations)} pairs evaluated")
            
            try:
                score = self._evaluate_pair(
                    asset_a, asset_b,
                    price_data[asset_a],
                    price_data[asset_b]
                )
                
                if score is not None:
                    self. pair_scores.append(score)
                    
            except Exception as e:
                continue
        
        # Sort by total score
        self.pair_scores.sort(key=lambda x: x. total_score, reverse=True)
        
        print(f"\nFound {len(self.pair_scores)} valid pairs")
        print(f"Returning top {min(top_n, len(self.pair_scores))} pairs")
        
        return self.pair_scores[:top_n]
    
    def _download_prices(
        self,
        tickers:  List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.Series]: 
        """Download price data for tickers."""
        try:
            import yfinance as yf
            
            print(f"Downloading data for {len(tickers)} assets...")
            
            price_data = {}
            for ticker in tickers:
                try: 
                    data = yf. download(ticker, start=start_date, end=end_date, 
                                      progress=False)['Adj Close']
                    if len(data) > 0:
                        price_data[ticker] = data
                except: 
                    continue
            
            return price_data
            
        except ImportError:
            raise ImportError("yfinance not installed. Install with: pip install yfinance")
    
    def _evaluate_pair(
        self,
        asset_a: str,
        asset_b: str,
        prices_a: pd.Series,
        prices_b: pd.Series
    ) -> Optional[PairScore]:
        """
        Evaluate a single pair. 
        
        Args:
            asset_a, asset_b: Asset tickers
            prices_a, prices_b: Price series
            
        Returns:
            PairScore or None if pair doesn't meet criteria
        """
        # Align series
        df = pd.DataFrame({'a': prices_a, 'b': prices_b}).dropna()
        
        if len(df) < 100:  # Need sufficient data
            return None
        
        prices_a = df['a']
        prices_b = df['b']
        
        # 1. Calculate correlation
        correlation = prices_a.corr(prices_b)
        
        if not (self.min_correlation <= abs(correlation) <= self.max_correlation):
            return None
        
        # 2. Test for cointegration
        try:
            _, pvalue, _ = coint(prices_a, prices_b)
            
            if pvalue > self.max_cointegration_pvalue:
                return None
                
        except:
            return None
        
        # 3. Calculate half-life
        try:
            half_life = self._calculate_half_life(prices_a, prices_b)
            
            if half_life > self.max_half_life or half_life <= 0:
                return None
                
        except:
            return None
        
        # 4. Calculate Hurst exponent
        try:
            # Calculate spread first
            from sklearn.linear_model import LinearRegression
            X = prices_b.values.reshape(-1, 1)
            y = prices_a.values
            model = LinearRegression()
            model.fit(X, y)
            spread = y - model.predict(X)
            
            hurst = self._calculate_hurst_exponent(spread)
            
            if not (self.min_hurst <= hurst <= self.max_hurst):
                return None
                
        except:
            hurst = 0.5
        
        # Calculate composite score
        score = self._calculate_composite_score(
            pvalue, correlation, half_life, hurst
        )
        
        return PairScore(
            asset_a=asset_a,
            asset_b=asset_b,
            cointegration_pvalue=pvalue,
            correlation=correlation,
            half_life=half_life,
            hurst_exponent=hurst,
            total_score=score
        )
    
    def _calculate_half_life(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series
    ) -> float:
        """
        Calculate half-life of mean reversion.
        
        Args:
            prices_a, prices_b: Price series
            
        Returns:
            Half-life in days
        """
        from sklearn.linear_model import LinearRegression
        
        # Calculate spread
        X = prices_b.values.reshape(-1, 1)
        y = prices_a.values
        model = LinearRegression()
        model.fit(X, y)
        spread = y - model.predict(X)
        
        # Fit AR(1) model: spread(t) - spread(t-1) = lambda * spread(t-1) + error
        spread_lag = spread[:-1]
        spread_diff = spread[1:] - spread[:-1]
        
        X_ar = spread_lag.reshape(-1, 1)
        y_ar = spread_diff
        
        ar_model = LinearRegression()
        ar_model.fit(X_ar, y_ar)
        
        lambda_param = ar_model.coef_[0]
        
        if lambda_param >= 0:
            return float('inf')
        
        half_life = -np.log(2) / lambda_param
        
        return half_life
    
    def _calculate_hurst_exponent(self, spread: np.ndarray) -> float:
        """
        Calculate Hurst exponent using R/S analysis.
        
        H < 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        
        Args:
            spread:  Spread series
            
        Returns: 
            Hurst exponent
        """
        lags = range(2, min(100, len(spread) // 2))
        tau = []
        
        for lag in lags:
            # Calculate standard deviation of differences
            std_dev = np. std(np.subtract(spread[lag:], spread[:-lag]))
            tau.append(std_dev)
        
        # Linear fit
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = poly[0]
        
        return hurst
    
    def _calculate_composite_score(
        self,
        cointegration_pvalue: float,
        correlation: float,
        half_life: float,
        hurst: float
    ) -> float:
        """
        Calculate composite score for pair.
        
        Args:
            cointegration_pvalue: P-value from cointegration test
            correlation: Correlation coefficient
            half_life: Half-life of mean reversion
            hurst: Hurst exponent
            
        Returns:
            Composite score (higher is better)
        """
        # Normalize components to 0-1 scale
        
        # Cointegration:  lower p-value is better
        coint_score = 1 - cointegration_pvalue
        
        # Correlation: closer to 0. 7-0.8 is better
        corr_score = 1 - abs(abs(correlation) - 0.75) / 0.25
        corr_score = max(0, min(1, corr_score))
        
        # Half-life: shorter is better (but not too short)
        hl_score = 1 - (half_life / self.max_half_life)
        hl_score = max(0, min(1, hl_score))
        
        # Hurst: closer to 0 is better (mean reverting)
        hurst_score = 1 - (hurst / 0.5)
        hurst_score = max(0, min(1, hurst_score))
        
        # Weighted combination
        weights = {
            'cointegration':  0.4,
            'correlation': 0.2,
            'half_life':  0.25,
            'hurst': 0.15
        }
        
        total_score = (
            weights['cointegration'] * coint_score +
            weights['correlation'] * corr_score +
            weights['half_life'] * hl_score +
            weights['hurst'] * hurst_score
        )
        
        return total_score
    
    def plot_pair_rankings(self, top_n: int = 20):
        """
        Plot top pair rankings.
        
        Args:
            top_n: Number of top pairs to display
        """
        if not self.pair_scores:
            print("No pairs evaluated yet.  Run scan_universe first.")
            return
        
        top_pairs = self.pair_scores[:top_n]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Total scores
        pair_names = [f"{p.asset_a}/{p.asset_b}" for p in top_pairs]
        scores = [p.total_score for p in top_pairs]
        
        axes[0, 0].barh(pair_names, scores, color='steelblue')
        axes[0, 0].set_xlabel('Composite Score')
        axes[0, 0].set_title('Top Pairs by Composite Score', fontweight='bold')
        axes[0, 0].invert_yaxis()
        
        # 2. Cointegration p-values
        pvalues = [p.cointegration_pvalue for p in top_pairs]
        
        axes[0, 1]. barh(pair_names, pvalues, color='coral')
        axes[0, 1].axvline(x=0.05, color='red', linestyle='--', label='p=0.05')
        axes[0, 1].set_xlabel('Cointegration P-Value')
        axes[0, 1].set_title('Cointegration Test Results', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].invert_yaxis()
        
        # 3. Half-lives
        half_lives = [p.half_life for p in top_pairs]
        
        axes[1, 0]. barh(pair_names, half_lives, color='lightgreen')
        axes[1, 0].set_xlabel('Half-Life (days)')
        axes[1, 0].set_title('Mean Reversion Speed', fontweight='bold')
        axes[1, 0].invert_yaxis()
        
        # 4. Scatter:  Half-life vs Cointegration
        half_lives_all = [p.half_life for p in self.pair_scores[: 50]]
        pvalues_all = [p. cointegration_pvalue for p in self.pair_scores[:50]]
        scores_all = [p.total_score for p in self.pair_scores[:50]]
        
        scatter = axes[1, 1]. scatter(half_lives_all, pvalues_all, 
                                    c=scores_all, cmap='RdYlGn',
                                    s=100, alpha=0.6)
        axes[1, 1].set_xlabel('Half-Life (days)')
        axes[1, 1].set_ylabel('Cointegration P-Value')
        axes[1, 1].set_title('Pair Quality Landscape', fontweight='bold')
        axes[1, 1].axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
        plt.colorbar(scatter, ax=axes[1, 1], label='Score')
        
        plt.tight_layout()
        plt.show()
    
    def print_top_pairs(self, n: int = 10):
        """
        Print top pairs with details.
        
        Args:
            n: Number of pairs to display
        """
        if not self.pair_scores:
            print("No pairs evaluated yet.")
            return
        
        print("\n" + "=" * 100)
        print(f"{'RANK':<6} {'PAIR':<15} {'SCORE':<10} {'P-VALUE':<12} "
              f"{'CORR':<10} {'HALF-LIFE':<12} {'HURST':<10}")
        print("=" * 100)
        
        for i, pair in enumerate(self.pair_scores[:n], 1):
            print(f"{i: <6} {pair.asset_a}/{pair.asset_b:<10} "
                  f"{pair.total_score:<10.4f} {pair.cointegration_pvalue:<12.4f} "
                  f"{pair.correlation:<10.4f} {pair.half_life:<12.2f} "
                  f"{pair.hurst_exponent:<10.4f}")
        
        print("=" * 100)


# Example usage
if __name__ == "__main__":
    # Example universe of ETFs
    universe = [
        'SPY', 'QQQ', 'IWM', 'DIA',  # Equity indices
        'GLD', 'SLV',  # Precious metals
        'USO', 'UNG',  # Commodities
        'TLT', 'IEF',  # Bonds
        'XLE', 'XLF', 'XLK', 'XLV'  # Sectors
    ]
    
    selector = PairSelector(
        min_correlation=0.5,
        max_cointegration_pvalue=0.05,
        max_half_life=60
    )
    
    # Scan universe
    top_pairs = selector.scan_universe(
        universe,
        start_date='2020-01-01',
        end_date='2024-01-01',
        top_n=10
    )
    
    # Display results
    selector.print_top_pairs(10)
    selector.plot_pair_rankings(10)