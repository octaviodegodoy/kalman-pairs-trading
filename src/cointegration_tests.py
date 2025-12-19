"""
cointegration_tests.py

Comprehensive statistical tests for pairs trading including cointegration,
stationarity, and mean reversion tests.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from statsmodels.tsa.stattools import adfuller, coint, kpss
from statsmodels.tsa. vector_ar. vecm import coint_johansen
from arch.unitroot import PhillipsPerron, VarianceRatio
import warnings
warnings.filterwarnings('ignore')


class CointegrationTester:
    """
    Statistical testing suite for pairs trading. 
    
    Includes: 
    - Augmented Dickey-Fuller (ADF) test
    - Engle-Granger cointegration test
    - Johansen cointegration test
    - Phillips-Perron test
    - KPSS stationarity test
    - Variance Ratio test
    - Half-life calculation
    - Hurst exponent
    """
    
    def __init__(self):
        """Initialize cointegration tester."""
        self.results = {}
    
    def adf_test(
        self,
        series: pd.Series,
        regression:  str = 'c',
        maxlag: Optional[int] = None
    ) -> Dict: 
        """
        Augmented Dickey-Fuller test for stationarity.
        
        H0: Series has a unit root (non-stationary)
        H1: Series is stationary
        
        Args: 
            series: Time series to test
            regression: Constant and trend order ('c', 'ct', 'ctt', 'n')
            maxlag: Maximum lag to use
            
        Returns:
            Dictionary with test results
        """
        result = adfuller(series. dropna(), regression=regression, maxlag=maxlag)
        
        return {
            'test_statistic': result[0],
            'pvalue': result[1],
            'used_lag': result[2],
            'nobs': result[3],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    
    def engle_granger_test(
        self,
        y: pd.Series,
        x: pd.Series,
        trend: str = 'c'
    ) -> Dict:
        """
        Engle-Granger two-step cointegration test.
        
        H0: No cointegration
        H1: Series are cointegrated
        
        Args:
            y: Dependent variable
            x: Independent variable
            trend: Trend specification
            
        Returns:
            Dictionary with test results
        """
        # Align series
        df = pd.DataFrame({'y': y, 'x':  x}).dropna()
        y_clean = df['y']
        x_clean = df['x']
        
        # Perform cointegration test
        score, pvalue, crit_value = coint(y_clean, x_clean, trend=trend)
        
        # Calculate hedge ratio
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(x_clean. values.reshape(-1, 1), y_clean.values)
        hedge_ratio = model.coef_[0]
        
        # Calculate spread
        spread = y_clean. values - hedge_ratio * x_clean.values
        
        # Test spread for stationarity
        spread_adf = self.adf_test(pd.Series(spread))
        
        return {
            'test_statistic': score,
            'pvalue': pvalue,
            'critical_value': crit_value,
            'is_cointegrated': pvalue < 0.05,
            'hedge_ratio': hedge_ratio,
            'spread_adf_pvalue': spread_adf['pvalue'],
            'spread_is_stationary': spread_adf['is_stationary']
        }
    
    def johansen_test(
        self,
        data: pd.DataFrame,
        det_order: int = 0,
        k_ar_diff: int = 1
    ) -> Dict:
        """
        Johansen cointegration test for multiple time series.
        
        Args:
            data: DataFrame with multiple time series
            det_order:  Deterministic term order (-1, 0, 1)
            k_ar_diff:  Number of lagged differences
            
        Returns:
            Dictionary with test results
        """
        result = coint_johansen(data.dropna(), det_order=det_order, k_ar_diff=k_ar_diff)
        
        # Extract results
        trace_stats = result. lr1
        max_eig_stats = result.lr2
        crit_values_90 = result.cvt[: , 0]
        crit_values_95 = result.cvt[:, 1]
        crit_values_99 = result.cvt[:, 2]
        
        # Determine number of cointegrating relationships at 95% confidence
        n_coint = sum(trace_stats > crit_values_95)
        
        return {
            'trace_statistics': trace_stats. tolist(),
            'max_eigenvalue_statistics': max_eig_stats.tolist(),
            'critical_values_90': crit_values_90.tolist(),
            'critical_values_95':  crit_values_95.tolist(),
            'critical_values_99': crit_values_99.tolist(),
            'n_cointegrating_relationships': n_coint,
            'eigenvectors': result.evec.tolist()
        }
    
    def phillips_perron_test(
        self,
        series: pd.Series,
        regression: str = 'c',
        lags: Optional[int] = None
    ) -> Dict:
        """
        Phillips-Perron unit root test.
        
        More robust to serial correlation than ADF. 
        
        Args:
            series: Time series to test
            regression: Regression type
            lags: Number of lags
            
        Returns:
            Dictionary with test results
        """
        pp = PhillipsPerron(series.dropna(), trend=regression, lags=lags)
        
        return {
            'test_statistic': pp.stat,
            'pvalue': pp.pvalue,
            'critical_values': pp.critical_values,
            'is_stationary': pp.pvalue < 0.05
        }
    
    def kpss_test(
        self,
        series: pd.Series,
        regression: str = 'c',
        nlags: str = 'auto'
    ) -> Dict:
        """
        KPSS test for stationarity.
        
        H0: Series is stationary (opposite of ADF!)
        H1: Series has a unit root
        
        Args: 
            series: Time series to test
            regression: 'c' for level, 'ct' for trend
            nlags: Number of lags
            
        Returns:
            Dictionary with test results
        """
        result = kpss(series.dropna(), regression=regression, nlags=nlags)
        
        return {
            'test_statistic': result[0],
            'pvalue': result[1],
            'lags_used': result[2],
            'critical_values': result[3],
            'is_stationary': result[1] > 0.05  # Note: inverted logic
        }
    
    def variance_ratio_test(
        self,
        series: pd. Series,
        lags: int = 2,
        trend: str = 'c'
    ) -> Dict:
        """
        Variance Ratio test for random walk hypothesis.
        
        VR = 1: Random walk
        VR > 1: Positive autocorrelation (trending)
        VR < 1: Negative autocorrelation (mean reverting)
        
        Args:
            series: Time series to test
            lags:  Lag parameter
            trend: Trend specification
            
        Returns:
            Dictionary with test results
        """
        vr = VarianceRatio(series.dropna(), lags=lags, trend=trend)
        
        return {
            'variance_ratio': vr.vr,
            'test_statistic': vr.stat,
            'pvalue': vr.pvalue,
            'critical_values': vr.critical_values,
            'is_mean_reverting': vr.vr < 1 and vr.pvalue < 0.05
        }
    
    def calculate_half_life(
        self,
        spread: pd.Series
    ) -> Dict:
        """
        Calculate half-life of mean reversion using Ornstein-Uhlenbeck process.
        
        Args:
            spread: Spread time series
            
        Returns: 
            Dictionary with half-life information
        """
        from sklearn.linear_model import LinearRegression
        
        spread_clean = spread.dropna()
        
        # Fit AR(1) model: Δspread(t) = λ * spread(t-1) + ε
        spread_lag = spread_clean.values[:-1]
        spread_diff = np.diff(spread_clean.values)
        
        model = LinearRegression()
        model.fit(spread_lag. reshape(-1, 1), spread_diff)
        
        lambda_param = model.coef_[0]
        
        if lambda_param >= 0:
            half_life = np.inf
            is_mean_reverting = False
        else:
            half_life = -np.log(2) / lambda_param
            is_mean_reverting = True
        
        return {
            'half_life': half_life,
            'lambda':  lambda_param,
            'is_mean_reverting': is_mean_reverting,
            'r_squared': model.score(spread_lag.reshape(-1, 1), spread_diff)
        }
    
    def calculate_hurst_exponent(
        self,
        series: pd.Series,
        max_lag: int = 100
    ) -> Dict:
        """
        Calculate Hurst exponent using R/S analysis.
        
        H < 0.5: Mean reverting (anti-persistent)
        H = 0.5: Random walk (Brownian motion)
        H > 0.5: Trending (persistent)
        
        Args: 
            series: Time series
            max_lag: Maximum lag to consider
            
        Returns:
            Dictionary with Hurst exponent information
        """
        series_clean = series.dropna().values
        
        lags = range(2, min(max_lag, len(series_clean) // 2))
        tau = []
        
        for lag in lags:
            # Calculate standard deviation of differences at this lag
            std_dev = np. std(np.subtract(series_clean[lag:], series_clean[:-lag]))
            tau.append(std_dev)
        
        # Log-log regression
        log_lags = np.log(list(lags))
        log_tau = np.log(tau)
        
        poly = np.polyfit(log_lags, log_tau, 1)
        hurst = poly[0]
        
        # Determine regime
        if hurst < 0.5:
            regime = 'mean_reverting'
        elif hurst > 0.5:
            regime = 'trending'
        else:
            regime = 'random_walk'
        
        return {
            'hurst_exponent': hurst,
            'regime': regime,
            'is_mean_reverting': hurst < 0.5,
            'r_squared': 1 - (np.sum((log_tau - np.polyval(poly, log_lags)) ** 2) / 
                             np.sum((log_tau - np.mean(log_tau)) ** 2))
        }
    
    def run_all_tests(
        self,
        asset_a: str,
        asset_b: str,
        prices_a: pd.Series,
        prices_b: pd.Series
    ) -> Dict:
        """
        Run complete test suite on a pair. 
        
        Args:
            asset_a, asset_b: Asset identifiers
            prices_a, prices_b: Price series
            
        Returns:
            Dictionary with all test results
        """
        # Align series
        df = pd.DataFrame({'a': prices_a, 'b': prices_b}).dropna()
        prices_a_clean = df['a']
        prices_b_clean = df['b']
        
        results = {
            'pair': f"{asset_a}/{asset_b}",
            'n_observations': len(df)
        }
        
        # 1. Individual stationarity tests
        print(f"Testing {asset_a}...")
        results['asset_a_adf'] = self.adf_test(prices_a_clean)
        results['asset_a_kpss'] = self.kpss_test(prices_a_clean)
        
        print(f"Testing {asset_b}...")
        results['asset_b_adf'] = self.adf_test(prices_b_clean)
        results['asset_b_kpss'] = self.kpss_test(prices_b_clean)
        
        # 2. Cointegration tests
        print("Testing cointegration...")
        results['engle_granger'] = self. engle_granger_test(prices_a_clean, prices_b_clean)
        
        # Johansen test
        johansen_data = pd.DataFrame({'a': prices_a_clean, 'b': prices_b_clean})
        results['johansen'] = self.johansen_test(johansen_data)
        
        # 3. Calculate spread
        hedge_ratio = results['engle_granger']['hedge_ratio']
        spread = prices_a_clean. values - hedge_ratio * prices_b_clean.values
        spread_series = pd.Series(spread, index=prices_a_clean. index)
        
        # 4. Spread analysis
        print("Analyzing spread...")
        results['spread_adf'] = self.adf_test(spread_series)
        results['spread_pp'] = self.phillips_perron_test(spread_series)
        results['spread_kpss'] = self.kpss_test(spread_series)
        results['spread_variance_ratio'] = self.variance_ratio_test(spread_series)
        
        # 5. Mean reversion properties
        print("Calculating mean reversion properties...")
        results['half_life'] = self.calculate_half_life(spread_series)
        results['hurst_exponent'] = self.calculate_hurst_exponent(spread_series)
        
        # 6. Correlation
        results['correlation'] = prices_a_clean.corr(prices_b_clean)
        
        # 7. Overall assessment
        results['is_suitable_for_pairs_trading'] = self._assess_suitability(results)
        
        self.results = results
        return results
    
    def _assess_suitability(self, results: Dict) -> Dict:
        """
        Assess overall suitability for pairs trading. 
        
        Args:
            results: Dictionary with all test results
            
        Returns: 
            Assessment dictionary
        """
        checks = {
            'prices_non_stationary': (
                not results['asset_a_adf']['is_stationary'] and 
                not results['asset_b_adf']['is_stationary']
            ),
            'cointegrated':  results['engle_granger']['is_cointegrated'],
            'spread_stationary': results['spread_adf']['is_stationary'],
            'mean_reverting': results['hurst_exponent']['is_mean_reverting'],
            'reasonable_half_life': 0 < results['half_life']['half_life'] < 100,
            'sufficient_correlation': abs(results['correlation']) > 0.5
        }
        
        n_passed = sum(checks.values())
        suitable = n_passed >= 4  # Require at least 4 out of 6 checks
        
        return {
            'is_suitable': suitable,
            'checks_passed': n_passed,
            'total_checks': len(checks),
            'details': checks,
            'recommendation': self._get_recommendation(suitable, checks)
        }
    
    def _get_recommendation(self, suitable: bool, checks: Dict) -> str:
        """Generate trading recommendation."""
        if suitable:
            return "RECOMMENDED:  Pair meets criteria for pairs trading"
        else: 
            failed_checks = [k for k, v in checks.items() if not v]
            return f"NOT RECOMMENDED: Failed checks: {', '.join(failed_checks)}"
    
    def print_results(self):
        """Print formatted test results."""
        if not self.results:
            print("No results to display.  Run tests first.")
            return
        
        print("\n" + "=" * 100)
        print(f"COINTEGRATION TEST RESULTS:  {self.results['pair']}")
        print("=" * 100)
        
        # Price stationarity
        print("\n1.  PRICE STATIONARITY TESTS")
        print("-" * 100)
        for asset in ['a', 'b']: 
            adf = self.results[f'asset_{asset}_adf']
            kpss = self.results[f'asset_{asset}_kpss']
            print(f"Asset {asset. upper()}:")
            print(f"  ADF Test:   Statistic={adf['test_statistic']:.4f}, "
                  f"p-value={adf['pvalue']:.4f}, Stationary={adf['is_stationary']}")
            print(f"  KPSS Test:  Statistic={kpss['test_statistic']:.4f}, "
                  f"p-value={kpss['pvalue']:.4f}, Stationary={kpss['is_stationary']}")
        
        # Cointegration
        print("\n2. COINTEGRATION TESTS")
        print("-" * 100)
        eg = self.results['engle_granger']
        print(f"Engle-Granger Test:")
        print(f"  Test Statistic: {eg['test_statistic']:.4f}")
        print(f"  P-value: {eg['pvalue']:.4f}")
        print(f"  Cointegrated: {eg['is_cointegrated']}")
        print(f"  Hedge Ratio: {eg['hedge_ratio']:.4f}")
        
        johansen = self.results['johansen']
        print(f"\nJohansen Test:")
        print(f"  Cointegrating Relationships: {johansen['n_cointegrating_relationships']}")
        
        # Spread analysis
        print("\n3. SPREAD ANALYSIS")
        print("-" * 100)
        spread_adf = self.results['spread_adf']
        spread_vr = self.results['spread_variance_ratio']
        print(f"ADF Test:  p-value={spread_adf['pvalue']:.4f}, "
              f"Stationary={spread_adf['is_stationary']}")
        print(f"Variance Ratio: {spread_vr['variance_ratio']:.4f}, "
              f"Mean Reverting={spread_vr['is_mean_reverting']}")
        
        # Mean reversion
        print("\n4. MEAN REVERSION PROPERTIES")
        print("-" * 100)
        hl = self.results['half_life']
        hurst = self.results['hurst_exponent']
        print(f"Half-Life: {hl['half_life']:.2f} days")
        print(f"Hurst Exponent: {hurst['hurst_exponent']:.4f} ({hurst['regime']})")
        
        # Correlation
        print("\n5. CORRELATION")
        print("-" * 100)
        print(f"Pearson Correlation: {self.results['correlation']:.4f}")
        
        # Overall assessment
        print("\n6. OVERALL ASSESSMENT")
        print("-" * 100)
        assessment = self. results['is_suitable_for_pairs_trading']
        print(f"Suitable for Trading: {assessment['is_suitable']}")
        print(f"Checks Passed: {assessment['checks_passed']}/{assessment['total_checks']}")
        print(f"Recommendation: {assessment['recommendation']}")
        
        print("\nDetailed Checks:")
        for check, passed in assessment['details'].items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check. replace('_', ' ').title()}")
        
        print("=" * 100)


# Example usage
if __name__ == "__main__":
    import yfinance as yf
    
    # Download sample data
    print("Downloading sample data...")
    gld = yf.download('GLD', start='2020-01-01', end='2024-01-01', progress=False)['Adj Close']
    gdx = yf.download('GDX', start='2020-01-01', end='2024-01-01', progress=False)['Adj Close']
    
    # Run tests
    tester = CointegrationTester()
    results = tester.run_all_tests('GLD', 'GDX', gld, gdx)
    
    # Display results
    tester.print_results()