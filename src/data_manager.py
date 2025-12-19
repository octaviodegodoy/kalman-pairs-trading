"""
data_manager.py

Data management utilities for pairs trading including data download,
caching, cleaning, and storage.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
import os
import pickle
import warnings
warnings.filterwarnings('ignore')


class DataManager:
    """
    Comprehensive data management for pairs trading.
    """
    
    def __init__(self, cache_dir: str = './data_cache'):
        """
        Initialize data manager.
        
        Args:
            cache_dir: Directory for caching data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def download_data(
        self,
        tickers: Union[str, List[str]],
        start_date: str,
        end_date: str,
        source: str = 'yahoo',
        use_cache: bool = True
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Download price data for ticker(s).
        
        Args:
            tickers: Single ticker or list of tickers
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            source: Data source ('yahoo', 'alpha_vantage', 'quandl')
            use_cache: Whether to use cached data
            
        Returns:
            Series (single ticker) or DataFrame (multiple tickers)
        """
        is_single = isinstance(tickers, str)
        if is_single:
            tickers = [tickers]
        
        # Check cache
        if use_cache: 
            cached_data = self._load_from_cache(tickers, start_date, end_date)
            if cached_data is not None: 
                print(f"Loaded data from cache")
                return cached_data[tickers[0]] if is_single else cached_data
        
        # Download data
        print(f"Downloading data for {len(tickers)} ticker(s) from {source}...")
        
        if source == 'yahoo':
            data = self._download_yahoo(tickers, start_date, end_date)
        elif source == 'alpha_vantage': 
            data = self._download_alpha_vantage(tickers, start_date, end_date)
        elif source == 'quandl':
            data = self._download_quandl(tickers, start_date, end_date)
        else:
            raise ValueError(f"Unknown data source: {source}")
        
        # Cache data
        if use_cache: 
            self._save_to_cache(data, tickers, start_date, end_date)
        
        return data[tickers[0]] if is_single else data
    
    def _download_yahoo(
        self,
        tickers:  List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Download data from Yahoo Finance."""
        try:
            import yfinance as yf
            
            data = {}
            for ticker in tickers: 
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if len(df) > 0:
                    data[ticker] = df['Adj Close']
                else:
                    print(f"Warning: No data for {ticker}")
            
            return pd.DataFrame(data)
            
        except ImportError:
            raise ImportError("yfinance not installed. Install with: pip install yfinance")
    
    def _download_alpha_vantage(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Download data from Alpha Vantage."""
        try:
            import requests
            
            api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            if not api_key:
                raise ValueError("ALPHA_VANTAGE_API_KEY environment variable not set")
            
            data = {}
            for ticker in tickers:
                url = f"https://www.alphavantage.co/query"
                params = {
                    'function': 'TIME_SERIES_DAILY_ADJUSTED',
                    'symbol': ticker,
                    'outputsize': 'full',
                    'apikey': api_key
                }
                
                response = requests.get(url, params=params)
                result = response.json()
                
                if 'Time Series (Daily)' in result:
                    ts_data = result['Time Series (Daily)']
                    df = pd.DataFrame. from_dict(ts_data, orient='index')
                    df.index = pd.to_datetime(df.index)
                    df = df.sort_index()
                    df = df[(df.index >= start_date) & (df.index <= end_date)]
                    data[ticker] = df['5. adjusted close']. astype(float)
            
            return pd.DataFrame(data)
            
        except Exception as e:
            raise RuntimeError(f"Error downloading from Alpha Vantage: {e}")
    
    def _download_quandl(
        self,
        tickers: List[str],
        start_date:  str,
        end_date:  str
    ) -> pd.DataFrame:
        """Download data from Quandl."""
        try:
            import quandl
            
            api_key = os.getenv('QUANDL_API_KEY')
            if api_key: 
                quandl.ApiConfig.api_key = api_key
            
            data = {}
            for ticker in tickers: 
                df = quandl.get(f"WIKI/{ticker}", start_date=start_date, end_date=end_date)
                data[ticker] = df['Adj.  Close']
            
            return pd.DataFrame(data)
            
        except ImportError:
            raise ImportError("quandl not installed. Install with: pip install quandl")
    
    def _load_from_cache(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """Load data from cache."""
        cache_key = f"{'_'.join(sorted(tickers))}_{start_date}_{end_date}"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os. path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                return data
            except: 
                return None
        
        return None
    
    def _save_to_cache(
        self,
        data: pd.DataFrame,
        tickers: List[str],
        start_date: str,
        end_date: str
    ):
        """Save data to cache."""
        cache_key = f"{'_'.join(sorted(tickers))}_{start_date}_{end_date}"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Warning: Failed to cache data: {e}")
    
    def clean_data(
        self,
        data: pd.DataFrame,
        method: str = 'forward_fill',
        max_missing_pct: float = 0.05
    ) -> pd.DataFrame:
        """
        Clean price data. 
        
        Args:
            data: DataFrame with price data
            method: Method for handling missing values
            max_missing_pct: Maximum allowed missing data percentage
            
        Returns: 
            Cleaned DataFrame
        """
        # Check missing data
        missing_pct = data.isna().sum() / len(data)
        
        for col in data.columns:
            if missing_pct[col] > max_missing_pct: 
                print(f"Warning:  {col} has {missing_pct[col]*100:.2f}% missing data")
        
        # Fill missing values
        if method == 'forward_fill': 
            data = data.fillna(method='ffill')
        elif method == 'backward_fill':
            data = data.fillna(method='bfill')
        elif method == 'interpolate':
            data = data. interpolate(method='linear')
        elif method == 'drop':
            data = data.dropna()
        
        # Remove duplicates
        data = data[~data.index.duplicated(keep='first')]
        
        # Sort by index
        data = data.sort_index()
        
        return data
    
    def align_data(
        self,
        *series: pd.Series,
        method: str = 'inner'
    ) -> List[pd.Series]:
        """
        Align multiple time series. 
        
        Args:
            *series: Variable number of Series to align
            method:  Alignment method ('inner', 'outer')
            
        Returns:
            List of aligned Series
        """
        df = pd.concat(series, axis=1, join=method)
        return [df.iloc[:, i] for i in range(len(series))]
    
    def resample_data(
        self,
        data: Union[pd.Series, pd.DataFrame],
        frequency: str = 'D'
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Resample data to different frequency.
        
        Args:
            data: Time series data
            frequency: Target frequency ('D', 'W', 'M', 'H')
            
        Returns:
            Resampled data
        """
        return data.resample(frequency).last().dropna()
    
    def calculate_returns(
        self,
        prices: Union[pd.Series, pd.DataFrame],
        method: str = 'simple'
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate returns from prices.
        
        Args:
            prices: Price data
            method: 'simple' or 'log'
            
        Returns:
            Returns series/dataframe
        """
        if method == 'simple':
            return prices.pct_change()
        elif method == 'log':
            return np.log(prices / prices.shift(1))
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def save_to_csv(
        self,
        data: pd.DataFrame,
        filename: str,
        include_index: bool = True
    ):
        """Save data to CSV file."""
        filepath = os.path.join(self.cache_dir, filename)
        data.to_csv(filepath, index=include_index)
        print(f"Data saved to {filepath}")
    
    def load_from_csv(
        self,
        filename:  str,
        index_col: Optional[int] = 0,
        parse_dates:  bool = True
    ) -> pd.DataFrame:
        """Load data from CSV file."""
        filepath = os.path.join(self.cache_dir, filename)
        return pd.read_csv(filepath, index_col=index_col, parse_dates=parse_dates)


def download_data(
    ticker1: str,
    ticker2: str,
    start_date:  str,
    end_date:  str
) -> pd.DataFrame:
    """
    Convenience function to download pair data.
    
    Args:
        ticker1: First ticker
        ticker2: Second ticker
        start_date: Start date
        end_date: End date
        
    Returns:
        DataFrame with columns 'asset1' and 'asset2'
    """
    dm = DataManager()
    data = dm.download_data([ticker1, ticker2], start_date, end_date)
    data = dm.clean_data(data)
    
    df = pd.DataFrame({
        'asset1': data[ticker1],
        'asset2': data[ticker2]
    })
    
    return df. dropna()


# Example usage
if __name__ == "__main__":
    dm = DataManager()
    
    # Download single ticker
    gld = dm.download_data('GLD', '2020-01-01', '2024-01-01')
    print(f"Downloaded {len(gld)} observations for GLD")
    
    # Download multiple tickers
    data = dm.download_data(['GLD', 'GDX', 'SLV'], '2020-01-01', '2024-01-01')
    print(f"Downloaded data for {len(data. columns)} tickers")
    
    # Clean data
    cleaned = dm.clean_data(data)
    print(f"Cleaned data:  {len(cleaned)} observations")
    
    # Calculate returns
    returns = dm.calculate_returns(cleaned)
    print(f"Returns shape: {returns.shape}")
    
    # Save to CSV
    dm.save_to_csv(cleaned, 'sample_data.csv')