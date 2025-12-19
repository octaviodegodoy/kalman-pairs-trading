"""
ml_features.py

Machine learning enhancements for pairs trading including LSTM beta prediction,
regime detection, and advanced feature engineering. 
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class LSTMBetaPredictor:
    """
    LSTM neural network for predicting time-varying hedge ratios.
    """
    
    def __init__(
        self,
        lookback_window: int = 30,
        hidden_units: int = 50,
        dropout_rate: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32
    ):
        """
        Initialize LSTM predictor. 
        
        Args:
            lookback_window: Number of past observations to use
            hidden_units: Number of LSTM units
            dropout_rate:  Dropout rate for regularization
            epochs: Training epochs
            batch_size:  Batch size for training
        """
        self.lookback_window = lookback_window
        self. hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self. scaler_x = None
        self.scaler_y = None
        
    def build_model(self, input_shape: Tuple):
        """Build LSTM model architecture."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
        except ImportError:
            raise ImportError("TensorFlow not installed.  Install with: pip install tensorflow")
        
        model = Sequential([
            LSTM(self.hidden_units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout_rate),
            LSTM(self.hidden_units // 2, return_sequences=False),
            Dropout(self. dropout_rate),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        self.model = model
        return model
    
    def prepare_data(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]: 
        """
        Prepare time series data for LSTM training.
        
        Args:
            prices_a: Price series for asset A
            prices_b: Price series for asset B
            
        Returns:
            X: Input sequences
            y: Target hedge ratios
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LinearRegression
        
        # Align series
        df = pd.DataFrame({'a': prices_a, 'b': prices_b}).dropna()
        
        # Calculate rolling hedge ratios as targets
        window = 20
        hedge_ratios = []
        
        for i in range(window, len(df)):
            window_data = df.iloc[i-window:i]
            X_window = window_data['b'].values. reshape(-1, 1)
            y_window = window_data['a'].values
            
            model = LinearRegression()
            model.fit(X_window, y_window)
            hedge_ratios.append(model. coef_[0])
        
        # Create sequences
        returns_a = df['a'].pct_change().fillna(0).values
        returns_b = df['b'].pct_change().fillna(0).values
        
        # Combine features
        features = np.column_stack([returns_a, returns_b])
        features = features[window:]  # Align with hedge_ratios
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(self.lookback_window, len(features)):
            X.append(features[i-self.lookback_window:i])
            y.append(hedge_ratios[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Normalize
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Reshape for scaling
        n_samples, n_timesteps, n_features = X. shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler_x.fit_transform(X_reshaped)
        X = X_scaled.reshape(n_samples, n_timesteps, n_features)
        
        y = self.scaler_y.fit_transform(y. reshape(-1, 1)).flatten()
        
        return X, y
    
    def train(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        validation_split: float = 0.2,
        verbose: int = 1
    ):
        """
        Train LSTM model. 
        
        Args:
            prices_a: Price series for asset A
            prices_b: Price series for asset B
            validation_split: Fraction of data for validation
            verbose:  Verbosity level
        """
        from tensorflow.keras. callbacks import EarlyStopping, ReduceLROnPlateau
        
        # Prepare data
        X, y = self.prepare_data(prices_a, prices_b)
        
        # Build model
        input_shape = (X.shape[1], X.shape[2])
        self.build_model(input_shape)
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        
        # Train
        history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )
        
        return history
    
    def predict(
        self,
        recent_prices_a: pd.Series,
        recent_prices_b: pd. Series
    ) -> float:
        """
        Predict next hedge ratio.
        
        Args:
            recent_prices_a: Recent prices for asset A
            recent_prices_b: Recent prices for asset B
            
        Returns:
            Predicted hedge ratio
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare input
        df = pd.DataFrame({'a': recent_prices_a, 'b': recent_prices_b}).dropna()
        
        returns_a = df['a'].pct_change().fillna(0).values
        returns_b = df['b'].pct_change().fillna(0).values
        
        features = np.column_stack([returns_a, returns_b])
        
        # Take last lookback_window observations
        if len(features) < self.lookback_window:
            raise ValueError(f"Need at least {self.lookback_window} observations")
        
        X = features[-self.lookback_window:].reshape(1, self.lookback_window, 2)
        
        # Scale
        X_reshaped = X.reshape(-1, 2)
        X_scaled = self. scaler_x.transform(X_reshaped)
        X = X_scaled.reshape(1, self.lookback_window, 2)
        
        # Predict
        y_pred_scaled = self.model.predict(X, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled. reshape(-1, 1))[0, 0]
        
        return y_pred


class RegimeDetector:
    """
    Hidden Markov Model for market regime detection.
    
    Detects different market regimes (trending, mean-reverting, volatile)
    to adjust trading strategy accordingly.
    """
    
    def __init__(self, n_regimes: int = 3):
        """
        Initialize regime detector. 
        
        Args:
            n_regimes: Number of market regimes to detect
        """
        self.n_regimes = n_regimes
        self. model = None
        
    def fit(
        self,
        spread: pd.Series,
        n_iter: int = 100
    ):
        """
        Fit Hidden Markov Model to spread data.
        
        Args:
            spread: Spread time series
            n_iter: Number of EM iterations
        """
        try:
            from hmmlearn import hmm
        except ImportError: 
            raise ImportError("hmmlearn not installed. Install with: pip install hmmlearn")
        
        # Prepare features:  spread returns and volatility
        spread_returns = spread.pct_change().fillna(0)
        spread_volatility = spread_returns.rolling(20).std().fillna(0)
        
        X = np.column_stack([
            spread_returns.values,
            spread_volatility.values
        ])
        
        # Fit Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type='full',
            n_iter=n_iter,
            random_state=42
        )
        
        self.model.fit(X)
        
        # Store feature statistics for interpretation
        self.regime_means = self.model.means_
        self.regime_covs = self.model.covars_
        
    def predict_regime(
        self,
        spread: pd.Series,
        return_probs: bool = False
    ):
        """
        Predict current market regime.
        
        Args:
            spread: Spread time series
            return_probs:  Whether to return regime probabilities
            
        Returns: 
            Predicted regime (and probabilities if requested)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prepare features
        spread_returns = spread.pct_change().fillna(0)
        spread_volatility = spread_returns.rolling(20).std().fillna(0)
        
        X = np.column_stack([
            spread_returns.values,
            spread_volatility. values
        ])
        
        # Predict
        regimes = self.model.predict(X)
        
        if return_probs:
            probs = self.model.predict_proba(X)
            return regimes[-1], probs[-1]
        
        return regimes[-1]
    
    def interpret_regimes(self) -> Dict:
        """
        Interpret detected regimes based on their characteristics.
        
        Returns:
            Dictionary with regime interpretations
        """
        if self.model is None:
            raise ValueError("Model not fitted.")
        
        interpretations = {}
        
        for i in range(self.n_regimes):
            mean_return = self.regime_means[i, 0]
            mean_vol = self.regime_means[i, 1]
            
            # Classify regime
            if abs(mean_return) < 0.001 and mean_vol < 0.01:
                regime_type = "STABLE_MEAN_REVERTING"
            elif abs(mean_return) > 0.01: 
                regime_type = "TRENDING"
            elif mean_vol > 0.02:
                regime_type = "HIGH_VOLATILITY"
            else:
                regime_type = "NORMAL"
            
            interpretations[i] = {
                'type':  regime_type,
                'mean_return': mean_return,
                'mean_volatility': mean_vol,
                'recommended_action': self._get_regime_recommendation(regime_type)
            }
        
        return interpretations
    
    def _get_regime_recommendation(self, regime_type: str) -> str:
        """Get trading recommendation for regime."""
        recommendations = {
            "STABLE_MEAN_REVERTING": "INCREASE_POSITION_SIZE",
            "TRENDING":  "REDUCE_POSITION_SIZE",
            "HIGH_VOLATILITY": "REDUCE_POSITION_SIZE",
            "NORMAL": "MAINTAIN_POSITION_SIZE"
        }
        return recommendations.get(regime_type, "MAINTAIN_POSITION_SIZE")


class FeatureEngineer: 
    """
    Advanced feature engineering for pairs trading.
    """
    
    @staticmethod
    def calculate_technical_features(
        prices_a: pd.Series,
        prices_b: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate technical indicators as features.
        
        Args:
            prices_a: Price series for asset A
            prices_b: Price series for asset B
            
        Returns:
            DataFrame with features
        """
        df = pd.DataFrame(index=prices_a.index)
        
        # Price ratios
        df['price_ratio'] = prices_a / prices_b
        df['log_price_ratio'] = np. log(prices_a / prices_b)
        
        # Moving averages
        for window in [5, 10, 20, 50]: 
            df[f'ratio_ma_{window}'] = df['price_ratio'].rolling(window).mean()
            df[f'ratio_std_{window}'] = df['price_ratio'].rolling(window).std()
        
        # RSI-like indicator for ratio
        delta = df['price_ratio'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['ratio_rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands for ratio
        df['ratio_bb_upper'] = df['ratio_ma_20'] + 2 * df['ratio_std_20']
        df['ratio_bb_lower'] = df['ratio_ma_20'] - 2 * df['ratio_std_20']
        df['ratio_bb_position'] = (df['price_ratio'] - df['ratio_bb_lower']) / (df['ratio_bb_upper'] - df['ratio_bb_lower'])
        
        # Momentum
        df['ratio_momentum_5'] = df['price_ratio']. pct_change(5)
        df['ratio_momentum_10'] = df['price_ratio'].pct_change(10)
        
        # Volatility
        df['ratio_volatility'] = df['price_ratio'].pct_change().rolling(20).std()
        
        # Correlation (rolling)
        df['rolling_correlation'] = prices_a.rolling(30).corr(prices_b)
        
        return df. fillna(method='bfill').fillna(0)
    
    @staticmethod
    def calculate_spread_features(spread: pd.Series) -> pd.DataFrame:
        """
        Calculate features from spread series.
        
        Args:
            spread: Spread time series
            
        Returns:
            DataFrame with spread features
        """
        df = pd.DataFrame(index=spread. index)
        
        # Z-score at different windows
        for window in [10, 20, 30, 60]: 
            mean = spread.rolling(window).mean()
            std = spread. rolling(window).std()
            df[f'zscore_{window}'] = (spread - mean) / std
        
        # Spread velocity and acceleration
        df['spread_velocity'] = spread.diff()
        df['spread_acceleration'] = df['spread_velocity'].diff()
        
        # Distance from moving averages
        for window in [10, 20, 50]:
            ma = spread.rolling(window).mean()
            df[f'distance_from_ma_{window}'] = spread - ma
        
        # Autocorrelation
        for lag in [1, 5, 10]:
            df[f'autocorr_lag_{lag}'] = spread. rolling(30).apply(
                lambda x: x.autocorr(lag=lag), raw=False
            )
        
        return df.fillna(method='bfill').fillna(0)


class EnsemblePredictor:
    """
    Ensemble model combining Kalman Filter with machine learning.
    """
    
    def __init__(self):
        """Initialize ensemble predictor."""
        self.kalman_weight = 0.5
        self.lstm_weight = 0.3
        self. ridge_weight = 0.2
        
        self.lstm_predictor = None
        self.ridge_model = None
        
    def train(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        kalman_betas: pd.Series
    ):
        """
        Train ensemble models.
        
        Args:
            prices_a: Price series for asset A
            prices_b:  Price series for asset B
            kalman_betas: Historical Kalman Filter beta estimates
        """
        from sklearn.linear_model import Ridge
        
        # Train LSTM
        self.lstm_predictor = LSTMBetaPredictor(lookback_window=30)
        self.lstm_predictor.train(prices_a, prices_b, verbose=0)
        
        # Train Ridge regression on features
        feature_eng = FeatureEngineer()
        features = feature_eng.calculate_technical_features(prices_a, prices_b)
        
        # Align with kalman betas
        aligned_data = pd.concat([features, kalman_betas], axis=1).dropna()
        X = aligned_data.iloc[:, :-1]. values
        y = aligned_data.iloc[:, -1].values
        
        self.ridge_model = Ridge(alpha=1.0)
        self.ridge_model.fit(X, y)
        
    def predict(
        self,
        recent_prices_a: pd.Series,
        recent_prices_b: pd.Series,
        kalman_beta: float
    ) -> float:
        """
        Predict hedge ratio using ensemble.
        
        Args:
            recent_prices_a: Recent prices for asset A
            recent_prices_b: Recent prices for asset B
            kalman_beta: Current Kalman Filter beta
            
        Returns:
            Ensemble prediction
        """
        # Kalman prediction
        kalman_pred = kalman_beta
        
        # LSTM prediction
        try:
            lstm_pred = self.lstm_predictor.predict(recent_prices_a, recent_prices_b)
        except:
            lstm_pred = kalman_pred
        
        # Ridge prediction
        try:
            feature_eng = FeatureEngineer()
            features = feature_eng.calculate_technical_features(recent_prices_a, recent_prices_b)
            X = features.iloc[-1:]. values
            ridge_pred = self.ridge_model.predict(X)[0]
        except:
            ridge_pred = kalman_pred
        
        # Ensemble
        ensemble_pred = (
            self.kalman_weight * kalman_pred +
            self.lstm_weight * lstm_pred +
            self.ridge_weight * ridge_pred
        )
        
        return ensemble_pred


# Example usage
if __name__ == "__main__":
    import yfinance as yf
    
    # Download sample data
    print("Downloading data...")
    gld = yf.download('GLD', start='2020-01-01', end='2024-01-01', progress=False)['Adj Close']
    gdx = yf.download('GDX', start='2020-01-01', end='2024-01-01', progress=False)['Adj Close']
    
    # LSTM Beta Prediction
    print("\n1. Training LSTM Beta Predictor...")
    lstm = LSTMBetaPredictor(lookback_window=30, epochs=50)
    lstm.train(gld, gdx, verbose=1)
    
    # Predict
    recent_gld = gld.iloc[-50:]
    recent_gdx = gdx.iloc[-50:]
    predicted_beta = lstm.predict(recent_gld, recent_gdx)
    print(f"Predicted Beta: {predicted_beta:.4f}")
    
    # Regime Detection
    print("\n2. Detecting Market Regimes...")
    from sklearn.linear_model import LinearRegression
    
    # Calculate spread
    df = pd.DataFrame({'gld': gld, 'gdx': gdx}).dropna()
    model = LinearRegression()
    model.fit(df['gdx'].values.reshape(-1, 1), df['gld'].values)
    spread = df['gld'].values - model.predict(df['gdx'].values.reshape(-1, 1))
    spread_series = pd.Series(spread, index=df.index)
    
    detector = RegimeDetector(n_regimes=3)
    detector.fit(spread_series)
    
    current_regime = detector.predict_regime(spread_series)
    interpretations = detector.interpret_regimes()
    
    print(f"Current Regime: {current_regime}")
    print("Regime Interpretations:")
    for regime_id, info in interpretations.items():
        print(f"  Regime {regime_id}: {info['type']} - {info['recommended_action']}")
    
    # Feature Engineering
    print("\n3. Engineering Features...")
    feature_eng = FeatureEngineer()
    technical_features = feature_eng.calculate_technical_features(gld, gdx)
    spread_features = feature_eng.calculate_spread_features(spread_series)
    
    print(f"Technical Features: {technical_features.shape}")
    print(f"Spread Features: {spread_features.shape}")
    print("\nSample Technical Features:")
    print(technical_features.tail())