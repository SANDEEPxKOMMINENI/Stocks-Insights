import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import ta
from typing import List, Dict, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler

class StockDataProcessor:
    """
    Class to handle stock data downloading, preprocessing and feature engineering
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize StockDataProcessor
        
        Args:
            cache_dir: Directory to cache downloaded stock data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.scaler = StandardScaler()
        
    def download_stock_data(self, 
                           ticker: str, 
                           start_date: str = None, 
                           end_date: str = None,
                           period: str = "5y",
                           interval: str = "1d",
                           force_download: bool = False) -> pd.DataFrame:
        """
        Download stock data using yfinance
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            period: Data period (default: 5y)
            interval: Data interval (default: 1d)
            force_download: Force download even if cached data exists
            
        Returns:
            DataFrame with stock data
        """
        cache_file = os.path.join(self.cache_dir, f"{ticker}_{interval}.csv")
        
        # If cache exists and not forcing download, load from cache
        if os.path.exists(cache_file) and not force_download:
            print(f"Loading {ticker} data from cache")
            df = pd.read_csv(cache_file)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            return df
        
        print(f"Downloading {ticker} data from Yahoo Finance")
        if start_date and end_date:
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        else:
            data = yf.download(ticker, period=period, interval=interval)
            
        # Save to cache
        data.to_csv(cache_file)
        return data
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the stock data
        
        Args:
            df: DataFrame with stock data (OHLCV)
            
        Returns:
            DataFrame with added technical indicators
        """
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Trend indicators
        data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
        data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
        data['SMA_200'] = ta.trend.sma_indicator(data['Close'], window=200)
        
        # Add MACD
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Diff'] = macd.macd_diff()
        
        # Add RSI
        data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
        
        # Add Bollinger Bands
        bollinger = ta.volatility.BollingerBands(data['Close'])
        data['BB_High'] = bollinger.bollinger_hband()
        data['BB_Low'] = bollinger.bollinger_lband()
        data['BB_Mid'] = bollinger.bollinger_mavg()
        
        # Add volume indicators
        data['Volume_SMA_20'] = ta.trend.sma_indicator(data['Volume'], window=20)
        
        # Add price rate of change
        data['ROC'] = ta.momentum.ROCIndicator(data['Close']).roc()
        
        # Calculate returns
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close']).diff()
        
        # Drop rows with NaN values (caused by indicators that need lookback periods)
        data.dropna(inplace=True)
        
        return data
    
    def prepare_sequences(self, 
                          df: pd.DataFrame, 
                          input_seq_len: int = 60, 
                          forecast_horizon: int = 5,
                          target_col: str = 'Close',
                          feature_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare input/output sequences for time series forecasting
        
        Args:
            df: DataFrame with stock data and indicators
            input_seq_len: Length of input sequences
            forecast_horizon: Number of days to forecast
            target_col: Target column to predict
            feature_columns: List of columns to use as features (if None, use all numeric columns)
            
        Returns:
            Tuple of (input sequences, target sequences)
        """
        # If no feature columns provided, use all numeric columns
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
        # Make sure target_col is in feature_columns
        if target_col not in feature_columns:
            feature_columns.append(target_col)
            
        # Extract features and target
        data = df[feature_columns].values
        
        # Normalize data
        normalized_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        
        target_idx = feature_columns.index(target_col)
        
        for i in range(len(normalized_data) - input_seq_len - forecast_horizon + 1):
            X.append(normalized_data[i:i+input_seq_len])
            
            # For the target, we only extract the target column
            target_seq = normalized_data[i+input_seq_len:i+input_seq_len+forecast_horizon, target_idx:target_idx+1]
            y.append(target_seq)
            
        return np.array(X), np.array(y)
    
    def inverse_transform_predictions(self, predictions: np.ndarray, target_col: str = 'Close') -> np.ndarray:
        """
        Inverse transform normalized predictions back to original scale
        
        Args:
            predictions: Normalized predictions from the model
            target_col: Target column that was predicted
            
        Returns:
            Predictions in original scale
        """
        # Get the column index of the target
        target_idx = self.scaler.feature_names_in_.tolist().index(target_col)
        
        # Create a dummy array filled with zeros
        dummy = np.zeros((predictions.shape[0], len(self.scaler.feature_names_in_)))
        
        # Fill in the target column with our predictions
        dummy[:, target_idx:target_idx+1] = predictions
        
        # Inverse transform
        inverse_transformed = self.scaler.inverse_transform(dummy)
        
        # Return only the target column
        return inverse_transformed[:, target_idx:target_idx+1]
    
    def create_train_val_test_split(self, X: np.ndarray, y: np.ndarray, 
                                   train_split: float = 0.7, 
                                   val_split: float = 0.15) -> Dict[str, np.ndarray]:
        """
        Split data into training, validation and test sets
        
        Args:
            X: Input sequences
            y: Target sequences
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
            
        Returns:
            Dictionary with train, val, test splits for X and y
        """
        n_samples = len(X)
        train_end = int(n_samples * train_split)
        val_end = train_end + int(n_samples * val_split)
        
        # Perform the split
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }