import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union, Any
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime

class StockVisualizer:
    """
    Class for visualizing stock data and model predictions
    """
    def __init__(self, save_dir: str = "plots"):
        """
        Initialize the visualizer
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set plot styling
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]
        
    def plot_stock_history(self, 
                          df: pd.DataFrame, 
                          ticker: str,
                          columns: List[str] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          title: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 6),
                          save: bool = True,
                          filename: Optional[str] = None) -> None:
        """
        Plot historical stock data
        
        Args:
            df: DataFrame with stock data
            ticker: Stock ticker symbol
            columns: Columns to plot
            start_date: Start date for the plot
            end_date: End date for the plot
            title: Plot title
            figsize: Figure size
            save: Whether to save the plot
            filename: Filename for saved plot
        """
        # Default to plotting Close prices if no columns specified
        if columns is None:
            columns = ['Close']
            
        # Filter by date range if specified
        plot_df = df.copy()
        if start_date:
            plot_df = plot_df[plot_df.index >= start_date]
        if end_date:
            plot_df = plot_df[plot_df.index <= end_date]
            
        # Create plot
        plt.figure(figsize=figsize)
        
        for i, col in enumerate(columns):
            if col in plot_df.columns:
                plt.plot(plot_df.index, plot_df[col], 
                         label=col, 
                         color=self.colors[i % len(self.colors)])
                
        # Set title and labels
        if title:
            plt.title(title, fontsize=16)
        else:
            plt.title(f"{ticker} Stock Price History", fontsize=16)
            
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price", fontsize=12)
        plt.legend()
        plt.tight_layout()
        
        # Save plot if required
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{ticker}_history_{timestamp}.png"
            plt.savefig(os.path.join(self.save_dir, filename))
            
        plt.show()
        
    def plot_technical_indicators(self, 
                                 df: pd.DataFrame, 
                                 ticker: str,
                                 indicators: List[str],
                                 price_col: str = 'Close',
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None,
                                 figsize: Tuple[int, int] = (12, 8),
                                 save: bool = True,
                                 filename: Optional[str] = None) -> None:
        """
        Plot stock price with technical indicators
        
        Args:
            df: DataFrame with stock data
            ticker: Stock ticker symbol
            indicators: List of technical indicators to plot
            price_col: Column to use for price
            start_date: Start date for the plot
            end_date: End date for the plot
            figsize: Figure size
            save: Whether to save the plot
            filename: Filename for saved plot
        """
        # Filter by date range if specified
        plot_df = df.copy()
        if start_date:
            plot_df = plot_df[plot_df.index >= start_date]
        if end_date:
            plot_df = plot_df[plot_df.index <= end_date]
            
        # Create subplot grid - one for price, one for each indicator
        fig, axes = plt.subplots(len(indicators) + 1, 1, figsize=figsize, sharex=True)
        
        # Plot price
        axes[0].plot(plot_df.index, plot_df[price_col], color=self.colors[0], label=price_col)
        axes[0].set_title(f"{ticker} Stock Price with Technical Indicators", fontsize=16)
        axes[0].set_ylabel("Price", fontsize=12)
        axes[0].legend(loc='upper left')
        
        # Plot each indicator
        for i, indicator in enumerate(indicators):
            if indicator in plot_df.columns:
                axes[i+1].plot(plot_df.index, plot_df[indicator], 
                             color=self.colors[(i+1) % len(self.colors)], 
                             label=indicator)
                axes[i+1].set_ylabel(indicator, fontsize=10)
                axes[i+1].legend(loc='upper left')
                
        # Set common x-axis label
        axes[-1].set_xlabel("Date", fontsize=12)
        
        plt.tight_layout()
        
        # Save plot if required
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{ticker}_indicators_{timestamp}.png"
            plt.savefig(os.path.join(self.save_dir, filename))
            
        plt.show()
        
    def plot_model_predictions(self, 
                              y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              dates: Optional[List[datetime]] = None,
                              title: str = "Model Predictions vs Actual",
                              figsize: Tuple[int, int] = (12, 6),
                              save: bool = True,
                              filename: Optional[str] = None) -> Dict[str, float]:
        """
        Plot model predictions against actual values
        
        Args:
            y_true: Actual values (np array or list)
            y_pred: Predicted values (np array or list)
            dates: Optional list of dates for x-axis
            title: Plot title
            figsize: Figure size
            save: Whether to save the plot
            filename: Filename for saved plot
            
        Returns:
            Dictionary with performance metrics
        """
        # Ensure arrays are flattened
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        
        # Calculate metrics
        mse = mean_squared_error(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        
        # Calculate directional accuracy
        direction_true = np.diff(y_true_flat)
        direction_pred = np.diff(y_pred_flat)
        directional_accuracy = np.mean((direction_true > 0) == (direction_pred > 0))
        
        # Create plot
        plt.figure(figsize=figsize)
        
        x_axis = dates if dates is not None else np.arange(len(y_true_flat))
        
        plt.plot(x_axis, y_true_flat, color=self.colors[0], label='Actual', linewidth=2)
        plt.plot(x_axis, y_pred_flat, color=self.colors[1], label='Predicted', linewidth=2, linestyle='--')
        
        # Add metrics to title
        title_with_metrics = f"{title}\nRMSE: {rmse:.4f}, MAE: {mae:.4f}, Directional Accuracy: {directional_accuracy:.2%}"
        plt.title(title_with_metrics, fontsize=14)
        
        if dates is not None:
            plt.xlabel("Date", fontsize=12)
        else:
            plt.xlabel("Time Steps", fontsize=12)
            
        plt.ylabel("Value", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot if required
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"model_predictions_{timestamp}.png"
            plt.savefig(os.path.join(self.save_dir, filename))
            
        plt.show()
        
        # Return metrics
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'directional_accuracy': directional_accuracy
        }
        
        return metrics
    
    def plot_prediction_sequences(self, 
                                 y_true_sequences: np.ndarray, 
                                 y_pred_sequences: np.ndarray,
                                 n_samples: int = 5,
                                 figsize: Tuple[int, int] = (15, 10),
                                 save: bool = True,
                                 filename: Optional[str] = None) -> None:
        """
        Plot multiple prediction sequences
        
        Args:
            y_true_sequences: True sequences (shape: [n_samples, seq_len, n_features])
            y_pred_sequences: Predicted sequences (shape: [n_samples, seq_len, n_features])
            n_samples: Number of sample sequences to plot
            figsize: Figure size
            save: Whether to save the plot
            filename: Filename for saved plot
        """
        n_samples = min(n_samples, y_true_sequences.shape[0])
        
        # Create subplot grid - one for each sample
        fig, axes = plt.subplots(n_samples, 1, figsize=figsize)
        
        # Make axes iterable for single sample case
        if n_samples == 1:
            axes = [axes]
            
        # Plot each sample
        for i in range(n_samples):
            true_seq = y_true_sequences[i, :, 0]  # Assuming feature 0 is the main prediction target
            pred_seq = y_pred_sequences[i, :, 0]
            
            axes[i].plot(true_seq, color=self.colors[0], label='Actual')
            axes[i].plot(pred_seq, color=self.colors[1], label='Predicted', linestyle='--')
            axes[i].set_title(f"Sequence {i+1}", fontsize=12)
            axes[i].set_ylabel("Value", fontsize=10)
            axes[i].legend(loc='upper right')
            axes[i].grid(True, alpha=0.3)
            
        # Set common x-axis label
        axes[-1].set_xlabel("Time Steps", fontsize=12)
        
        plt.tight_layout()
        
        # Save plot if required
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"prediction_sequences_{timestamp}.png"
            plt.savefig(os.path.join(self.save_dir, filename))
            
        plt.show()
        
    def plot_loss_history(self, 
                         train_losses: List[float], 
                         val_losses: Optional[List[float]] = None,
                         title: str = "Training and Validation Loss",
                         figsize: Tuple[int, int] = (10, 6),
                         save: bool = True,
                         filename: Optional[str] = None) -> None:
        """
        Plot training and validation loss history
        
        Args:
            train_losses: List of training losses
            val_losses: Optional list of validation losses
            title: Plot title
            figsize: Figure size
            save: Whether to save the plot
            filename: Filename for saved plot
        """
        plt.figure(figsize=figsize)
        
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, color=self.colors[0], label='Training Loss')
        if val_losses:
            plt.plot(epochs, val_losses, color=self.colors[1], label='Validation Loss')
            
        plt.title(title, fontsize=16)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot if required
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"loss_history_{timestamp}.png"
            plt.savefig(os.path.join(self.save_dir, filename))
            
        plt.show()
        
    def plot_feature_importance(self, 
                               feature_names: List[str], 
                               importance_scores: List[float],
                               title: str = "Feature Importance",
                               figsize: Tuple[int, int] = (10, 8),
                               save: bool = True,
                               filename: Optional[str] = None) -> None:
        """
        Plot feature importance
        
        Args:
            feature_names: Names of features
            importance_scores: Importance scores for each feature
            title: Plot title
            figsize: Figure size
            save: Whether to save the plot
            filename: Filename for saved plot
        """
        # Sort features by importance
        sorted_idx = np.argsort(importance_scores)
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_scores = [importance_scores[i] for i in sorted_idx]
        
        # Create horizontal bar plot
        plt.figure(figsize=figsize)
        
        y_pos = np.arange(len(sorted_features))
        
        plt.barh(y_pos, sorted_scores, color=self.colors[0])
        plt.yticks(y_pos, sorted_features)
        plt.title(title, fontsize=16)
        plt.xlabel("Importance Score", fontsize=12)
        plt.tight_layout()
        
        # Save plot if required
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"feature_importance_{timestamp}.png"
            plt.savefig(os.path.join(self.save_dir, filename))
            
        plt.show()
        
    def plot_profit_simulation(self, 
                              y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              dates: Optional[List[datetime]] = None,
                              initial_investment: float = 10000.0,
                              trade_cost: float = 0.001,  # 0.1% transaction cost
                              strategy: str = 'follow_prediction',
                              title: str = "Trading Profit Simulation",
                              figsize: Tuple[int, int] = (12, 6),
                              save: bool = True,
                              filename: Optional[str] = None) -> Dict[str, float]:
        """
        Simulate trading profit based on model predictions
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            dates: Optional list of dates for x-axis
            initial_investment: Initial investment amount
            trade_cost: Transaction cost as a fraction
            strategy: Trading strategy ('follow_prediction' or 'threshold')
            title: Plot title
            figsize: Figure size
            save: Whether to save the plot
            filename: Filename for saved plot
            
        Returns:
            Dictionary with performance metrics
        """
        # Flatten arrays
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        
        # Initialize portfolio values
        portfolio_value_model = np.zeros(len(y_true_flat))
        portfolio_value_buyhold = np.zeros(len(y_true_flat))
        
        # Initial investment
        portfolio_value_model[0] = initial_investment
        portfolio_value_buyhold[0] = initial_investment
        
        # Simulate trading based on predictions
        shares_owned_model = initial_investment / y_true_flat[0]
        shares_owned_buyhold = initial_investment / y_true_flat[0]
        
        for i in range(1, len(y_true_flat)):
            # Buy and hold strategy
            portfolio_value_buyhold[i] = shares_owned_buyhold * y_true_flat[i]
            
            # Model-based strategy
            if strategy == 'follow_prediction':
                # Buy if predicted to go up, sell if predicted to go down
                if y_pred_flat[i] > y_pred_flat[i-1]:
                    # Only buy if not already holding
                    if shares_owned_model == 0:
                        # Buy with transaction cost
                        cash = portfolio_value_model[i-1]
                        shares_owned_model = (cash * (1 - trade_cost)) / y_true_flat[i]
                elif y_pred_flat[i] < y_pred_flat[i-1]:
                    # Only sell if holding shares
                    if shares_owned_model > 0:
                        # Sell with transaction cost
                        cash = shares_owned_model * y_true_flat[i] * (1 - trade_cost)
                        shares_owned_model = 0
                        portfolio_value_model[i-1] = cash
            
            # Update portfolio value
            if shares_owned_model > 0:
                portfolio_value_model[i] = shares_owned_model * y_true_flat[i]
            else:
                portfolio_value_model[i] = portfolio_value_model[i-1]
                
        # Calculate returns
        model_return = (portfolio_value_model[-1] - initial_investment) / initial_investment
        buyhold_return = (portfolio_value_buyhold[-1] - initial_investment) / initial_investment
        
        # Create plot
        plt.figure(figsize=figsize)
        
        x_axis = dates if dates is not None else np.arange(len(y_true_flat))
        
        plt.plot(x_axis, portfolio_value_model, color=self.colors[0], label='Model Strategy')
        plt.plot(x_axis, portfolio_value_buyhold, color=self.colors[1], label='Buy & Hold')
        
        # Add returns to title
        title_with_returns = f"{title}\nModel Return: {model_return:.2%}, Buy & Hold Return: {buyhold_return:.2%}"
        plt.title(title_with_returns, fontsize=14)
        
        if dates is not None:
            plt.xlabel("Date", fontsize=12)
        else:
            plt.xlabel("Time Steps", fontsize=12)
            
        plt.ylabel("Portfolio Value ($)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot if required
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"profit_simulation_{timestamp}.png"
            plt.savefig(os.path.join(self.save_dir, filename))
            
        plt.show()
        
        # Return metrics
        metrics = {
            'model_return': model_return,
            'buyhold_return': buyhold_return,
            'outperformance': model_return - buyhold_return,
            'final_portfolio_model': portfolio_value_model[-1],
            'final_portfolio_buyhold': portfolio_value_buyhold[-1]
        }
        
        return metrics