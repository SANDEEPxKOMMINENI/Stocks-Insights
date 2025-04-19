import gradio as gr
import numpy as np
import pandas as pd
import torch
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf

# Import our modules
from data.stock_data import StockDataProcessor
from models.informer import Informer
from models.base_model import StockPredictionModel
from visualization.visualizer import StockVisualizer

# Constants
DEFAULT_TICKER = "AAPL"
DEFAULT_LOOKBACK = 60
DEFAULT_HORIZON = 5
MODEL_PATH = "models/stock_predictor_best"

class StockPredictionApp:
    """
    Gradio application for stock price prediction
    """
    def __init__(self):
        """
        Initialize the Stock Prediction App
        """
        # Initialize data processor
        self.data_processor = StockDataProcessor(cache_dir="data/cache")
        
        # Initialize visualizer
        self.visualizer = StockVisualizer(save_dir="plots")
        
        # Load config if exists or use defaults
        self.config = self.load_config()
        
        # Load model if exists or it will be created on first prediction
        self.model = None
        if os.path.exists(f"{MODEL_PATH}.pt"):
            self.load_model()
            
    def load_config(self) -> dict:
        """
        Load configuration from file or return defaults
        
        Returns:
            Configuration dictionary
        """
        config_path = "config/app_config.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                "model_params": {
                    "enc_in": 10,  # Number of input features
                    "dec_in": 10,  # Number of input features for decoder
                    "c_out": 1,    # Number of output features
                    "seq_len": DEFAULT_LOOKBACK,
                    "label_len": 5,  # Length of known labels for decoder input
                    "out_len": DEFAULT_HORIZON,
                    "factor": 5,     # Prob attention factor
                    "d_model": 64,   # Model dimension
                    "n_heads": 4,    # Number of attention heads
                    "e_layers": 2,   # Number of encoder layers
                    "d_layers": 1,   # Number of decoder layers
                    "d_ff": 256,     # Dimension of feed forward network
                    "dropout": 0.1,
                    "attn": "prob",  # Attention type: 'prob' or 'full'
                    "activation": "gelu"
                }
            }
    
    def load_model(self) -> None:
        """
        Load the pre-trained model
        """
        # Make sure the directory exists
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        # Check if model exists
        if os.path.exists(f"{MODEL_PATH}.pt"):
            # Load model config
            with open(f"{MODEL_PATH}_config.json", "r") as f:
                model_config = json.load(f)
                
            # Create model architecture
            informer = Informer(
                enc_in=model_config.get("enc_in", 10),
                dec_in=model_config.get("dec_in", 10),
                c_out=model_config.get("c_out", 1),
                seq_len=model_config.get("seq_len", DEFAULT_LOOKBACK),
                label_len=model_config.get("label_len", 5),
                out_len=model_config.get("out_len", DEFAULT_HORIZON),
                factor=model_config.get("factor", 5),
                d_model=model_config.get("d_model", 64),
                n_heads=model_config.get("n_heads", 4),
                e_layers=model_config.get("e_layers", 2),
                d_layers=model_config.get("d_layers", 1),
                d_ff=model_config.get("d_ff", 256),
                dropout=model_config.get("dropout", 0.1),
                attn=model_config.get("attn", "prob"),
                activation=model_config.get("activation", "gelu")
            )
            
            # Create the wrapper model
            self.model = StockPredictionModel(informer, model_config)
            
            # Load weights
            self.model.load_state_dict(torch.load(f"{MODEL_PATH}.pt"))
            
            # Set to evaluation mode
            self.model.eval()
        else:
            # If no model exists, we'll create one with default parameters when needed
            self._create_default_model()
            
    def _create_default_model(self) -> None:
        """
        Create a default model with parameters from config
        """
        params = self.config["model_params"]
        
        # Create model architecture
        informer = Informer(
            enc_in=params["enc_in"],
            dec_in=params["dec_in"],
            c_out=params["c_out"],
            seq_len=params["seq_len"],
            label_len=params["label_len"],
            out_len=params["out_len"],
            factor=params["factor"],
            d_model=params["d_model"],
            n_heads=params["n_heads"],
            e_layers=params["e_layers"],
            d_layers=params["d_layers"],
            d_ff=params["d_ff"],
            dropout=params["dropout"],
            attn=params["attn"],
            activation=params["activation"]
        )
        
        # Create the wrapper model
        self.model = StockPredictionModel(informer, params)
        
        # Set to evaluation mode
        self.model.eval()
        
    def fetch_stock_data(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """
        Fetch stock data for the given ticker
        
        Args:
            ticker: Stock ticker symbol
            period: Period for which to fetch data
            
        Returns:
            DataFrame with stock data
        """
        # Download stock data
        df = self.data_processor.download_stock_data(
            ticker=ticker,
            period=period,
            interval="1d",
            force_download=False
        )
        
        # Add technical indicators
        df = self.data_processor.add_technical_indicators(df)
        
        return df
        
    def prepare_input_data(self, 
                          df: pd.DataFrame, 
                          lookback_window: int = DEFAULT_LOOKBACK,
                          target_column: str = "Close") -> np.ndarray:
        """
        Prepare input data for the model
        
        Args:
            df: DataFrame with stock data
            lookback_window: Number of days to look back
            target_column: Target column to predict
            
        Returns:
            Processed input data for the model
        """
        # Get the latest data for the lookback window
        latest_data = df.iloc[-lookback_window:].copy()
        
        # Select features - this should match what the model was trained on
        feature_columns = [
            "Open", "High", "Low", "Close", "Volume",
            "SMA_20", "RSI", "MACD", "BB_High", "BB_Low"
        ]
        
        # Make sure all required columns exist
        for col in feature_columns:
            if col not in latest_data.columns:
                # If a column is missing, add it with zeros
                latest_data[col] = 0
        
        # Get sequences
        X, _ = self.data_processor.prepare_sequences(
            latest_data,
            input_seq_len=lookback_window,
            forecast_horizon=DEFAULT_HORIZON,
            target_col=target_column,
            feature_columns=feature_columns
        )
        
        return X
        
    def predict_stock_prices(self, 
                           ticker: str, 
                           prediction_days: int = DEFAULT_HORIZON) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Predict stock prices for the given ticker
        
        Args:
            ticker: Stock ticker symbol
            prediction_days: Number of days to predict
            
        Returns:
            Tuple of (past_prices, predicted_prices, stock_data)
        """
        # Fetch stock data
        df = self.fetch_stock_data(ticker)
        
        # Prepare input data
        input_data = self.prepare_input_data(df)
        
        # Make sure model is loaded
        if self.model is None:
            self.load_model()
            
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            # Convert input to tensor
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            
            # Get predictions
            predictions = self.model(input_tensor).numpy()
            
        # Inverse transform the predictions
        orig_predictions = self.data_processor.inverse_transform_predictions(
            predictions[0], target_col="Close"
        )
        
        # Get the past actual prices for comparison
        past_prices = df["Close"].values[-DEFAULT_LOOKBACK:]
        
        return past_prices, orig_predictions.flatten(), df
    
    def create_prediction_plot(self, 
                              past_prices: np.ndarray, 
                              predicted_prices: np.ndarray,
                              ticker: str) -> plt.Figure:
        """
        Create a plot of past prices and predictions
        
        Args:
            past_prices: Past stock prices
            predicted_prices: Predicted stock prices
            ticker: Stock ticker symbol
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig = plt.figure(figsize=(10, 6))
        
        # Create x-axis for past and future
        past_range = np.arange(len(past_prices))
        future_range = np.arange(len(past_prices), len(past_prices) + len(predicted_prices))
        
        # Create date labels
        today = datetime.now().date()
        past_dates = [(today - timedelta(days=len(past_prices) - i)) for i in range(len(past_prices))]
        future_dates = [(today + timedelta(days=i+1)) for i in range(len(predicted_prices))]
        
        # Plot past prices
        plt.plot(past_range, past_prices, color="blue", label="Historical Prices")
        
        # Plot predicted prices
        plt.plot(future_range, predicted_prices, color="red", label="Predicted Prices", linestyle="dashed")
        
        # Add vertical line for today
        plt.axvline(x=len(past_prices)-1, color="green", linestyle="--", label="Today")
        
        # Add labels and title
        plt.title(f"{ticker} Stock Price Prediction", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price ($)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.xticks(
            np.concatenate([past_range[::10], future_range]),
            [d.strftime("%m/%d") for d in np.concatenate([past_dates[::10], future_dates])],
            rotation=45
        )
        
        plt.tight_layout()
        
        return fig
    
    def create_technical_plot(self, df: pd.DataFrame, ticker: str) -> plt.Figure:
        """
        Create a plot with technical indicators
        
        Args:
            df: DataFrame with stock data
            ticker: Stock ticker symbol
            
        Returns:
            Matplotlib figure
        """
        # Get the last 120 days of data
        plot_df = df.iloc[-120:].copy()
        
        # Create figure with multiple subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot 1: Price and SMAs
        ax1.plot(plot_df.index, plot_df["Close"], color="blue", label="Close Price")
        ax1.plot(plot_df.index, plot_df["SMA_20"], color="orange", label="SMA 20")
        ax1.plot(plot_df.index, plot_df["SMA_50"], color="red", label="SMA 50")
        ax1.plot(plot_df.index, plot_df["SMA_200"], color="purple", label="SMA 200")
        ax1.set_title(f"{ticker} Stock Price with Technical Indicators", fontsize=16)
        ax1.set_ylabel("Price ($)", fontsize=12)
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Bollinger Bands
        ax2.plot(plot_df.index, plot_df["Close"], color="blue", label="Close Price")
        ax2.plot(plot_df.index, plot_df["BB_High"], color="red", label="BB High")
        ax2.plot(plot_df.index, plot_df["BB_Mid"], color="green", label="BB Mid")
        ax2.plot(plot_df.index, plot_df["BB_Low"], color="red", label="BB Low")
        ax2.set_ylabel("Price ($)", fontsize=12)
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: RSI
        ax3.plot(plot_df.index, plot_df["RSI"], color="purple", label="RSI")
        ax3.axhline(y=70, color="red", linestyle="--", alpha=0.5)
        ax3.axhline(y=30, color="green", linestyle="--", alpha=0.5)
        ax3.set_ylabel("RSI", fontsize=12)
        ax3.set_xlabel("Date", fontsize=12)
        ax3.legend(loc="upper left")
        ax3.grid(True, alpha=0.3)
        
        # Format dates
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        return fig
    
    def predict_and_plot(self, ticker: str, days: int) -> Tuple[plt.Figure, plt.Figure, str]:
        """
        Predict stock prices and create plots
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to predict
            
        Returns:
            Tuple of (prediction_plot, technical_plot, prediction_text)
        """
        if not ticker:
            ticker = DEFAULT_TICKER
            
        try:
            # Make predictions
            past_prices, predictions, stock_data = self.predict_stock_prices(ticker, days)
            
            # Create plots
            prediction_plot = self.create_prediction_plot(past_prices, predictions, ticker)
            technical_plot = self.create_technical_plot(stock_data, ticker)
            
            # Create prediction text
            last_price = past_prices[-1]
            next_day_price = predictions[0]
            pct_change = ((next_day_price - last_price) / last_price) * 100
            direction = "UP 📈" if pct_change > 0 else "DOWN 📉"
            
            prediction_text = f"""## Prediction Summary for {ticker}

**Current Price:** ${last_price:.2f}

**Next Trading Day:** ${next_day_price:.2f} ({pct_change:.2f}%) {direction}

**5-Day Forecast:**
"""
            
            # Add each day's prediction
            today = datetime.now().date()
            for i, price in enumerate(predictions):
                forecast_date = today + timedelta(days=i+1)
                day_of_week = forecast_date.strftime("%A")
                if day_of_week not in ["Saturday", "Sunday"]:  # Skip weekends
                    prediction_text += f"- {forecast_date.strftime('%Y-%m-%d')} ({day_of_week}): ${price:.2f}\n"
            
            # Add disclaimer
            prediction_text += """
---
**Disclaimer:** These predictions are based on historical patterns and technical indicators. 
Financial markets are complex systems affected by numerous factors, and all predictions should
be considered as estimates rather than guarantees. Always conduct your own research before making
investment decisions.
"""
            
            return prediction_plot, technical_plot, prediction_text
        except Exception as e:
            # Return error message
            error_text = f"""## Error making prediction

There was an error processing your request for ticker **{ticker}**:

```
{str(e)}
```

Please check the ticker symbol and try again. If the problem persists, it might be due to:
1. Invalid ticker symbol
2. Missing or insufficient historical data
3. Server-side issue with data provider

Try another popular ticker like AAPL, MSFT, GOOG, or AMZN.
"""
            # Create empty figures
            fig1, fig2 = plt.figure(), plt.figure()
            
            return fig1, fig2, error_text
    
    def build_gradio_interface(self) -> gr.Blocks:
        """
        Build the Gradio interface
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="Stock Price Prediction") as interface:
            gr.Markdown("# 📈 Stock Price Prediction with Deep Learning")
            gr.Markdown("""
            This application uses a **Transformer-based deep learning model** (Informer architecture) to predict stock prices.
            Enter a stock ticker symbol and prediction horizon to get started.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    ticker_input = gr.Textbox(label="Stock Ticker Symbol", 
                                            placeholder="e.g., AAPL, MSFT, GOOG",
                                            value=DEFAULT_TICKER)
                    days_input = gr.Slider(minimum=1, maximum=30, value=5, step=1, 
                                        label="Prediction Horizon (Days)")
                    predict_button = gr.Button("Predict Stock Prices", variant="primary")
                    
                with gr.Column(scale=2):
                    prediction_text = gr.Markdown()
            
            with gr.Tabs():
                with gr.Tab("Price Prediction"):
                    prediction_plot = gr.Plot()
                with gr.Tab("Technical Analysis"):
                    technical_plot = gr.Plot()
                    
            predict_button.click(
                fn=self.predict_and_plot,
                inputs=[ticker_input, days_input],
                outputs=[prediction_plot, technical_plot, prediction_text]
            )
            
            gr.Markdown("""
            ## About the Model
            
            This stock prediction model is based on the **Informer** architecture, a Transformer-based model designed for 
            long sequence time-series forecasting. It analyzes historical price data and technical indicators to generate predictions.
            
            **Features used:**
            - Historical OHLCV (Open, High, Low, Close, Volume) data
            - Technical indicators (SMA, RSI, MACD, Bollinger Bands)
            - Price momentum and volatility metrics
            
            ## How It Works
            
            1. Historical data is fetched for the requested ticker
            2. Technical indicators are calculated and added as features
            3. Data is normalized and processed into sequences
            4. The model makes predictions based on recent price patterns
            5. Results are visualized and presented with a confidence estimate
            
            ## Disclaimer
            
            This tool is for educational and research purposes only. The predictions are based on historical 
            patterns and should not be used as financial advice. Always conduct thorough research before 
            making investment decisions.
            """)
            
        return interface
    
    def launch(self, share: bool = False) -> None:
        """
        Launch the Gradio interface
        
        Args:
            share: Whether to create a shareable link
        """
        interface = self.build_gradio_interface()
        interface.launch(share=share)
        
def main():
    """
    Main function to run the app
    """
    # Create app directory structure
    os.makedirs("data/cache", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    
    # Initialize and launch app
    app = StockPredictionApp()
    app.launch()
    
if __name__ == "__main__":
    main()