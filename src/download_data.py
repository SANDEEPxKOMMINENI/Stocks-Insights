import os
import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path

# Import our data processor
from data.stock_data import StockDataProcessor

def parse_args():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Download stock data")
    
    # Data arguments
    parser.add_argument("--tickers", type=str, nargs="+", default=["AAPL", "MSFT", "GOOG", "AMZN", "META"],
                       help="List of stock tickers to download")
    parser.add_argument("--period", type=str, default="5y", help="Data period (e.g., 5y, 10y)")
    parser.add_argument("--interval", type=str, default="1d", help="Data interval (e.g., 1d, 1h)")
    parser.add_argument("--force", action="store_true", help="Force download even if cached data exists")
    parser.add_argument("--indicators", action="store_true", help="Add technical indicators")
    parser.add_argument("--save_csv", action="store_true", help="Save to CSV files")
    parser.add_argument("--output_dir", type=str, default="data/stocks", help="Output directory for CSV files")
    
    return parser.parse_args()

def main():
    """
    Main function to download stock data
    """
    # Parse arguments
    args = parse_args()
    
    # Create data processor
    data_processor = StockDataProcessor(cache_dir="data/cache")
    
    # Create output directory if saving to CSV
    if args.save_csv:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Download data for each ticker
    for ticker in args.tickers:
        print(f"Downloading {ticker} data...")
        
        try:
            # Download stock data
            df = data_processor.download_stock_data(
                ticker=ticker,
                period=args.period,
                interval=args.interval,
                force_download=args.force
            )
            
            # Add technical indicators if requested
            if args.indicators:
                print(f"Adding technical indicators for {ticker}...")
                df = data_processor.add_technical_indicators(df)
            
            # Print data info
            print(f"{ticker} data shape: {df.shape}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            
            # Save to CSV if requested
            if args.save_csv:
                output_file = os.path.join(args.output_dir, f"{ticker}_{args.interval}.csv")
                df.to_csv(output_file)
                print(f"Saved to {output_file}")
                
            print(f"Successfully processed {ticker}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error downloading {ticker}: {str(e)}")
    
    print("Data download completed!")

if __name__ == "__main__":
    main()