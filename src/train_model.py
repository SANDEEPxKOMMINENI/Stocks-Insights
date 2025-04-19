import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

# Import our modules
from data.stock_data import StockDataProcessor
from models.informer import Informer
from models.base_model import StockPredictionModel
from training.trainer import StockModelTrainer
from visualization.visualizer import StockVisualizer

def parse_args():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train a stock prediction model")
    
    # Data arguments
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker to use for training")
    parser.add_argument("--period", type=str, default="5y", help="Data period (e.g., 5y, 10y)")
    parser.add_argument("--train_split", type=float, default=0.7, help="Proportion of data for training")
    parser.add_argument("--val_split", type=float, default=0.15, help="Proportion of data for validation")
    
    # Model arguments
    parser.add_argument("--seq_len", type=int, default=60, help="Input sequence length")
    parser.add_argument("--out_len", type=int, default=5, help="Prediction horizon")
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--e_layers", type=int, default=2, help="Number of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="Number of decoder layers")
    parser.add_argument("--d_ff", type=int, default=256, help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment name")
    
    # Other arguments
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    return parser.parse_args()

def set_seed(seed):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    """
    Main function to train the model
    """
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create experiment name if not provided
    if args.experiment is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment = f"{args.ticker}_seq{args.seq_len}_out{args.out_len}_{timestamp}"
        
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
        
    print(f"Training {args.ticker} model with experiment name: {args.experiment}")
    print(f"Using device: {device}")
    
    # Create data processor
    data_processor = StockDataProcessor(cache_dir="data/cache")
    
    # Download and preprocess data
    print(f"Downloading and preprocessing {args.ticker} data...")
    stock_data = data_processor.download_stock_data(
        ticker=args.ticker,
        period=args.period,
        interval="1d",
        force_download=False
    )
    
    # Add technical indicators
    stock_data = data_processor.add_technical_indicators(stock_data)
    
    # Define feature columns
    feature_columns = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_20", "RSI", "MACD", "BB_High", "BB_Low"
    ]
    
    # Prepare sequences
    print("Preparing data sequences...")
    X, y = data_processor.prepare_sequences(
        df=stock_data,
        input_seq_len=args.seq_len,
        forecast_horizon=args.out_len,
        target_col="Close",
        feature_columns=feature_columns
    )
    
    # Split data
    data_splits = data_processor.create_train_val_test_split(
        X=X,
        y=y,
        train_split=args.train_split,
        val_split=args.val_split
    )
    
    # Create model
    print("Creating model...")
    enc_in = dec_in = X.shape[2]  # Number of input features
    c_out = 1  # Number of output features
    
    informer = Informer(
        enc_in=enc_in,
        dec_in=dec_in,
        c_out=c_out,
        seq_len=args.seq_len,
        label_len=5,  # Length of known labels for decoder input
        out_len=args.out_len,
        factor=5,     # Prob attention factor
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        attn="prob",  # Attention type: 'prob' or 'full'
        activation="gelu"
    )
    
    # Create model config
    model_config = {
        "name": f"{args.ticker}_stock_predictor",
        "architecture": "Informer",
        "enc_in": enc_in,
        "dec_in": dec_in,
        "c_out": c_out,
        "seq_len": args.seq_len,
        "label_len": 5,
        "out_len": args.out_len,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "e_layers": args.e_layers,
        "d_layers": args.d_layers,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "attn": "prob",
        "activation": "gelu",
        "features": feature_columns,
        "training_data": args.ticker,
        "data_period": args.period
    }
    
    # Create stock prediction model
    model = StockPredictionModel(informer, model_config)
    
    # Create trainer
    trainer = StockModelTrainer(
        model=model,
        device=device,
        experiment_name=args.experiment
    )
    
    # Create dataloaders
    print("Creating dataloaders...")
    dataloaders = trainer.create_dataloaders(
        data_dict=data_splits,
        batch_size=args.batch_size,
        shuffle_train=True
    )
    
    # Train model
    print("Training model...")
    history = trainer.train(
        dataloaders=dataloaders,
        epochs=args.epochs,
        lr=args.lr,
        early_stopping_patience=args.patience,
        save_best_model=True,
        verbose=args.verbose
    )
    
    # Evaluate model on test set
    print("Evaluating model on test set...")
    test_metrics = trainer.evaluate(dataloaders["test"])
    print(f"Test metrics: {test_metrics}")
    
    # Plot training history
    trainer.plot_training_history(save=True)
    
    # Plot predictions
    print("Generating prediction plots...")
    visualizer = StockVisualizer(save_dir=os.path.join("experiments", args.experiment, "plots"))
    
    # Get some test samples
    test_inputs = data_splits["X_test"][:5]
    test_targets = data_splits["y_test"][:5]
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        test_input_tensor = torch.tensor(test_inputs, dtype=torch.float32).to(device)
        test_predictions = model(test_input_tensor).cpu().numpy()
    
    # Plot prediction sequences
    visualizer.plot_prediction_sequences(
        y_true_sequences=test_targets,
        y_pred_sequences=test_predictions,
        n_samples=5,
        save=True,
        filename="test_predictions.png"
    )
    
    # Save for Hugging Face
    print("Preparing model for Hugging Face...")
    hf_dir = trainer.save_for_huggingface(
        model_name=f"{args.ticker.lower()}-stock-prediction",
        model_description=f"Stock price prediction model for {args.ticker} using the Informer architecture"
    )
    
    print(f"Model saved for Hugging Face at: {hf_dir}")
    print(f"Experiment completed: {args.experiment}")
    
if __name__ == "__main__":
    main()