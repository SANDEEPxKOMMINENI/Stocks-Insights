# Stock Price Prediction Model with Transformer Architecture

This project implements a deep learning model for stock price prediction using the Informer architecture, a Transformer-based model optimized for time series forecasting.

## Features

- Data collection and preprocessing pipeline for stock time series data
- Advanced Transformer-based model architecture (Informer) for time series forecasting
- Comprehensive training and evaluation framework
- Feature engineering including technical indicators and time embeddings
- Interactive Gradio interface for model demonstration
- Export functionality for Hugging Face deployment

## Project Structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── app.py                   # Gradio web interface
│   ├── train_model.py           # Model training script
│   ├── download_data.py         # Data download script
│   ├── data/                    # Data processing modules
│   │   └── stock_data.py        # Stock data processor
│   ├── models/                  # Model architectures
│   │   ├── base_model.py        # Base model class
│   │   └── informer.py          # Informer model implementation
│   ├── training/                # Training utilities
│   │   └── trainer.py           # Model trainer
│   └── visualization/           # Visualization tools
│       └── visualizer.py        # Stock data visualizer
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/stock-prediction-transformer.git
cd stock-prediction-transformer
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Download Stock Data

To download historical stock data:

```bash
python src/download_data.py --tickers AAPL MSFT GOOG --period 5y --indicators --save_csv
```

### Train the Model

To train a stock prediction model:

```bash
python src/train_model.py --ticker AAPL --period 5y --seq_len 60 --out_len 5 --epochs 50
```

### Run the Web Interface

To launch the Gradio web interface:

```bash
python src/app.py
```

## Model Architecture

The stock prediction model is based on the Informer architecture, a Transformer-based model designed for long sequence time-series forecasting. Key components include:

- Probability Sparse Self-attention mechanism for efficient computation
- Encoder-decoder architecture with distilling operations
- Feature-level attention for capturing cross-dimensional dependencies

## Hugging Face Integration

After training a model, you can upload it to Hugging Face:

1. The model will be saved in the `experiments/{experiment_name}/huggingface` directory
2. Use the Hugging Face CLI to upload the model:

```bash
huggingface-cli login
huggingface-cli upload {experiment_name}/huggingface your-username/model-name
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Informer architecture is based on the paper [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436) by Zhou et al.
- Stock data is sourced from Yahoo Finance using the `yfinance` package.