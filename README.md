# StockSense AI — Stock Price Forecasting with RNN/LSTM

A beginner-friendly, portfolio-ready Streamlit application for exploring historical stock prices, technical indicators, deep-learning forecasts, and honest model evaluation.

> **Important:** This project is for education and experimentation only. Forecasts are uncertain and are not financial advice.

## Features

- Historical stock data for Indian NSE and US tickers through `yfinance`
- Offline demo mode using synthetic sample data (clearly labelled; not real market history)
- Interactive close-price and candlestick charts
- SMA 20, SMA 50, EMA 20, RSI 14, MACD and MACD signal
- Simple RNN and LSTM model selection
- Chronological train/validation/test split
- Scaling fitted on the training set only
- Early stopping and learning-rate reduction
- Next-session and recursive multi-session forecasts
- MAE, RMSE and directional accuracy
- Naive last-close baseline
- CSV downloads for historical data, forecasts and test predictions
- Separated source files and starter tests

## Recommended environment

- Windows 10/11, 64-bit
- Python 3.11 (recommended for compatibility with the TensorFlow dependency)
- Internet connection for live market data and package installation

## Fastest way to run on Windows

1. Install **Python 3.11 64-bit** from the official Python website: https://www.python.org/downloads/
2. Extract this project folder.
3. Open the folder in VS Code.
4. Double-click `run_windows.bat` (or run it from the VS Code terminal).
5. Wait for dependency installation. The app opens at `http://localhost:8501`.
6. Select a ticker, period and model in the sidebar, then click **Train & forecast**.

The first setup downloads Python packages and may take several minutes. Later launches are faster.

## Manual VS Code setup

Open a terminal in this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Using `.venv\Scripts\python.exe` directly avoids PowerShell activation-policy problems.

## Example tickers

- India: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `^NSEI` (Nifty 50 index, if supported by provider)
- US: `AAPL`, `MSFT`, `NVDA`

Market-data symbols, availability and history depend on the data provider. `yfinance` is an unofficial interface to Yahoo Finance; review provider terms and data-use restrictions before redistribution or commercial use.

## Project structure

```text
stock-price-forecasting-lstm/
├── app.py
├── requirements.txt
├── README.md
├── run_windows.bat
├── .gitignore
├── src/
│   ├── data_loader.py
│   ├── indicators.py
│   ├── preprocessing.py
│   ├── model_builder.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualization.py
├── data/
├── models/
├── reports/
└── tests/
```

## How the model works

The model receives a window of previous closing prices (default: 60 trading sessions) and learns to predict the next session. The default LSTM contains 64 recurrent units, dropout, a dense layer and a single output neuron. A Simple RNN is included for comparison.

- **RNN:** recurrent model for sequential data.
- **LSTM:** recurrent model with gates designed to retain useful sequence information.
- **Naive baseline:** predicts the next price using the last observed close.
- **MAE:** mean absolute error in price units.
- **RMSE:** root mean squared error; penalizes large errors more strongly.
- **Directional accuracy:** fraction of consecutive price-direction changes predicted correctly.

The split is chronological: 70% training, 15% validation, and 15% testing. The scaler is fitted only on training data. Results can change with the ticker, period, model, random initialization and market conditions. Always compare with the baseline; do not assume LSTM is best.

## Limitations

- Historical prices alone do not include all relevant factors, such as news and macroeconomic events.
- Recursive multi-day predictions can accumulate error.
- Forecast dates use business days and do not remove exchange holidays.
- RSI and moving averages describe historical data; they do not guarantee future direction.
- The demo dataset is synthetic and must not be presented as real historical market data.
- No buy/sell recommendation or profit guarantee is produced.

## Troubleshooting

**`py -3.11` is not recognized or Python 3.11 is missing**
Install Python 3.11 64-bit, enable the Python Launcher if offered, and reopen VS Code.

**TensorFlow installation fails**
Confirm that the environment is Python 3.11 64-bit. Remove the project's `.venv` folder and rerun `run_windows.bat`. Check the TensorFlow installation documentation if the platform has a compatibility issue.

**No data returned**
Check internet access, ticker spelling and selected period. Try `RELIANCE.NS` for Reliance Industries or `AAPL` for Apple. Provider availability can change.

**Port 8501 is already in use**
Run Streamlit with another port:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Suggested next improvements

- Walk-forward validation across multiple test windows
- Multi-feature models using OHLCV and technical indicators
- Prediction intervals calibrated on historical validation errors
- Save/load models with explicit metadata and version checks
- A model comparison page that trains both RNN and LSTM on identical splits
- Additional baseline models and transaction-cost-aware backtesting
