# src/config.py

# Trading parameters
CURRENCIES = ["EUR", "GBP", "JPY"]
BASE_CURRENCY = "USD"
INITIAL_BALANCE = 1000
TRADE_MAX_PERCENTAGE = 0.2
LEVERAGE = 60

# Model parameters
BARS_COUNT = 30
MAX_STEPS = 5000
LSTM_HIDDEN_SIZE = 128
FEATURES_DIM = 256

# Training parameters
TRAIN_DATA_PATH = "data/Hourly_Rates.csv"
MODEL_SAVE_PATH = "models/rppo_forex_v13.zip"
LOG_DIR = "rppo_logs/"
TOTAL_TIMESTEPS = 1000000
N_ENVS = 4
N_STEPS = 256

# MT5 parameters

MT5_PATH=r"C:\Program Files\MetaTrader 5\terminal64.exe"
MT5_LOGIN=5034726293
MT5_PASSWORD="Uo*hEbE8"
MT5_SERVER="MetaQuotes-Demo"
MT5_TIMEOUT=10000
MT5_PORTABLE=True