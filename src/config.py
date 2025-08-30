# src/config.py

# Trading parameters
CURRENCIES = ["EUR", "GBP", "JPY"]
BASE_CURRENCY = "USD"
TRADE_MAX_PERCENTAGE = 0.2
LEVERAGE = 20


# Environment parameters
BARS_COUNT = 30
MAX_STEPS_TRAINING = 5000  # Max steps for the training environment

# Model parameters
LSTM_HIDDEN_SIZE = 128
FEATURES_DIM = 256

# Training parameters
TRAIN_DATA_PATH = "data/Hourly_Rates.csv"
MODEL_SAVE_PATH = "models/rppo_forex_v14.zip" # Incremented version
LOG_DIR = "rppo_logs/"
TOTAL_TIMESTEPS = 1000000
N_ENVS = 4
N_STEPS = 256