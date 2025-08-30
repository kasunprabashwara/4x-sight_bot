import pandas as pd
from src.config import TRAIN_DATA_PATH

def get_forex_data():
    """Load and preprocess the forex data from the CSV file."""
    data_set = pd.read_csv(TRAIN_DATA_PATH, na_values='ND', parse_dates=['Timestamp'])
    data_set.interpolate(inplace=True)
    df = data_set[['Timestamp', 'EURUSD', 'GBPUSD', 'JPYUSD']].copy()
    return df
