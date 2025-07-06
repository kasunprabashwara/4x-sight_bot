import pandas as pd
from src.config import TRAIN_DATA_PATH

def get_forex_data():
    """Load and preprocess the forex data from the CSV file."""
    data_set = pd.read_csv(TRAIN_DATA_PATH, na_values='ND', parse_dates=['Timestamp'])
    data_set.interpolate(inplace=True)
    df = data_set[['Timestamp', 'EURUSD', 'GBPUSD', 'JPYUSD']].copy()
    return df

def get_pair_price_from_row(row, pair, base_currency):
    """
    Given a row (a pandas Series) and a pair (tuple of two currencies),
    return the exchange rate.
    """
    A, B = pair
    if A == B:
        return 1.0
    if B == base_currency:
        return row[f"{A}{base_currency}"]
    elif A == base_currency:
        return 1.0 / row[f"{B}{base_currency}"]
    else:
        return row[f"{A}{base_currency}"] / row[f"{B}{base_currency}"]
