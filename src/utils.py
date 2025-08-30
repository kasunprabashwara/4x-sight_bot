# src/utils.py
import pandas as pd

def get_pair_price_from_row(row, pair, base_currency):
    """
    Calculates the exchange rate for a given pair from a row of data
    containing prices quoted against a common base currency.
    This is robust to direct (e.g., EURUSD) and inverse (e.g., USDJPY) pairs.
    
    :param row: A pandas Series (a row from a DataFrame) where keys are symbol names.
    :param pair: A tuple of two currency strings, e.g., ("EUR", "JPY").
    :param base_currency: The base currency for the prices in the row (e.g., "USD").
    :return: The exchange rate for the pair (Price of A / Price of B).
    """
    A, B = pair
    if A == B:
        return 1.0

    # Inner function to safely get the price of a single currency against the base currency.
    # It checks for both direct (CURRENCY/BASE) and inverse (BASE/CURRENCY) symbols.
    def get_price_in_base(currency, data_row):
        if currency == base_currency:
            return 1.0
        
        direct_symbol = f"{currency}{base_currency}"   # e.g., EURUSD
        inverse_symbol = f"{base_currency}{currency}"  # e.g., USDJPY

        if direct_symbol in data_row.index and not pd.isna(data_row[direct_symbol]):
            return data_row[direct_symbol]
        elif inverse_symbol in data_row.index and not pd.isna(data_row[inverse_symbol]):
            price = data_row[inverse_symbol]
            return 1.0 / price if price != 0 else 0
        else:
            # This error means the initial data fetch failed to get a required symbol.
            raise KeyError(f"Price for '{currency}' against base '{base_currency}' not found in the provided data row. Available symbols: {list(data_row.index)}")

    try:
        # Get the value of both currencies in the common base currency
        price_A_in_base = get_price_in_base(A, row)
        price_B_in_base = get_price_in_base(B, row)
    except KeyError as e:
        print(f"Error calculating cross-rate for pair ({A}/{B}): {e}")
        return 0 # Return 0 for the price if data is missing, preventing a crash

    # Calculate the cross rate: (A/base) / (B/base) = A/B
    if price_B_in_base == 0:
        return 0 # Avoid division by zero
        
    return price_A_in_base / price_B_in_base