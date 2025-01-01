import pandas as pd

def get_forex_data():
    """
    Loads and processes the foreign exchange rate dataset, returning the EUR/USD and JPY/USD DataFrame.
    
    Returns:
        pd.DataFrame: A DataFrame containing the EUR/USD and JPY/USD exchange rates.
    """
    # Load the dataset
    data_set = pd.read_csv('data/Foreign_Exchange_Rates.csv', na_values='ND')

    # Interpolate missing values to handle missing data
    data_set = data_set.infer_objects(copy=False)  # Ensure non-numeric columns are correctly inferred
    data_set.interpolate(inplace=True)

    # Select only the columns for EUR/USD and JPY/USD exchange rates
    df = data_set[['EURO AREA - EURO/US$', 'JAPAN - YEN/US$']].copy()
    df['YEN/EURO'] = df['JAPAN - YEN/US$'] / df['EURO AREA - EURO/US$']
    return df

get_forex_data()
