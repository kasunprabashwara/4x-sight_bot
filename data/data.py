import pandas as pd

def get_forex_data():
    # Load the dataset
    data_set = pd.read_csv('data\Foreign_Exchange_Rates.csv', na_values='ND', parse_dates=['Timestamp'])
    
    # Interpolate missing values to handle missing data
    data_set.interpolate(inplace=True)
    
    # Select only the required columns
    df = data_set[['Timestamp','EURUSD', 'GBPUSD','JPYUSD']].copy()
    
    
    return df
