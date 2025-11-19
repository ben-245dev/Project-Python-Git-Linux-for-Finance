import pandas as pd
from pandas_datareader import data as pdr


def load_temp_data():
    """
    Temporary data loader for development of the Quant B module.
    This will later be replaced by the official Quant A data scraper.

    Returns
    -------
    DataFrame
        A clean dataframe of adjusted close prices for several assets.
    """
    assets = ["AAPL", "MSFT", "NVDA"]

    # Fetch historical prices from Yahoo Finance
    df = pdr.get_data_yahoo(assets, start="2023-01-01")["Adj Close"]

    # Remove missing values
    df = df.dropna()

    return df
