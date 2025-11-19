import pandas as pd
import os

def load_from_single_csv(path):
    """
    Load a CSV containing multiple assets in columns.
    Expected format:
        date, asset1, asset2, ...
    """
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df = df.sort_index()
    df = df.dropna()
    return df


def load_from_multiple_csv(folder_path):
    """
    Load several CSV files (one per asset) and merge them.
    Expected format inside each CSV:
        date, price
    """
    prices = {}

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            asset = file.replace(".csv", "")
            file_path = os.path.join(folder_path, file)

            df = pd.read_csv(file_path, parse_dates=["date"], index_col="date")
            df = df.sort_index()
            df = df.dropna()

            # assume the price is in the first column
            prices[asset] = df.iloc[:, 0]

    merged_df = pd.DataFrame(prices)
    merged_df = merged_df.dropna()

    return merged_df


def load_data_auto(path):
    """
    Automatically detect if the input path is:
    - a single CSV file
    - a folder containing several CSV files
    """
    if os.path.isfile(path):
        return load_from_single_csv(path)

    elif os.path.isdir(path):
        return load_from_multiple_csv(path)

    else:
        raise ValueError("Path must be a CSV file or a folder containing CSVs.")
