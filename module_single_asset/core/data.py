import yfinance as yf
import pandas as pd


def fetch_data(ticker, start, end):
    # Récupération via yfinance
    df = yf.download(ticker, start=start, end=end)
    # Filtrage stricte et réindexation:
    desired_cols = ["Open", "High", "Low", "Close", "Volume"]
    actual_cols = [col for col in desired_cols if col in df.columns]
    df = df[actual_cols]
    # Ajout de colonnes absentes avec des valeurs None (pour éviter tout plantage/KeyError)
    for col in desired_cols:
        if col not in df.columns:
            df[col] = None
    # Dates en colonne, pas en index, pour cohérence graphique
    df = df.reset_index()
    return df


def detect_col(df, ticker, label):
    # Mono-ticker = simple; si multi-ticker, adapter à ta structure
    if label in df.columns:
        return label
    else:
        # Parfois le label peut venir en format f"{ticker}_{label}" (ex multi-ticker)
        label2 = f"{ticker}_{label}"
        if label2 in df.columns:
            return label2
    return None
