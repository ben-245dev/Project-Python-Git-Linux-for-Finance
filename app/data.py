from datetime import datetime

import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf

INDICES = {
    "S&P 500": "^GSPC",
    "DAX": "^GDAXI",
    "CAC 40": "^FCHI",
    "Nikkei 225": "^N225",
    "FTSE 100": "^FTSE",
    "Dow Jones": "^DJI",
}

def get_live_index_data():
    data = {}
    for name, ticker in INDICES.items():
        try:
            df = yf.download(ticker, period="1d", interval="1m", progress=False)
            if not df.empty:
                last_row = df.iloc[-1]
                price = float(last_row["Close"])
                ts_utc = (
                    last_row.name.tz_convert("UTC")
                    if last_row.name.tzinfo
                    else last_row.name.tz_localize("UTC")
                )
                data[name] = {
                    "price": round(price, 2),
                    "time_utc": ts_utc,
                }
            else:
                data[name] = None
        except Exception:
            data[name] = None
    return data

def load_ticker_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    return df.dropna()


def get_price_series(df: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    if isinstance(df.columns, pd.MultiIndex):
        if (field, ticker) in df.columns:
            return df[(field, ticker)]
        if field in df.columns.get_level_values(0):
            return df[field]
        raise KeyError(f"Colonne {(field, ticker)} ou {field} introuvable dans df.columns")
    else:
        if field not in df.columns:
            raise KeyError(f"Colonne {field} introuvable dans df.columns")
        return df[field]

