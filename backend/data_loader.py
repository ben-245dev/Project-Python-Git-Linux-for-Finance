import pandas as pd
import yfinance as yf
import streamlit as st
from config import INDICES

@st.cache_data(ttl=60)  # 1 minute cache
def get_live_index_data():
    data = {}
    tickers = list(INDICES.values())
    try:
        df = yf.download(tickers, period="1d", interval="1m", progress=False)["Close"]
        if df.empty: return {}
        
        last_row = df.iloc[-1]
        timestamp = last_row.name
        
        for name, ticker in INDICES.items():
            if ticker in last_row:
                price = last_row[ticker]
                # Gestion NaN
                if pd.isna(price): continue
                data[name] = {"price": price, "time_utc": timestamp}
    except Exception:
        pass
    return data

@st.cache_data(ttl=3600) # 1 hour cache
def load_ticker_data(ticker: str, period: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False, multi_level_index=False)
        return df.dropna()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_batch_data(tickers: list, start_date, end_date) -> pd.DataFrame:
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)["Close"]
        return data.dropna()
    except Exception:
        return pd.DataFrame()

def get_price_series(df: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    """
    Get the price series for a specific field and ticker from a DataFrame.
    Handles both MultiIndex and single Index DataFrames.
    """
    if df.empty:
        return pd.Series()

    # Case 1 : MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        if (field, ticker) in df.columns:
            return df[(field, ticker)]
        if field in df.columns.get_level_values(0):
            return df[field]
    
    # Cas 2 : simple index
    else:
        if field in df.columns:
            return df[field]
        # Fallback
        if field == "Close" and len(df.columns) == 1:
            return df.iloc[:, 0]

    return pd.Series()