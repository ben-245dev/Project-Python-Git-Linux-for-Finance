import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import pytz
import streamlit as st
import yfinance as yf
from scipy.stats import norm  # prêt pour extensions VaR
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from strategy import page_strategy  # strategy.py
from portfolio import page_portfolio  # portfolio.py*
from home import page_home  # home.py
from chart import price_chart
from data import get_price_series

# -----------------------------
# Config générale
# -----------------------------
st.set_page_config(
    page_title="Trading Dashboard",
    layout="wide",
)

# -----------------------------
# Constantes communes
# -----------------------------
INDICES = {
    "S&P 500": "^GSPC",
    "DAX": "^GDAXI",
    "CAC 40": "^FCHI",
    "Nikkei 225": "^N225",
    "FTSE 100": "^FTSE",
    "Dow Jones": "^DJI",
}

TIMEZONES = {
    "New York": "America/New_York",
    "Londres": "Europe/London",
    "Paris": "Europe/Paris",
    "Tokyo": "Asia/Tokyo",
}

TRADING_TIMEZONES = {
    "New York": "America/New_York",
    "London": "Europe/London",
    "Paris": "Europe/Paris",
    "Tokyo": "Asia/Tokyo",
    "Sydney": "Australia/Sydney",
}


def main():
    st.sidebar.title("Navigation")
    if st.sidebar.button("🔄 Recharger l'application"):
        st.rerun()

    page = st.sidebar.radio(
        "Aller à",
        ["Accueil", "Stratégie de trading et backtest", "Portefeuille multi-actifs"],
        index=0,
    )

    if page == "Accueil":
        page_home()
    elif page == "Stratégie de trading et backtest":
        page_strategy()
    else:
        page_portfolio()


if __name__ == "__main__":
    main()
