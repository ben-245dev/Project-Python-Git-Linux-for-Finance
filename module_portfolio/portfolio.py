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
from strategy import build_strategies
from metrics import compute_strategy_metrics

def page_portfolio():
    st.title("📊 Module B : Analyse de Portefeuille Multi-Actifs")
    st.markdown("---")

    st.subheader("Configuration des Actifs et des Poids")

    N_assets = st.selectbox(
        "Choisissez le nombre d'actifs dans le portefeuille (min. 3) :",
        options=list(range(3, 11)),
        index=0,
        key="port_n_assets",
    )

    TICKER_OPTIONS = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "NVDA",
        "TSLA",
        "AMZN",
        "BTC-USD",
        "ETH-USD",
        "EURUSD=X",
        "GC=F",
    ]

    asset_names = []
    weights_input = []

    col_names = st.columns(N_assets)
    col_weights = st.columns(N_assets)

    for i in range(N_assets):
        name = col_names[i].selectbox(
            f"Actif #{i+1}",
            options=TICKER_OPTIONS,
            index=i % len(TICKER_OPTIONS),
            key=f"port_asset_{i}",
        )
        asset_names.append(name)

        weight = col_weights[i].number_input(
            f"Poids {name}",
            min_value=-1.0,
            max_value=2.0,
            value=1.0 / N_assets,
            step=0.01,
            format="%.2f",
            key=f"port_weight_{i}",
        )
        weights_input.append(weight)

    st.markdown("---")
    col_date_start, col_date_end, col_button = st.columns(3)
    start_date = col_date_start.date_input(
        "Date de début", pd.to_datetime("2023-01-01"), key="port_start"
    )
    end_date = col_date_end.date_input(
        "Date de fin", pd.to_datetime("today"), key="port_end"
    )

    # Paramètres financiers dans la sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Paramètres Financiers (Portefeuille)")
    initial_investment = st.sidebar.number_input(
        "Investissement initial (€)",
        min_value=100.0,
        value=10000.0,
        step=500.0,
        key="port_init_inv",
    )

    risk_free_rate = (
        st.sidebar.number_input(
            "Taux sans risque annuel (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            key="port_rf",
        )
        / 100
    )
