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

from data import get_price_series


def price_chart(df: pd.DataFrame, ticker: str, chart_type: str):
    if df.empty:
        st.warning("Pas de données disponibles pour ce ticker / cette période.")
        return

    if chart_type == "Bougies":
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df.index,
                    open=get_price_series(df, "Open", ticker),
                    high=get_price_series(df, "High", ticker),
                    low=get_price_series(df, "Low", ticker),
                    close=get_price_series(df, "Close", ticker),
                    name=ticker,
                )
            ]
        )
    else:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=df.index,
                    y=get_price_series(df, "Close", ticker),
                    mode="lines",
                    name=ticker,
                )
            ]
        )

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Date",
        yaxis_title="Prix",
    )
    st.plotly_chart(fig, use_container_width=True)
