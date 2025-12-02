import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import pytz
import streamlit as st
import yfinance as yf

from data import get_live_index_data, load_ticker_data, get_price_series
from chart import price_chart
from metrics import compute_drawdown
from ui import render_trading_clocks, render_banner
from export import render_export_button

def page_home():
    st.markdown(
        """
        <h1 style="color:#FFFFFF;">Dashboard de trading</h1>
        <p style="color:#AAAAAA;">Surveillance des marchés, visualisation des prix et analyse des risques.</p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.subheader("Indices majeurs en direct")
    live_data = get_live_index_data()
    render_trading_clocks()
    render_banner(live_data)
    st.markdown("---")

    st.sidebar.header("Paramètres backtest (Accueil)")
    ticker = st.sidebar.text_input("Ticker (ex: AAPL, MSFT, ^GSPC)", value="AAPL", key="home_ticker")
    period = st.sidebar.selectbox(
        "Période de backtest",
        ["6mo", "1y", "3y", "5y", "max"],
        index=1,
        key="home_period",
    )

    run_backtest = st.sidebar.button("Mettre à jour le backtest", key="home_run_bt")

    if "home_last_ticker" not in st.session_state:
        st.session_state.home_last_ticker = ticker
    if "home_last_period" not in st.session_state:
        st.session_state.home_last_period = period

    if run_backtest:
        st.session_state.home_last_ticker = ticker
        st.session_state.home_last_period = period

    ticker_used = st.session_state.home_last_ticker
    period_used = st.session_state.home_last_period
    
    st.write(f"Backtest en cours sur: {ticker_used} / {period_used}")

    chart_type = st.sidebar.selectbox("Type de graphique", ["Bougies", "Courbes"], key="home_chart_type")
    quantile = st.sidebar.slider(
        "Quantile pour la VaR (historique)",
        min_value=0.90,
        max_value=0.99,
        value=0.95,
        step=0.01,
        key="home_var_q",
    )

    if ticker_used:
        df = load_ticker_data(ticker_used, period_used)
        if df.empty:
            st.warning("Pas de données téléchargées. Vérifie le ticker ou la période.")
            return

        st.subheader(f"Cours de {ticker_used}")
        price_chart(df, ticker_used, chart_type)

        close = get_price_series(df, "Close", ticker_used)
        returns = close.pct_change().dropna()
        if returns.empty:
            st.warning("Pas assez de données pour les stats.")
            return

        cum_ret_series = (1 + returns).cumprod() - 1
        cum_ret = float(cum_ret_series.iloc[-1])
        dd_series, max_dd = compute_drawdown(close)
        mean_ret = float(returns.mean())
        var_value = float(returns.quantile(1 - quantile))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rendement cumulé", f"{cum_ret*100:.2f} %")
        with col2:
            st.metric("Drawdown max", f"{max_dd*100:.2f} %")
        with col3:
            st.metric("Rendement moyen", f"{mean_ret*100:.2f} %")

        st.markdown("---")
        st.subheader("Distribution des rendements")

        fig = go.Figure(
            data=[
                go.Histogram(
                    x=returns.values,
                    nbinsx=50,
                    marker_color="#00c3ff",
                )
            ]
        )
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Rendement",
            yaxis_title="Fréquence",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**VaR historique {quantile*100:.1f}%** : {var_value*100:.2f} %")
        st.subheader(f"Exporter l’historique de {ticker}")
        render_export_button(df, filename=f"historique_{ticker}.csv")
