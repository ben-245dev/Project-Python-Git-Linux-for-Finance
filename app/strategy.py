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
from data import get_price_series, load_ticker_data
from metrics import compute_strategy_metrics
from forecast import arima_forecast


def build_strategies(close: pd.Series, params: dict):
    prices = close
    rets = prices.pct_change().fillna(0.0)
    strategies = {}

    # 1) Buy & Hold
    bh_returns = rets.copy()
    bh_equity = (1 + bh_returns).cumprod()
    strategies["Buy & Hold"] = {
        "returns": bh_returns,
        "equity": bh_equity,
    }

    # 2) Momentum
    lb = params.get("mom_lookback", 20)
    mom_signal = (prices > prices.shift(lb)).astype(int)
    mom_returns = mom_signal.shift(1).fillna(0) * rets
    mom_equity = (1 + mom_returns).cumprod()
    strategies["Momentum"] = {
        "returns": mom_returns,
        "equity": mom_equity,
    }

    # 3) Mean Reversion
    mr_signal = (prices < prices.shift(lb)).astype(int)
    mr_returns = mr_signal.shift(1).fillna(0) * rets
    mr_equity = (1 + mr_returns).cumprod()
    strategies["Mean Reversion"] = {
        "returns": mr_returns,
        "equity": mr_equity,
    }

    # 4) Moving Average Crossover
    fast = params.get("fast_ma", 20)
    slow = params.get("slow_ma", 50)
    ma_fast = prices.rolling(fast).mean()
    ma_slow = prices.rolling(slow).mean()
    mac_signal = (ma_fast > ma_slow).astype(int)
    mac_returns = mac_signal.shift(1).fillna(0) * rets
    mac_equity = (1 + mac_returns).cumprod()
    strategies["MA Crossover"] = {
        "returns": mac_returns,
        "equity": mac_equity,
    }

    # 5) Breakout
    n_break = params.get("breakout_lookback", 50)
    rolling_max = prices.shift(1).rolling(n_break).max()
    bo_signal = (prices > rolling_max).astype(int)
    bo_returns = bo_signal.shift(1).fillna(0) * rets
    bo_equity = (1 + bo_returns).cumprod()
    strategies["Breakout"] = {
        "returns": bo_returns,
        "equity": bo_equity,
    }

    return strategies


def page_strategy():
    st.markdown(
        """
        <h1 style="color:#FFFFFF;">Stratégie de trading et backtest</h1>
        <p style="color:#AAAAAA;">
        Définis une stratégie, ajuste ses paramètres, observe les métriques de performance
        et une prévision des prix (ARIMA).
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Paramètres backtest (Stratégie)")
    ticker = st.sidebar.text_input("Ticker (ex: AAPL, MSFT, ^GSPC)", value="AAPL", key="strat_ticker")
    period = st.sidebar.selectbox(
        "Période de backtest",
        ["6mo", "1y", "3y", "5y", "max"],
        index=1,
        key="strat_period",
    )
    freq = st.sidebar.selectbox("Périodicité", ["Daily", "Weekly", "Monthly"], index=0, key="strat_freq")

    st.sidebar.subheader("Paramètres des stratégies")
    mom_lb = st.sidebar.slider("Momentum lookback (jours)", 5, 120, 20, step=5, key="strat_mom_lb")
    fast_ma = st.sidebar.slider("MA rapide (jours)", 5, 60, 20, step=5, key="strat_fast_ma")
    slow_ma = st.sidebar.slider("MA lente (jours)", 20, 200, 50, step=10, key="strat_slow_ma")
    breakout_lb = st.sidebar.slider("Breakout lookback (jours)", 20, 200, 50, step=10, key="strat_breakout")

    strategy_name = st.sidebar.selectbox(
        "Stratégie",
        ["Buy & Hold", "Momentum", "Mean Reversion", "MA Crossover", "Breakout"],
        key="strat_name",
    )

    horizon_forecast = st.sidebar.slider(
        "Horizon de prévision (jours)", 5, 90, 30, step=5, key="strat_horizon"
    )

    if not ticker:
        st.warning("Renseigne un ticker.")
        return

    df = load_ticker_data(ticker, period)
    if df.empty:
        st.warning("Pas de données pour ce ticker / période.")
        return

    close = get_price_series(df, "Close", ticker)

    if freq == "Weekly":
        close = close.resample("W").last()
    elif freq == "Monthly":
        close = close.resample("M").last()

    close = close.dropna()
    if close.empty:
        st.warning("Pas de données après agrégation.")
        return

    params = {
        "mom_lookback": mom_lb,
        "fast_ma": fast_ma,
        "slow_ma": slow_ma,
        "breakout_lookback": breakout_lb,
    }

    strategies = build_strategies(close, params)
    strat = strategies[strategy_name]
    strat_returns = strat["returns"]
    strat_equity = strat["equity"]

    equity_bh = strategies["Buy & Hold"]["equity"]

    st.subheader(f"Backtest - {strategy_name} sur {ticker}")
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=close.index,
            y=close.values,
            mode="lines",
            name="Prix",
            yaxis="y1",
            line=dict(color="#00c3ff"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=strat_equity.index,
            y=strat_equity.values,
            mode="lines",
            name="Equity stratégie",
            yaxis="y2",
            line=dict(color="#ff9900"),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(domain=[0.0, 1.0]),
        yaxis=dict(title="Prix", side="left"),
        yaxis2=dict(
            title="Equity (normalisée)",
            overlaying="y",
            side="right",
        ),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Métriques de performance")
    metrics = compute_strategy_metrics(strat_returns, strat_equity)
    bh_metrics = compute_strategy_metrics(strategies["Buy & Hold"]["returns"], equity_bh)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total return (strat)", f"{metrics['total_return']*100:.2f} %")
    with col2:
        st.metric("Max drawdown (strat)", f"{metrics['max_drawdown']*100:.2f} %")
    with col3:
        st.metric("Sharpe (strat)", f"{metrics['sharpe']:.2f}")
    with col4:
        st.metric("Vol annualisée (strat)", f"{metrics['vol']*100:.2f} %")

    col5, col6, col7 = st.columns(3)
    with col5:
        st.metric("Total return (BH)", f"{bh_metrics['total_return']*100:.2f} %")
    with col6:
        st.metric("Max drawdown (BH)", f"{bh_metrics['max_drawdown']*100:.2f} %")
    with col7:
        st.metric("Sharpe (BH)", f"{bh_metrics['sharpe']:.2f}")

    st.subheader("Distribution des rendements de la stratégie")
    fig_hist = go.Figure(
        data=[
            go.Histogram(
                x=strat_returns.values,
                nbinsx=50,
                marker_color="#ff9900",
            )
        ]
    )
    fig_hist.update_layout(
        template="plotly_dark",
        xaxis_title="Rendement",
        yaxis_title="Fréquence",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Prévision (ARIMA)")
    forecast_df = arima_forecast(close, horizon=horizon_forecast)
    st.caption(
        f"ARIMA(p,d,q) sélectionné: ({int(forecast_df['best_p'].iloc[0])}, "
        f"{int(forecast_df['best_d'].iloc[0])}, {int(forecast_df['best_q'].iloc[0])}) "
        f" | AIC = {forecast_df['aic'].iloc[0]:.2f}"
    )

    fig_forecast = go.Figure()
    fig_forecast.add_trace(
        go.Scatter(
            x=close.index,
            y=close.values,
            mode="lines",
            name="Historique",
            line=dict(color="#00c3ff"),
        )
    )
    fig_forecast.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df["y_pred"],
            mode="lines",
            name="Prévision",
            line=dict(color="#ff0000"),
        )
    )
    fig_forecast.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df["ci_high"],
            mode="lines",
            name="IC haut",
            line=dict(color="rgba(255,0,0,0.3)", dash="dot"),
            showlegend=False,
        )
    )
    fig_forecast.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df["ci_low"],
            mode="lines",
            name="IC bas",
            line=dict(color="rgba(255,0,0,0.3)", dash="dot"),
            fill="tonexty",
            fillcolor="rgba(255,0,0,0.1)",
            showlegend=False,
        )
    )

    fig_forecast.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Date",
        yaxis_title="Prix",
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
