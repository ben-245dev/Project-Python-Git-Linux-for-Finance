import numpy as np
import streamlit as st
import plotly.graph_objects as go

def summarize_perf(curve):
    perf = (curve.iloc[-1] - 1) * 100
    drawdown = ((curve / curve.cummax()) - 1).min() * 100
    return perf, drawdown

def plot_equity_curve(st, df, strat_curve, bnh_curve, strat_name):
    fig_eq = go.Figure([
        go.Scatter(x=df['Date'], y=strat_curve, name=strat_name, line=dict(color="blue")),
        go.Scatter(x=df['Date'], y=bnh_curve, name="Buy & Hold", line=dict(color="black", dash="dot"))
    ])
    st.plotly_chart(fig_eq, use_container_width=True)

def plot_trade_markers(df, price_col, entry_col='signal'):
    entries = df.index[df[entry_col].diff().fillna(0) == 1]
    exits   = df.index[df[entry_col].diff().fillna(0) == -1]
    entry_dates = df['Date'].iloc[entries] if 'Date' in df.columns else df.index[entries]
    exit_dates  = df['Date'].iloc[exits]   if 'Date' in df.columns else df.index[exits]
    entry_prices= df[price_col].iloc[entries]
    exit_prices = df[price_col].iloc[exits]
    return entry_dates, entry_prices, exit_dates, exit_prices

def plot_signals_overlay(st, df, price_col, strat_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df[price_col], name="Prix", line=dict(color='grey')))
    entry_dates, entry_prices, exit_dates, exit_prices = plot_trade_markers(df, price_col, entry_col='signal')
    fig.add_trace(go.Scatter(x=entry_dates, y=entry_prices, mode="markers", marker=dict(color="green", size=8, symbol="triangle-up"), name="Entrée"))
    fig.add_trace(go.Scatter(x=exit_dates, y=exit_prices, mode="markers", marker=dict(color="red", size=8, symbol="triangle-down"), name="Sortie"))
    st.plotly_chart(fig, use_container_width=True)
