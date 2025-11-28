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

def compute_drawdown(equity: pd.Series):
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return dd, dd.min()


def compute_sharpe(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0):
    if returns.std() == 0 or returns.empty:
        return np.nan
    excess = returns - rf / periods_per_year
    return np.sqrt(periods_per_year) * excess.mean() / excess.std()


def compute_strategy_metrics(strategy_returns: pd.Series, equity: pd.Series):
    total_ret = equity.iloc[-1] - 1.0
    dd_series, max_dd = compute_drawdown(equity)
    sharpe = compute_sharpe(strategy_returns)
    vol = strategy_returns.std() * np.sqrt(252)
    return {
        "total_return": total_ret,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "vol": vol,
    }
