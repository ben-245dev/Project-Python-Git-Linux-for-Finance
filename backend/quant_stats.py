import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

def calculate_cointegration(series_a: pd.Series, series_b: pd.Series):
    """
    Testing for cointegration between two time series.
    Returns the p-value and hedge ratio.
    """
    df = pd.concat([series_a, series_b], axis=1).dropna()
    S1 = df.iloc[:, 0]
    S2 = df.iloc[:, 1]
    
    # Hedge ratio via OLS
    S1 = sm.add_constant(S1)
    results = sm.OLS(S2, S1).fit()
    hedge_ratio = results.params.iloc[1]
    
    # Cointegration test
    score, pvalue, _ = coint(df.iloc[:, 0], df.iloc[:, 1])
    
    return pvalue, hedge_ratio

def calculate_zscore(series: pd.Series, window=30):
    """
    Z-Score calculation over a rolling window.
    Z = (X - mean) / std
    """
    r_mean = series.rolling(window=window).mean()
    r_std = series.rolling(window=window).std()
    z_score = (series - r_mean) / r_std
    return z_score

def calculate_kelly_criterion(win_rate, win_loss_ratio):
    """
    Kelly formula: f = p - (q / b)
    p = probability of winning, q = probability of losing, b = win/loss ratio
    Returns the % of capital to invest.
    """
    if win_loss_ratio == 0: return 0
    kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
    return kelly