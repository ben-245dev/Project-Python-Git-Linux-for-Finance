
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm  # prêt pour extensions VaR
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

def _make_series_stationary(close: pd.Series):
    series = close.asfreq("D").ffill()
    log_returns = np.log(series).diff().dropna()
    try:
        _ = adfuller(log_returns)[1]
    except Exception:
        pass
    return log_returns, series


def _select_arima_order(y: pd.Series, max_p: int = 3, max_q: int = 3, d: int = 0):
    best_aic = np.inf
    best_order = None
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(y, order=(p, d, q))
                    res = model.fit()
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = (p, d, q)
            except Exception:
                continue
    if best_order is None:
        best_order = (1, d, 1)
    return best_order, best_aic

def arima_forecast(
    close: pd.Series,
    horizon: int = 30,
    max_p: int = 3,
    max_q: int = 3,
    d: int = 0,
):
    if len(close) < 50:
        raise ValueError("Pas assez de points pour un ARIMA robuste (min ~50).")

    y, price_series = _make_series_stationary(close)
    best_order, best_aic = _select_arima_order(y, max_p=max_p, max_q=max_q, d=d)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(y, order=best_order)
        fit = model.fit()

    res = fit.get_forecast(steps=horizon)
    y_pred = res.predicted_mean
    ci = res.conf_int()

    last_price = float(price_series.iloc[-1])
    cum_log_ret = y_pred.cumsum()
    price_forecast = last_price * np.exp(cum_log_ret)

    cum_low = ci.iloc[:, 0].cumsum()
    cum_high = ci.iloc[:, 1].cumsum()
    price_low = last_price * np.exp(cum_low)
    price_high = last_price * np.exp(cum_high)

    forecast_df = pd.DataFrame(
        {
            "y_pred": price_forecast,
            "ci_low": price_low,
            "ci_high": price_high,
            "best_p": best_order[0],
            "best_d": best_order[1],
            "best_q": best_order[2],
            "aic": best_aic,
        },
        index=y_pred.index,
    )
    return forecast_df


def linear_regression_forecast(close: pd.Series, horizon: int = 30):
    df = pd.DataFrame({"y": close.values})
    df["t"] = np.arange(len(df))
    X = df[["t"]].values
    y = df["y"].values
    model = LinearRegression()
    model.fit(X, y)

    t_future = np.arange(len(df), len(df) + horizon)
    X_future = t_future.reshape(-1, 1)
    y_pred = model.predict(X_future)

    residuals = y - model.predict(X)
    sigma = residuals.std()
    ci_upper = y_pred + 1.96 * sigma
    ci_lower = y_pred - 1.96 * sigma

    hist_index = close.index
    freq = pd.infer_freq(hist_index) or "D"
    future_index = pd.date_range(
        start=hist_index[-1] + pd.tseries.frequencies.to_offset(freq),
        periods=horizon,
        freq=freq,
    )

    forecast_df = pd.DataFrame(
        {
            "y_pred": y_pred,
            "ci_low": ci_lower,
            "ci_high": ci_upper,
        },
        index=future_index,
    )
    return forecast_df