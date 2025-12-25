from .base import TradingStrategy
import numpy as np
import pandas as pd

class BreakoutStrategy(TradingStrategy):
    name = "High-Low Breakout"
    def __init__(self, lookback=20, price_type="Close"):
        self.lookback = lookback
        self.price_type = price_type

    def generate_signals(self, df, price_col):
        high_max = df[price_col].rolling(self.lookback).max()
        low_min  = df[price_col].rolling(self.lookback).min()
        df['signal'] = 0
        position = 0
        for i in range(1, len(df)):
            price = df[price_col].iloc[i]
            if price > high_max.iloc[i-1]:
                position = 1
            elif price < low_min.iloc[i-1]:
                position = 0
            df['signal'].iloc[i] = position
        return df

    def compute_equity_curve(self, df, price_col):
        returns = df[price_col].pct_change().fillna(0)
        curve = np.ones(len(df))
        for i in range(1, len(df)):
            if df['signal'].iloc[i]:
                curve[i] = curve[i-1] * (1+returns.iloc[i])
            else:
                curve[i] = curve[i-1]
        return pd.Series(curve, index=df.index)
