from .base import TradingStrategy
import numpy as np
import pandas as pd

class MeanReversionStrategy(TradingStrategy):
    name = "Mean Reversion (z-score)"
    def __init__(self, lookback=20, z_entry=1.5, z_exit=0.5, price_type="Close"):
        self.lookback = lookback
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.price_type = price_type

    def generate_signals(self, df, price_col):
        rolling_mean = df[price_col].rolling(self.lookback).mean()
        rolling_std = df[price_col].rolling(self.lookback).std()
        zscore = (df[price_col] - rolling_mean) / rolling_std
        df['zscore'] = zscore
        df['signal'] = 0
        position = 0
        for i in range(1, len(df)):
            if position == 0 and zscore.iloc[i] < -self.z_entry:
                position = 1
            elif position == 1 and abs(zscore.iloc[i]) < self.z_exit:
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
