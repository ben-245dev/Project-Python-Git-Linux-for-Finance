from .base import TradingStrategy
import numpy as np
import pandas as pd

class MomentumStrategy(TradingStrategy):
    name = "Momentum (SMA Fast/SMA Slow)"
    def __init__(self, sma_fast=20, sma_slow=50, threshold=0, price_type="Close"):
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.threshold = threshold
        self.price_type = price_type

    def generate_signals(self, df, price_col):
        df['SMA_FAST'] = df[price_col].rolling(self.sma_fast).mean()
        df['SMA_SLOW'] = df[price_col].rolling(self.sma_slow).mean()
        df['momentum'] = (df['SMA_FAST'] - df['SMA_SLOW']) / df['SMA_SLOW'] * 100
        df['signal'] = (df['momentum'] > self.threshold).astype(int).shift(1, fill_value=0)
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
