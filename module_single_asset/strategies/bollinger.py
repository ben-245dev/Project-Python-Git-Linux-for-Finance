from .base import TradingStrategy
import numpy as np
import pandas as pd

class BollingerStrategy(TradingStrategy):
    name = "Bollinger Bands"
    def __init__(self, period=20, nb_std=2, price_type="Close"):
        self.period = period
        self.nb_std = nb_std
        self.price_type = price_type

    def generate_signals(self, df, price_col):
        sma = df[price_col].rolling(self.period).mean()
        std = df[price_col].rolling(self.period).std()
        df['BollUpper'] = sma + self.nb_std * std
        df['BollLower'] = sma - self.nb_std * std
        df['signal'] = 0
        in_trade = False
        for i in range(1, len(df)):
            if not in_trade and df[price_col].iloc[i] < df['BollLower'].iloc[i]:
                in_trade = True
            elif in_trade and df[price_col].iloc[i] > df['BollUpper'].iloc[i]:
                in_trade = False
            df['signal'].iloc[i] = int(in_trade)
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
