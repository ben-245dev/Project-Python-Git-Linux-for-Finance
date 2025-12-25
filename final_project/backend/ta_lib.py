import pandas_ta as ta
import pandas as pd

def add_technical_indicators(df: pd.DataFrame):
    """Ajoute RSI, MACD et Bollinger Bands au DataFrame"""
    # Copie pour éviter les warnings SettingWithCopy
    data = df.copy()
    
    # RSI
    data.ta.rsi(length=14, append=True)
    
    # MACD
    data.ta.macd(fast=12, slow=26, signal=9, append=True)
    
    # Bollinger Bands
    data.ta.bbands(length=20, std=2, append=True)
    
    return data