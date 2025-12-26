import pandas as pd
import pandas_ta as ta
import numpy as np

def build_strategies(close: pd.Series, params: dict):
    # Conversion Series -> DataFrame pour pandas-ta
    df = close.to_frame(name="Close")
    
    # --- Récupération des Paramètres (avec valeurs par défaut) ---
    # Moyennes Mobiles
    fast_ma_len = params.get("fast_ma", 20)
    slow_ma_len = params.get("slow_ma", 50)
    
    # RSI
    rsi_len = params.get("rsi_len", 14)
    rsi_buy_level = params.get("rsi_buy", 30)
    rsi_sell_level = params.get("rsi_sell", 70)
    
    # MACD
    macd_fast = params.get("macd_fast", 12)
    macd_slow = params.get("macd_slow", 26)
    macd_sig = params.get("macd_sig", 9)
    
    # Bollinger
    bb_len = params.get("bb_len", 20)
    bb_std = params.get("bb_std", 2.0)
    
    # Adaptive
    target_vol = params.get("target_vol", 0.02)

    # --- Calcul des Indicateurs ---
    
    # 1. EMA
    df.ta.ema(length=fast_ma_len, append=True, col_names=("EMA_Fast",))
    df.ta.ema(length=slow_ma_len, append=True, col_names=("EMA_Slow",))
    
    # 2. RSI
    # Note : pandas_ta nomme souvent la colonne RSI_14, on force le nom pour être sûr
    rsi_series = df.ta.rsi(length=rsi_len)
    if rsi_series is not None:
        df["RSI"] = rsi_series
    
    # 3. MACD
    macd_df = df.ta.macd(fast=macd_fast, slow=macd_slow, signal=macd_sig)
    if macd_df is not None:
        macd_df.columns = ["MACD_Line", "MACD_Hist", "MACD_Signal"]
        df = pd.concat([df, macd_df], axis=1)
    
    # 4. Bollinger Bands
    bb_df = df.ta.bbands(length=bb_len, std=bb_std)
    if bb_df is not None:
        df = pd.concat([df, bb_df], axis=1)
        col_lower, col_mid, col_upper = bb_df.columns[0], bb_df.columns[1], bb_df.columns[2]
    else:
        col_lower, col_mid, col_upper = "BBL", "BBM", "BBU"

    # ATR (Approximation)
    df["Vol_ATR_Pct"] = df["Close"].pct_change().rolling(14).std()

    market_returns = df["Close"].pct_change().fillna(0.0)
    strategies = {}

    # --- 1. Buy & Hold ---
    strategies["Buy & Hold"] = {
        "returns": market_returns,
        "equity": df["Close"] / df["Close"].iloc[0],
        "signals": pd.Series(1, index=df.index) 
    }

    # --- 2. MA Crossover ---
    if "EMA_Fast" in df.columns and "EMA_Slow" in df.columns:
        signal_ma = (df["EMA_Fast"] > df["EMA_Slow"]).astype(int)
        position_ma = signal_ma.shift(1).fillna(0)
        strat_ret_ma = position_ma * market_returns
        strategies["MA Crossover"] = {
            "returns": strat_ret_ma,
            "equity": (1 + strat_ret_ma).cumprod(),
            "signals": signal_ma.diff().fillna(0)
        }
    else:
        strategies["MA Crossover"] = strategies["Buy & Hold"]

    # --- 3. RSI Reversion ---
    if "RSI" in df.columns:
        rsi = df["RSI"]
        signal_rsi = pd.Series(np.nan, index=df.index)
        signal_rsi[rsi < rsi_buy_level] = 1 
        signal_rsi[rsi > rsi_sell_level] = 0
        signal_rsi = signal_rsi.ffill().fillna(0)
        position_rsi = signal_rsi.shift(1).fillna(0)
        strat_ret_rsi = position_rsi * market_returns
        strategies["RSI Reversion"] = {
            "returns": strat_ret_rsi,
            "equity": (1 + strat_ret_rsi).cumprod(),
            "signals": signal_rsi.diff().fillna(0)
        }
    else:
        strategies["RSI Reversion"] = strategies["Buy & Hold"]

    # --- 4. MACD Trend ---
    if "MACD_Line" in df.columns:
        signal_macd = (df["MACD_Line"] > df["MACD_Signal"]).astype(int)
        position_macd = signal_macd.shift(1).fillna(0)
        strat_ret_macd = position_macd * market_returns
        strategies["MACD Trend"] = {
            "returns": strat_ret_macd,
            "equity": (1 + strat_ret_macd).cumprod(),
            "signals": signal_macd.diff().fillna(0)
        }
    else:
        strategies["MACD Trend"] = strategies["Buy & Hold"]

    # --- 5. Bollinger Breakout ---
    if bb_df is not None:
        close_price = df["Close"]
        bb_upper_series = df[col_upper]
        bb_mid_series = df[col_mid]
        ema_slow_series = df["EMA_Slow"] if "EMA_Slow" in df.columns else df["Close"]
        
        signal_bb = pd.Series(np.nan, index=df.index)
        
        # Conditions paramétrables implicitement via EMA_Slow et BB settings
        buy_condition = (close_price > bb_upper_series) & (close_price > ema_slow_series)
        sell_condition = (close_price < bb_mid_series)
        
        signal_bb[buy_condition] = 1 
        signal_bb[sell_condition] = 0
        
        signal_bb = signal_bb.ffill().fillna(0)
        position_bb = signal_bb.shift(1).fillna(0)
        strat_ret_bb = position_bb * market_returns
        
        strategies["Bollinger Breakout"] = {
            "returns": strat_ret_bb,
            "equity": (1 + strat_ret_bb).cumprod(),
            "signals": signal_bb.diff().fillna(0)
        }
    else:
        strategies["Bollinger Breakout"] = strategies["Buy & Hold"]

    # --- 6. Adaptive Trend ---
    if "EMA_Fast" in df.columns and "RSI" in df.columns:
        direction = pd.Series(0, index=df.index)
        cond_long = (df["EMA_Fast"] > df["EMA_Slow"]) & (df["RSI"] > 50)
        cond_short = (df["EMA_Fast"] < df["EMA_Slow"]) & (df["RSI"] < 50)
        direction[cond_long] = 1
        direction[cond_short] = -1
        
        real_vol = df["Vol_ATR_Pct"].replace(0, 0.01)
        leverage = (target_vol / real_vol).clip(0.1, 2.0)
        
        final_position = (direction * leverage).shift(1).fillna(0)
        strat_ret_complex = final_position * market_returns
        
        strategies["Adaptive Trend (Long/Short)"] = {
            "returns": strat_ret_complex,
            "equity": (1 + strat_ret_complex).cumprod(),
            "signals": direction.diff().fillna(0),
            "leverage": leverage
        }
    else:
        strategies["Adaptive Trend (Long/Short)"] = strategies["Buy & Hold"]

    return strategies, df