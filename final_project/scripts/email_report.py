import os
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import yfinance as yf
import datetime

# Import backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.orders import get_portfolio_positions, get_user_setting

# CONFIGURATION
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")


def get_market_data(positions_df):
    if positions_df.empty: return positions_df, 0.0, 0.0
    tickers = positions_df["Ticker"].tolist()
    data = yf.download(tickers, period="5d", interval="1d", progress=False)["Close"]
    
    positions_df["Prix Actuel"] = 0.0
    positions_df["Variation Day (%)"] = 0.0
    positions_df["PnL Day ($)"] = 0.0
    
    total_day_pnl = 0.0
    total_value = 0.0
    
    for index, row in positions_df.iterrows():
        ticker = row["Ticker"]
        qty = row["Quantité"]
        try:
            series = data[ticker] if isinstance(data, pd.DataFrame) else data
            series = series.dropna()
            
            if len(series) >= 2:
                last = float(series.iloc[-1])
                prev = float(series.iloc[-2])
                positions_df.at[index, "Prix Actuel"] = last
                positions_df.at[index, "Variation Day (%)"] = (last - prev) / prev
                positions_df.at[index, "PnL Day ($)"] = (last - prev) * qty
                total_day_pnl += (last - prev) * qty
                total_value += last * qty
        except: pass
        
    return positions_df, total_value, total_day_pnl
