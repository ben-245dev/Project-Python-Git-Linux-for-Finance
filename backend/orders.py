import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, time
import pytz
import streamlit as st

DB_FILE = "paper_trading.db"

def init_db():
    """Initialise la table des ordres si elle n'existe pas."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            ticker TEXT,
            action TEXT,
            quantity REAL,
            price REAL,
            fees REAL
        )
    ''')
    conn.commit()
    conn.close()

def is_market_open(ticker):
    """
   Check if the market is open for the given ticker.
   Returns (True, "") if open, else (False, "Reason").
    """
    ticker = ticker.upper()
    
    if "-USD" in ticker:
        return True, ""

    try:
        ny_tz = pytz.timezone('America/New_York')
        now_ny = datetime.now(ny_tz)
        
        if now_ny.weekday() >= 5:
            return False, "Le marché (US) est fermé le week-end."

        market_open = time(9, 30)
        market_close = time(16, 0)
        current_time = now_ny.time()

        if current_time < market_open or current_time > market_close:
            return False, f"Le marché (US) est fermé. Heure NY : {current_time.strftime('%H:%M')} (Ouverture : 09:30-16:00)."
            
    except Exception as e:
        print(f"Erreur timezone: {e}")

    return True, ""

def place_order(ticker, action, quantity, price):
    """Save an order in the database after checking market status."""
    
    # 1. Check if the market is open
    open_status, msg = is_market_open(ticker)
    if not open_status:
        return False, msg  # Return False and the explanatory error message

    # 2. Save in database
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fees = quantity * price * 0.001  # Simulation of fees (0.1%)
        
        c.execute(
            "INSERT INTO orders (date, ticker, action, quantity, price, fees) VALUES (?, ?, ?, ?, ?, ?)",
            (date_str, ticker, action, quantity, price, fees)
        )
        conn.commit()
        conn.close()
        return True, "Orders executed successfully."
    except Exception as e:
        return False, f"Database error: {e}"

def get_transaction_history():
    """Get history of all transactions."""
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query("SELECT * FROM orders ORDER BY date DESC", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df

def get_portfolio_positions():
    """
    Actual portfolio positions based on transaction history.
    Calculates average price (PRU) and total invested amount.
    """
    df = get_transaction_history()
    if df.empty:
        return pd.DataFrame()

    positions = {}

    for _, row in df.iterrows():
        tick = row['ticker']
        qty = row['quantity']
        price = row['price']
        action = row['action']

        if tick not in positions:
            positions[tick] = {"quantity": 0, "total_cost": 0, "avg_price": 0}

        if action == "ACHAT":
            positions[tick]["quantity"] += qty
            positions[tick]["total_cost"] += (qty * price)
        elif action == "VENTE":
            positions[tick]["quantity"] -= qty
            if positions[tick]["quantity"] > 0:
                positions[tick]["total_cost"] = positions[tick]["quantity"] * positions[tick]["avg_price"]
            else:
                positions[tick]["total_cost"] = 0

        # PRU
        if positions[tick]["quantity"] > 0:
            positions[tick]["avg_price"] = positions[tick]["total_cost"] / positions[tick]["quantity"]
        else:
            positions[tick]["avg_price"] = 0

   
    pos_list = []
    for tick, data in positions.items():
        if data["quantity"] > 0.0001:  # Keeping only positive positions
            pos_list.append({
                "Ticker": tick,
                "Quantité": data["quantity"],
                "PRU (Prix Moyen)": data["avg_price"],
                "Investi": data["total_cost"]
            })

    return pd.DataFrame(pos_list)

def get_current_prices(tickers_list):
    """
    Get actual prices 
    Handles the mix of Crypto (24/7) and Stocks (closed on weekends).
    """
    if not tickers_list:
        return {}
    try:
        data = yf.download(tickers_list, period="5d", interval="15m", progress=False)["Close"]
        
        if data.empty: return {}
        
        # one ticker
        if isinstance(data, pd.Series):
            return {tickers_list[0]: float(data.dropna().iloc[-1])}
        
        # Cas 2 : several tickers (DataFrame)
        # ffill() fill the holes (weekends for stocks)
        last_prices = data.ffill().iloc[-1]
        
        return last_prices.to_dict()
    except Exception as e:
        print(f"Erreur pricing: {e}")
        return {}

def init_settings_db():
    """Create a table to store user preferences (email)."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_user_setting(key, value):
    """Save a user setting."""
    init_settings_db() # Security
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()

def get_user_setting(key):
    """Get a user setting."""
    init_settings_db()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT value FROM settings WHERE key=?", (key,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None
