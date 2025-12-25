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
    Vérifie si le marché est ouvert pour un ticker donné.
    - Crypto (contient '-USD') : Toujours ouvert (24/7).
    - Actions US : Ouvert lun-ven de 09h30 à 16h00 (Heure New York).
    """
    ticker = ticker.upper()
    
    # 1. Cas Crypto : Toujours ouvert
    if "-USD" in ticker:
        return True, ""

    # 2. Cas Actions US (Par défaut pour simplifier)
    # On se base sur le fuseau horaire de New York
    try:
        ny_tz = pytz.timezone('America/New_York')
        now_ny = datetime.now(ny_tz)
        
        # Vérification Week-end (Lundi=0, ..., Samedi=5, Dimanche=6)
        if now_ny.weekday() >= 5:
            return False, "Le marché (US) est fermé le week-end."

        # Vérification Horaires (09:30 - 16:00 NY)
        market_open = time(9, 30)
        market_close = time(16, 0)
        current_time = now_ny.time()

        if current_time < market_open or current_time > market_close:
            return False, f"Le marché (US) est fermé. Heure NY : {current_time.strftime('%H:%M')} (Ouverture : 09:30-16:00)."
            
    except Exception as e:
        # En cas d'erreur de timezone, on laisse passer (fail-open) ou on bloque, au choix.
        # Ici on log juste l'erreur
        print(f"Erreur timezone: {e}")

    return True, ""

def place_order(ticker, action, quantity, price):
    """Enregistre un ordre UNIQUEMENT si le marché est ouvert."""
    
    # 1. Vérification d'ouverture du marché
    open_status, msg = is_market_open(ticker)
    if not open_status:
        return False, msg  # On retourne False et le message d'erreur explicatif
    
    # 2. Enregistrement en base
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fees = quantity * price * 0.001  # Simulation de frais (0.1%)
        
        c.execute(
            "INSERT INTO orders (date, ticker, action, quantity, price, fees) VALUES (?, ?, ?, ?, ?, ?)",
            (date_str, ticker, action, quantity, price, fees)
        )
        conn.commit()
        conn.close()
        return True, "Ordre exécuté avec succès."
    except Exception as e:
        return False, f"Erreur base de données : {e}"

def get_transaction_history():
    """Récupère tout l'historique des transactions."""
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query("SELECT * FROM orders ORDER BY date DESC", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df

def get_portfolio_positions():
    """
    Calcule les positions actuelles (Quantité nette et Prix moyen pondéré).
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
            # Mise à jour du coût total proportionnellement
            if positions[tick]["quantity"] > 0:
                positions[tick]["total_cost"] = positions[tick]["quantity"] * positions[tick]["avg_price"]
            else:
                positions[tick]["total_cost"] = 0

        # Recalcul du PRU (Prix de Revient Unitaire)
        if positions[tick]["quantity"] > 0:
            positions[tick]["avg_price"] = positions[tick]["total_cost"] / positions[tick]["quantity"]
        else:
            positions[tick]["avg_price"] = 0

    # Conversion en DataFrame
    pos_list = []
    for tick, data in positions.items():
        if data["quantity"] > 0.0001:  # On ne garde que les positions ouvertes
            pos_list.append({
                "Ticker": tick,
                "Quantité": data["quantity"],
                "PRU (Prix Moyen)": data["avg_price"],
                "Investi": data["total_cost"]
            })

    return pd.DataFrame(pos_list)

def get_current_prices(tickers_list):
    """
    Récupère les prix actuels de manière robuste.
    Gère le mélange Crypto (24/7) et Actions (Fermé le week-end).
    """
    if not tickers_list:
        return {}
    try:
        # On demande 5 jours pour être sûr d'avoir la dernière clôture (cas du week-end)
        data = yf.download(tickers_list, period="5d", interval="15m", progress=False)["Close"]
        
        if data.empty: return {}
        
        # Cas 1 : Un seul ticker (Series)
        if isinstance(data, pd.Series):
            return {tickers_list[0]: float(data.dropna().iloc[-1])}
        
        # Cas 2 : Plusieurs tickers (DataFrame)
        # ffill() remplit les trous (le prix du vendredi est propagé au samedi/dimanche)
        last_prices = data.ffill().iloc[-1]
        
        return last_prices.to_dict()
    except Exception as e:
        print(f"Erreur pricing: {e}")
        return {}