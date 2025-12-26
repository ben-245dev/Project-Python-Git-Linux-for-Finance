import os
import sys
import datetime
import pandas as pd
import yfinance as yf
import numpy as np

# Configuration : Liste des actifs à surveiller dans le rapport
ASSETS = ["BTC-USD", "ETH-USD", "AAPL", "MSFT", "EURUSD=X"]
REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")

# Assurer que le dossier de rapport existe
os.makedirs(REPORT_DIR, exist_ok=True)

def compute_drawdown(series):
    running_max = series.cummax()
    dd = series / running_max - 1.0
    return dd.min()

def generate_daily_report():
    today = datetime.date.today()
    filename = os.path.join(REPORT_DIR, f"report_{today}.txt")
    
    print(f"Génération du rapport pour le {today}...")
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"=== RAPPORT JOURNALIER DE TRADING : {today} ===\n")
        f.write(f"Généré à : {datetime.datetime.now().strftime('%H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
        
        for ticker in ASSETS:
            try:
                # Téléchargement des données (2 jours pour avoir open/close fiable)
                # 'auto_adjust=True' pour des prix propres
                df = yf.download(ticker, period="2d", interval="1d", progress=False, auto_adjust=True)
                
                if df.empty:
                    f.write(f"[{ticker}] : Pas de données disponibles.\n")
                    continue
                
                # Récupération de la dernière journée complète
                last_candle = df.iloc[-1]
                
                # Si c'est un actif crypto (24/7), le 'Close' est le prix actuel
                # Si c'est une action, c'est le close d'hier ou d'aujourd'hui
                
                open_price = float(last_candle["Open"])
                close_price = float(last_candle["Close"])
                high_price = float(last_candle["High"])
                low_price = float(last_candle["Low"])
                
                # Variation
                change = close_price - open_price
                change_pct = (change / open_price) * 100
                
                # Volatilité (Range Journalier)
                volatility_day = ((high_price - low_price) / open_price) * 100
                
                # Max Drawdown Intraday (Approximation : Low vs Open)
                # Pour un calcul précis, il faudrait des données minute, 
                # ici on fait simple : (Low - High) / High
                dd_intraday = ((low_price - high_price) / high_price) * 100

                f.write(f"ACTIF : {ticker}\n")
                f.write(f"---------------------------\n")
                f.write(f"  Open:    {open_price:.2f}\n")
                f.write(f"  Close:   {close_price:.2f}\n")
                f.write(f"  Perf:    {change_pct:+.2f}%\n")
                f.write(f"  Volatilité (High-Low): {volatility_day:.2f}%\n")
                f.write(f"  Max Drawdown (Jour):   {dd_intraday:.2f}%\n")
                f.write("\n")
                
            except Exception as e:
                f.write(f"[{ticker}] : Erreur lors de l'analyse ({str(e)})\n")

    print(f"Rapport sauvegardé : {filename}")

if __name__ == "__main__":
    generate_daily_report()