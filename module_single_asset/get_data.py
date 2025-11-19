import requests
import pandas as pd
import time

API_KEY = "d4evlp1r01qrcbrulkd0d4evlp1r01qrcbrulkdg"  # key Finnhub
symbol = "AAPL"
resolution = "D"  # Données journalières

# période : 1 an
to_date = int(time.time())
from_date = to_date - 365 * 24 * 60 * 60  #1 an en secondes

url = f"https://finnhub.io/api/v1/stock/candle"
params = {
    "symbol": symbol,
    "resolution": resolution,
    "from": from_date,
    "to": to_date,
    "token": API_KEY
}

response = requests.get(url, params=params)
data = response.json()

if data["s"] == "ok":
    # Transformer en DataFrame
    df = pd.DataFrame({
        "Date": pd.to_datetime(data["t"], unit="s"),
        "Open": data["o"],
        "High": data["h"],
        "Low": data["l"],
        "Close": data["c"],
        "Volume": data["v"]
    })
    df.to_csv("data_aapl.csv", index=False)
    print("Sauvegarde réussie !")
else:
    print("Erreur : pas de données")

