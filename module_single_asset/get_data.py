import yfinance as yf
import pandas as pd

# Paramètres
symbol = "AAPL"
start_date = "2023-01-01"
end_date = "2023-12-31"

# Récupération des données
data = yf.download(symbol, start=start_date, end=end_date)
data = data.droplevel('Ticker', axis=1)
data.reset_index().to_csv("data_aapl.csv", index=False)
print("Sauvegarde réussie !")