import requests
import pandas as pd

API_KEY = "d4evlp1r01qrcbrulkd0d4evlp1r01qrcbrulkdg"  # key Finnhub

symbol = "AAPL"
url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={API_KEY}"

response = requests.get(url)
data = response.json()

df = pd.DataFrame([data])
print(df)
df.to_csv("data_aapl.csv", index=False)
