import streamlit as st
from core.data import fetch_data, detect_col
import pandas as pd

def run():
    st.title("Screeners multi-actifs")
    tickers = st.sidebar.text_input("Tickers (séparés par ,)", "AAPL,MSFT,TSLA,GSPC")
    start = st.sidebar.date_input("Début", value="2022-01-01")
    end = st.sidebar.date_input("Fin", value="2023-12-31")
    seuil_perf = st.sidebar.slider("Performance mini (%)", -100, 100, 5)
    seuil_vol = st.sidebar.slider("Volatilité maxi (%)", 0, 100, 60)

    if st.sidebar.button("Lancer le screener"):
        res = []
        for ticker in [t.strip() for t in tickers.split(',')]:
            df = fetch_data(ticker, start, end)
            close_col = detect_col(df, ticker, "Close")
            if not close_col: continue
            perf = (df[close_col].iloc[-1] / df[close_col].iloc[0] - 1) * 100
            vol = df[close_col].pct_change().std() * (252**0.5) * 100
            if perf >= seuil_perf and vol <= seuil_vol:
                res.append({"Ticker": ticker, "Performance (%)": perf, "Volatilité (%)": vol})
        st.write("Résultats filtrés :")
        st.dataframe(pd.DataFrame(res))
    else:
        st.info("Clique sur Lancer le screener.")
