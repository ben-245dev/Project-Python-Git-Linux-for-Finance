import streamlit as st
from core.data import fetch_data, detect_col
import numpy as np

def run():
    st.title("Gestion du risque & reporting")
    ticker = st.sidebar.text_input("Ticker", "AAPL")
    start = st.sidebar.date_input("Début", value="2022-01-01")
    end = st.sidebar.date_input("Fin", value="2023-12-31")
    horizon = st.sidebar.number_input("Horizon VaR (jours)", 1, 30, 5)
    level = st.sidebar.slider("Quantile (%)", 90, 99, 95)

    if st.sidebar.button("Analyser"):
        df = fetch_data(ticker, start, end)
        close_col = detect_col(df, ticker, "Close")
        if not close_col: st.error("Pas de prix."); st.stop()
        returns = df[close_col].pct_change().dropna()
        var = np.percentile(returns * np.sqrt(horizon), 100-level)
        st.metric(f"VaR {level}% sur {horizon}j", f"{var*100:.2f} %")
        st.line_chart(returns)
    else:
        st.info("Paramètres puis cliquer sur Analyser.")
