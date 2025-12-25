import streamlit as st
from core.data import fetch_data, detect_col
import plotly.graph_objects as go

def run():
    st.title("Portefeuille - Visualisation et allocation")
    tickers = st.sidebar.text_input("Tickers (séparés par ,)", "AAPL,MSFT,TSLA,GSPC")
    poids_text = st.sidebar.text_input("Poids (séparés par ,)", "0.25,0.25,0.25,0.25")
    start = st.sidebar.date_input("Début", value="2022-01-01")
    end = st.sidebar.date_input("Fin", value="2023-12-31")

    if st.sidebar.button("Calculer"):
        tickers_list = [t.strip() for t in tickers.split(',')]
        poids = [float(x) for x in poids_text.split(',')]
        curves = []
        for i, ticker in enumerate(tickers_list):
            df = fetch_data(ticker, start, end)
            close_col = detect_col(df, ticker, "Close")
            if not close_col: continue
            curve = (df[close_col] / df[close_col].iloc[0]) * poids[i]
            curves.append(curve)
            st.line_chart(curve.rename(f"{ticker} (part pondérée)"))
        if len(curves) > 0:
            port_curve = sum(curves)
            st.subheader("Performance totale du portefeuille")
            st.line_chart(port_curve.rename("Portefeuille"))
    else:
        st.info("Saisis tickers et poids puis Clique sur Calculer.")
