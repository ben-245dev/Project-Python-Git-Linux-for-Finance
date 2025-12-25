import streamlit as st
from core.data import fetch_data, detect_col
import plotly.graph_objects as go
import pandas as pd
def run():
    st.title("Analyse & graphique interactive")
    ticker = st.sidebar.text_input("Ticker", "AAPL")
    start = st.sidebar.date_input("Début", value="2022-01-01")
    end = st.sidebar.date_input("Fin", value="2023-12-31")
    sma1 = st.sidebar.number_input("SMA rapide", 2, 100, 20)
    sma2 = st.sidebar.number_input("SMA lente", 2, 200, 50)
    show_ema = st.sidebar.checkbox(f"Afficher EMA {sma1}")

    if st.sidebar.button("Charger"):
        df = fetch_data(ticker, start, end)
        if df.empty:
            st.error(f"Aucune donnée pour le ticker {ticker} ou sur la période sélectionnée.")
            st.stop()
        # Recherche des colonnes
        close_col = detect_col(df, ticker, "Close")
        open_col = detect_col(df, ticker, "Open")
        high_col = detect_col(df, ticker, "High")
        low_col = detect_col(df, ticker, "Low")
        # Vérification de l'existence des données minimales
        if not all([close_col, open_col, high_col, low_col]):
            st.error("Données OHLC incomplètes pour ce ticker, impossible d'afficher le chandelier.")
            st.stop()
        # Au moins une donnée de prix
        if df[close_col].dropna().empty:
            st.error("La colonne de prix de clôture est vide.")
            st.stop()
        # Calcul indicateurs
        df['SMA1'] = df[close_col].rolling(sma1).mean()
        df['SMA2'] = df[close_col].rolling(sma2).mean()
        if show_ema:
            df['EMA1'] = df[close_col].ewm(span=sma1).mean()
        # Graphique chandelier OHLC + overlays
        st.subheader("Graphique chandelier & indicateurs")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=pd.to_datetime(df['Date']),
            open=df[open_col], high=df[high_col], low=df[low_col], close=df[close_col],
            name='Chandelier'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA1'], name=f"SMA {sma1}", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA2'], name=f"SMA {sma2}", line=dict(color='orange')))
        if show_ema and 'EMA1' in df:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA1'], name=f"EMA {sma1}", line=dict(color='purple', dash='dot')))
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Sélectionne un ticker et clique sur Charger.")
