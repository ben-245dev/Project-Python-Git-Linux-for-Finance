import streamlit as st
import pandas as pd
import yfinance as yf

st.title("Dashboard interactif mono-actif")

# Saisie du ticker et de la période
ticker = st.text_input("Entrez le ticker (ex : AAPL, MSFT, TSLA)", "AAPL")
start_date = st.date_input("Date de début", pd.to_datetime("2023-01-01"))
end_date = st.date_input("Date de fin", pd.to_datetime("2023-12-31"))

if st.button("Charger le dashboard"):
    # Télécharger les données
    with st.spinner(f"Téléchargement pour {ticker}..."):
        data = yf.download(ticker, start=start_date, end=end_date, group_by='ticker')
        if data.empty:
            st.error("Aucune donnée trouvée pour ce ticker ou cette période.")
        else:
            # Gestion automatique de l'index (mono ou multi-ticker)
            if isinstance(data.columns, pd.MultiIndex):
                if ticker in data.columns.get_level_values(0):
                    df = data[ticker]
                    df = df.reset_index()
                else:
                    st.error("Le ticker spécifié est absent des données.")
                    st.stop()
            else:
                df = data.reset_index()

            # Nettoyage des colonnes
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            st.success("Données récupérées !")
            st.subheader("Aperçu des données")
            st.write(df.head())

            st.subheader("Prix de clôture")
            st.line_chart(df['Close'])

            df['Returns'] = df['Close'].pct_change()
            st.subheader("Rendement moyen journalier")
            st.write(round(df['Returns'].mean(), 4))

            st.subheader("Volatilité annualisée")
            vol_annual = df['Returns'].std() * (252 ** 0.5)
            st.write(round(vol_annual, 4))

            if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']) and 'Date' in df.columns:
                st.subheader("Graphique Chandelier (OHLC)")
                import plotly.graph_objects as go
                fig = go.Figure(data=[go.Candlestick(
                    x=pd.to_datetime(df['Date']),
                    open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close']
                )])
                st.plotly_chart(fig)

            if 'Volume' in df.columns:
                st.subheader("Volume échangé")
                st.line_chart(df['Volume'])

            st.subheader("Télécharger les données")
            st.download_button("Télécharger le CSV", df.to_csv(index=False), f"{ticker}.csv")
