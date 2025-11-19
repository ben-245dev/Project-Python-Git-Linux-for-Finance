import streamlit as st
import pandas as pd

# Chargement des données
try:
    df = pd.read_csv("data_aapl.csv")
except FileNotFoundError:
    st.error("Le fichier data_aapl.csv n'a pas été trouvé. Lance d'abord get_data.py.")
    st.stop()

if 'Close' not in df.columns:
    st.error("La colonne 'Close' est absente du fichier CSV. Vérifie le contenu.")
    st.stop()

# Gestion des colonnes manquantes pour le graphique chandelier
has_ohlc = all(col in df.columns for col in ['Open', 'High', 'Low', 'Close'])

# Interface utilisateur
st.title("Dashboard financier : Apple (AAPL)")

st.subheader("Aperçu des données")
st.write(df.head())

# Courbe de clôture
st.subheader("Prix de clôture")
st.line_chart(df['Close'])

# Calcul et affichage des rendements journaliers
df['Returns'] = df['Close'].pct_change()
st.subheader("Rendement moyen journalier")
st.write(round(df['Returns'].mean(), 4))

# Volatilité annualisée
st.subheader("Volatilité annualisée")
vol_annual = df['Returns'].std() * (252 ** 0.5)
st.write(round(vol_annual, 4))

# Graphique chandelier (OHLC) si les colonnes existent
if has_ohlc and 'Date' in df.columns:
    st.subheader("Graphique Chandelier (OHLC)")
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Candlestick(
        x=pd.to_datetime(df['Date']),
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])
    st.plotly_chart(fig)
else:
    st.warning("Colonnes OHLC ou Date manquantes pour afficher le graphique chandelier.")

# Affichage du volume (optionnel)
if 'Volume' in df.columns:
    st.subheader("Volume échangé")
    st.line_chart(df['Volume'])
else:
    st.info("La colonne Volume est absente du fichier CSV.")

# Options d’export/download (bonus)
st.subheader("Télécharger les données")
st.download_button("Télécharger le CSV", df.to_csv(index=False), "data_aapl.csv")

