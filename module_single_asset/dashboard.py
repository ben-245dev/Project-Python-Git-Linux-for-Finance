import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv("data_aapl.csv")

st.title("Analyse de l'actif Apple (AAPL)")

# Afficher les données brutes
st.write("Données brutes :", df)

# Supposons que 'c' correspond au prix de clôture Finnhub
if 'c' in df.columns:
    st.line_chart(df['c'], height=300)

    # Calcul du rendement simple
    df['returns'] = df['c'].pct_change()
    st.write("Rendement moyen :", df['returns'].mean())

    # Calcul de la volatilité annualisée (approx)
    st.write("Volatilité (annuelle) :", df['returns'].std() * (252**0.5))
else:
    st.warning("La colonne 'c' (prix de clôture) n'a pas été trouvée.")

