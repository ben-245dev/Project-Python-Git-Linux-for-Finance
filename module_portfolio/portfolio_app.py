import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm 
import os, sys

# --- Configuration de la Page ---
st.set_page_config(page_title="Quant B - Portefeuille Multi-Actifs", layout="wide")

st.title("📊 Module B : Analyse de Portefeuille Multi-Actifs")
st.markdown("---") 

# --- 1. Zone de configuration (Affichée en haut de la page) ---
st.subheader("Configuration des Actifs et des Poids")

# 1.1. Sélection du Nombre d'Actifs (Menu déroulant)
N_assets = st.selectbox(
    "Choisissez le nombre d'actifs dans le portefeuille (min. 3) :",
    options=list(range(3, 11)), # Propose de 3 à 10 actifs
    index=0 # Sélectionne 3 par défaut
)

# Liste de tous les tickers potentiels pour les menus déroulants
TICKER_OPTIONS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMZN", "BTC-USD", "ETH-USD", "EURUSD=X", "GC=F"]

# Conteneurs pour les inputs d'actifs et de poids
asset_names = []
weights_input = []

# Créer des colonnes pour aligner les menus déroulants et les inputs de poids
col_names = st.columns(N_assets)
col_weights = st.columns(N_assets)

# 1.2. Création dynamique des inputs (Choix des actifs et des poids)
for i in range(N_assets):
    # Menu déroulant pour le choix de l'actif
    name = col_names[i].selectbox(
        f"Actif #{i+1}",
        options=TICKER_OPTIONS,
        index=i % len(TICKER_OPTIONS) # Choisit une valeur par défaut différente pour chaque slot
    )
    asset_names.append(name)
    
    # Input numérique pour le poids (Allocation personnalisable)
    weight = col_weights[i].number_input(
        f"Poids {name}",
        min_value=0.0,
        max_value=1.0,
        value=1.0 / N_assets, # Équipondération par défaut
        step=0.01,
        format="%.2f",
        key=f"weight_{i}"
    )
    weights_input.append(weight)

# 1.3. Récupération des dates et du bouton de lancement
st.markdown("---")
col_date_start, col_date_end, col_button = st.columns(3)
start_date = col_date_start.date_input("Date de début", pd.to_datetime("2023-01-01"))
end_date = col_date_end.date_input("Date de fin", pd.to_datetime("today"))

# Taux sans risque (pour le Sharpe Ratio)
risk_free_rate = st.sidebar.number_input("Taux sans risque annuel (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1) / 100

# Le bouton de lancement est sur la page principale
if col_button.button("Lancer l'Analyse du Portefeuille"):
    
    # --- 2. Validation des Inputs ---
    tickers = asset_names
    weights = np.array(weights_input)
    
    if N_assets < 3:
        st.error("🚨 Le module Portfolio nécessite au moins 3 actifs différents (Exigence projet).")
        st.stop()
    
    with st.spinner("Récupération et calcul des données..."):
        
        # --- 3. Normalisation des Poids (Sécurité) ---
        if np.sum(weights) == 0:
            st.error("Les poids ne peuvent pas tous être nuls.")
            st.stop()
        
        if abs(np.sum(weights) - 1.0) > 0.001:
            st.warning(f"La somme des poids est de {np.sum(weights):.2f}. Normalisation automatique appliquée.")
            weights = weights / np.sum(weights) 

        # --- 4. Récupération des données et Rendements (Robustesse) ---
        try:
            # Récupère les données et nettoie
            data = yf.download(tickers, start=start_date, end=end_date)['Close']
            data = data.dropna() 
            returns = data.pct_change().dropna()
            
            if data.empty:
                st.error("Aucune donnée trouvée ou trop de valeurs manquantes.")
                st.stop()
            
        except Exception as e:
            st.error(f"Erreur lors du téléchargement des données (Vérifiez l'API ou les Tickers) : {e}")
            st.stop()
        
        # --- 5. Calculs des Métriques ---
        
        # Rendement et Valeur Cumulative du portefeuille
        portfolio_return_daily = returns.dot(weights)
        cumulative_portfolio = (1 + portfolio_return_daily).cumprod()
        
        # Métriques Annualisées
        annual_return = portfolio_return_daily.mean() * 252 * 100
        annual_volatility = portfolio_return_daily.std() * np.sqrt(252) * 100
        
        # Sharpe Ratio
        annual_excess_return = portfolio_return_daily.mean() * 252 - risk_free_rate
        sharpe_ratio = annual_excess_return / portfolio_return_daily.std() * np.sqrt(252)
        
        # Max Drawdown
        roll_max = cumulative_portfolio.cummax()
        daily_drawdown = cumulative_portfolio / roll_max - 1.0
        max_drawdown = daily_drawdown.min() * 100
        
        # Value at Risk (VaR) à 95% (Risk Indicator)
        var_95 = np.percentile(portfolio_return_daily, 5) * -100 # Exprimé en perte positive
        
        # --- 6. Affichage des KPIs ---
        st.subheader("📈 Métriques de Performance et de Risque")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Rendement Annualisé", f"{annual_return:.2f} %")
        col2.metric("Volatilité (Annualisée)", f"{annual_volatility:.2f} %")
        col3.metric("Ratio de Sharpe", f"{sharpe_ratio:.2f}")
        col4.metric("Max Drawdown", f"{max_drawdown:.2f} %")
        col5.metric("VaR (95%) Quotidienne", f"{var_95:.2f} %") 

        # --- 7. Visualisations ---

        # 7.1 Graphique de Performance Comparée (Base 100)
        st.subheader("Performance Cumulée (Actifs vs Portefeuille)")
        
        cumulative_returns_assets = (1 + returns).cumprod() * 100
        cumulative_returns_assets['Portfolio'] = cumulative_portfolio * 100 
        
        fig_perf = px.line(
            cumulative_returns_assets, 
            x=cumulative_returns_assets.index, 
            y=cumulative_returns_assets.columns,
            title="Performance Cumulée (Base 100)",
            labels={"value": "Valeur (Base 100)", "variable": "Actif"},
            template="plotly_dark"
        )
        
        # Mettre en évidence le portefeuille
        fig_perf.data[-1].line.width = 4
        fig_perf.data[-1].line.color = 'yellow'

        st.plotly_chart(fig_perf, use_container_width=True)

        # 7.2 Matrice de Corrélation (Heatmap)
        st.subheader("Matrice de Corrélation (Effet de Diversification)")
        
        corr_matrix = returns.corr()
        
        fig_corr = px.imshow(
            corr_matrix, 
            text_auto=".2f",
            aspect="auto", 
            color_continuous_scale='RdBu_r', 
            title="Corrélation entre les actifs"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # 7.3 Composition du portefeuille (Pie Chart)
        st.subheader("Répartition de l'Allocation")
        
        df_weights = pd.DataFrame({
            'Asset': data.columns,
            'Weight': weights
        })
        fig_pie = px.pie(df_weights, values='Weight', names='Asset', title="Allocation des actifs")
        st.plotly_chart(fig_pie)