import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm 

# Configuration de la page
st.set_page_config(page_title="Quant B - Portefeuille Multi-Actifs", layout="wide")

st.title("📊 Module B : Analyse de Portefeuille Multi-Actifs")
st.markdown("---") # Sépare le titre du contenu principal pour forcer l'affichage

# --- Début de la Barre Latérale pour les Inputs (User Controls) ---
st.sidebar.header("Configuration du Portefeuille")

# Définition des Tickers (Multi-Assets)
default_tickers = "AAPL, MSFT, GOOGL, NVDA"
tickers_input = st.sidebar.text_input("Tickers (séparés par virgule, min. 3)", default_tickers)
tickers = [x.strip().upper() for x in tickers_input.split(',')]

# Sélecteurs de dates
start_date = st.sidebar.date_input("Date de début", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("Date de fin", pd.to_datetime("today"))

# --- Bouton de Lancement ---
if st.sidebar.button("Lancer l'Analyse du Portefeuille"):
    
    # 1. Vérification des inputs (Robustesse)
    if len(tickers) < 3:
        st.error("🚨 Le module Portfolio nécessite au moins 3 actifs différents (Exigence projet).")
        st.stop()
    
    with st.spinner("Récupération et calcul des données..."):
        
        # --- 2. Récupération des données et Rendements ---
        try:
            # Télécharge les prix de clôture pour tous les tickers
            data = yf.download(tickers, start=start_date, end=end_date)['Close']
            data = data.dropna() 
            returns = data.pct_change().dropna()
            
            if data.empty:
                st.error("Aucune donnée trouvée ou trop de valeurs manquantes.")
                st.stop()
            
        except Exception as e:
            st.error(f"Erreur lors du téléchargement des données (Vérifiez l'API ou les Tickers) : {e}")
            st.stop()
            
        # --- 3. Gestion des Poids (Allocation Personnalisable) ---
        st.subheader("Configuration des Poids du Portefeuille")
        st.markdown("Utilisez les sliders pour définir l'allocation (La somme doit être proche de 1.0).")
        
        weights_input = []
        initial_weight = 1.0 / len(data.columns)
        
        cols = st.columns(len(data.columns))
        
        for i, ticker in enumerate(data.columns):
            w = cols[i].slider(
                f"Poids {ticker}", 
                min_value=0.0, 
                max_value=1.0, 
                value=initial_weight, 
                step=0.01, 
                key=f"w_{ticker}"
            )
            weights_input.append(w)
            
        weights = np.array(weights_input)
        
        # Normalisation (Sécurité)
        if np.sum(weights) == 0:
            st.error("Les poids ne peuvent pas tous être nuls.")
            st.stop()
        
        if abs(np.sum(weights) - 1.0) > 0.001:
            st.warning(f"La somme des poids est de {np.sum(weights):.2f}. Normalisation automatique appliquée.")
            weights = weights / np.sum(weights) 

        # --- 4. Calculs des Métriques ---
        
        # Rendement Portefeuille
        portfolio_return_daily = returns.dot(weights)
        cumulative_portfolio = (1 + portfolio_return_daily).cumprod()
        
        # Métriques Annualisées
        annual_return = portfolio_return_daily.mean() * 252 * 100
        annual_volatility = portfolio_return_daily.std() * np.sqrt(252) * 100
        
        # Sharpe Ratio (hypothèse taux sans risque = 2%)
        risk_free_rate = 0.02 
        annual_excess_return = portfolio_return_daily.mean() * 252 - (risk_free_rate / 252)
        sharpe_ratio = annual_excess_return / portfolio_return_daily.std() * np.sqrt(252)
        
        # Max Drawdown
        roll_max = cumulative_portfolio.cummax()
        daily_drawdown = cumulative_portfolio / roll_max - 1.0
        max_drawdown = daily_drawdown.min() * 100
        
        # --- Affichage des KPIs ---
        st.subheader("📈 Métriques du Portefeuille")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rendement Annualisé", f"{annual_return:.2f} %")
        col2.metric("Volatilité (Annualisée)", f"{annual_volatility:.2f} %")
        col3.metric("Ratio de Sharpe", f"{sharpe_ratio:.2f}")
        col4.metric("Max Drawdown", f"{max_drawdown:.2f} %")

        # --- 5. Visualisations ---

        # 5.1 Graphique de Performance Comparée (Base 100)
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

        # 5.2 Matrice de Corrélation (Heatmap)
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

        # 5.3 Composition du portefeuille (Pie Chart)
        st.subheader("Répartition de l'Allocation")
        
        df_weights = pd.DataFrame({
            'Asset': data.columns,
            'Weight': weights
        })
        fig_pie = px.pie(df_weights, values='Weight', names='Asset', title="Allocation des actifs")
        st.plotly_chart(fig_pie)