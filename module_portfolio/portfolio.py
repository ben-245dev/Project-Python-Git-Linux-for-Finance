import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import pytz
import streamlit as st
import yfinance as yf
from scipy.stats import norm  # prêt pour extensions VaR
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from strategy import build_strategies
from metrics import compute_strategy_metrics

def page_portfolio():
    st.title("📊 Module B : Analyse de Portefeuille Multi-Actifs")
    st.markdown("---")

    st.subheader("Configuration des Actifs et des Poids")

    N_assets = st.selectbox(
        "Choisissez le nombre d'actifs dans le portefeuille (min. 3) :",
        options=list(range(3, 11)),
        index=0,
        key="port_n_assets",
    )

    TICKER_OPTIONS = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "NVDA",
        "TSLA",
        "AMZN",
        "BTC-USD",
        "ETH-USD",
        "EURUSD=X",
        "GC=F",
    ]

    asset_names = []
    weights_input = []

    col_names = st.columns(N_assets)
    col_weights = st.columns(N_assets)

    for i in range(N_assets):
        name = col_names[i].selectbox(
            f"Actif #{i+1}",
            options=TICKER_OPTIONS,
            index=i % len(TICKER_OPTIONS),
            key=f"port_asset_{i}",
        )
        asset_names.append(name)

        weight = col_weights[i].number_input(
            f"Poids {name}",
            min_value=-1.0,
            max_value=2.0,
            value=1.0 / N_assets,
            step=0.01,
            format="%.2f",
            key=f"port_weight_{i}",
        )
        weights_input.append(weight)

    st.markdown("---")
    col_date_start, col_date_end, col_button = st.columns(3)
    start_date = col_date_start.date_input(
        "Date de début", pd.to_datetime("2023-01-01"), key="port_start"
    )
    end_date = col_date_end.date_input(
        "Date de fin", pd.to_datetime("today"), key="port_end"
    )

    # Paramètres financiers dans la sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Paramètres Financiers (Portefeuille)")
    initial_investment = st.sidebar.number_input(
        "Investissement initial (€)",
        min_value=100.0,
        value=10000.0,
        step=500.0,
        key="port_init_inv",
    )

    risk_free_rate = (
        st.sidebar.number_input(
            "Taux sans risque annuel (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            key="port_rf",
        )
        / 100
    )
    # Paramètres de stratégies sur la page (corps)
    st.markdown("---")
    st.subheader("Paramètres de stratégies appliquées au portefeuille")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        port_strategy_name = st.selectbox(
            "Stratégie sur le portefeuille",
            ["Buy & Hold", "Momentum", "Mean Reversion", "MA Crossover", "Breakout"],
            key="port_strategy_name",
        )

    with col_s2:
        port_mom_lb = st.slider(
            "Momentum lookback (jours)", 5, 120, 20, step=5, key="port_mom_lb"
        )

    col_s3, col_s4 = st.columns(2)
    with col_s3:
        port_fast_ma = st.slider(
            "MA rapide (jours)", 5, 60, 20, step=5, key="port_fast_ma"
        )
    with col_s4:
        port_slow_ma = st.slider(
            "MA lente (jours)", 20, 200, 50, step=10, key="port_slow_ma"
        )

    port_breakout_lb = st.slider(
        "Breakout lookback (jours)", 20, 200, 50, step=10, key="port_breakout_lb"
    )

    if col_button.button("Lancer l'Analyse du Portefeuille", key="port_run"):
        tickers = asset_names
        weights = np.array(weights_input)

        if N_assets < 3:
            st.error("🚨 Le module Portfolio nécessite au moins 3 actifs différents.")
            st.stop()

        with st.spinner("Récupération et calcul des données..."):
            if np.sum(weights) == 0:
                st.error("Les poids ne peuvent pas tous être nuls.")
                st.stop()

            if abs(np.sum(weights) - 1.0) > 0.001:
                st.warning(
                    f"La somme des poids est de {np.sum(weights):.2f}. Normalisation automatique appliquée."
                )
                weights = weights / np.sum(weights)

            try:
                data = yf.download(tickers, start=start_date, end=end_date)["Close"]
                data = data.dropna()
                returns = data.pct_change().dropna()

                if data.empty:
                    st.error("Aucune donnée trouvée ou trop de valeurs manquantes.")
                    st.stop()

            except Exception as e:
                st.error(
                    f"Erreur lors du téléchargement des données (Vérifiez l'API ou les Tickers) : {e}"
                )
                st.stop()

            # Rendements et prix du portefeuille
            portfolio_return_daily = returns.dot(weights)
            cumulative_portfolio = (1 + portfolio_return_daily).cumprod()
            price_portfolio = cumulative_portfolio * 100  # base 100

            final_portfolio_ratio = cumulative_portfolio.iloc[-1]
            total_rate_of_return = (final_portfolio_ratio - 1) * 100
            final_portfolio_value = final_portfolio_ratio * initial_investment

            annual_return = portfolio_return_daily.mean() * 252 * 100
            annual_volatility = portfolio_return_daily.std() * np.sqrt(252) * 100

            annual_excess_return = portfolio_return_daily.mean() * 252 - risk_free_rate
            sharpe_ratio = (
                annual_excess_return / portfolio_return_daily.std() * np.sqrt(252)
            )

            roll_max = cumulative_portfolio.cummax()
            daily_drawdown = cumulative_portfolio / roll_max - 1.0
            max_drawdown = daily_drawdown.min() * 100

            var_95 = np.percentile(portfolio_return_daily, 5) * -100

            st.subheader("📈 Métriques de Performance et de Risque")

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Rendement Annualisé", f"{annual_return:.2f} %")
            col2.metric("Volatilité (Annualisée)", f"{annual_volatility:.2f} %")
            col3.metric("Ratio de Sharpe", f"{sharpe_ratio:.2f}")
            col4.metric("Max Drawdown", f"{max_drawdown:.2f} %")
            col5.metric("VaR (95%) Quotidienne", f"{var_95:.2f} %")

            # Performance cumulée
            st.subheader("Performance Cumulée (Actifs vs Portefeuille)")

            cumulative_returns_assets = (1 + returns).cumprod() * 100
            cumulative_returns_assets["Portfolio"] = cumulative_portfolio * 100

            fig_perf = px.line(
                cumulative_returns_assets,
                x=cumulative_returns_assets.index,
                y=cumulative_returns_assets.columns,
                title="Performance Cumulée (Base 100)",
                labels={"value": "Valeur (Base 100)", "variable": "Actif"},
                template="plotly_dark",
            )

            fig_perf.data[-1].line.width = 4
            fig_perf.data[-1].line.color = "yellow"

            st.plotly_chart(fig_perf, use_container_width=True)

            # Corrélation
            st.subheader("Matrice de Corrélation (Effet de Diversification)")
            corr_matrix = returns.corr()

            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Corrélation entre les actifs",
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            # Exposition
            st.subheader("Exposition par Actif (Long/Short)")

            df_weights = pd.DataFrame(
                {
                    "Asset": data.columns,
                    "Weight": weights,
                }
            )

            fig_bar = px.bar(
                df_weights,
                x="Asset",
                y="Weight",
                title="Poids du Portefeuille (Positions Longues et Courtes)",
                color="Weight",
                color_continuous_scale=["red", "blue"],
            )

            fig_bar.add_hline(y=0, line_dash="solid", line_color="white")
            fig_bar.update_layout(yaxis_tickformat=".0%")

            st.plotly_chart(fig_bar, use_container_width=True)

            gross_exposure = np.sum(np.abs(weights)) * 100
            st.metric("Exposition Brute (Risque total)", f"{gross_exposure:.0f} %")
            if gross_exposure > 100:
                st.warning(
                    "⚠️ Exposition brute supérieure à 100% : "
                    "Le portefeuille utilise de l'effet de levier ou des positions courtes."
                )


