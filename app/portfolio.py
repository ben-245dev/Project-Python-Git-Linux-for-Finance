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
from export import render_export_button
def page_portfolio():
    st.title("📊 Module B : Analyse de Portefeuille Multi-Actifs")
    st.markdown("---")

    st.subheader("Configuration des Actifs et des Poids")

    # 1) Nombre d'actifs
    N_assets = st.selectbox(
        "Nombre d'actifs dans le portefeuille (min. 3) :",
        options=list(range(3, 21)),
        index=0,
        key="port_n_assets",
    )
    
    # 2) Saisie libre des tickers
    default_tickers = "AAPL, MSFT, GOOGL, NVDA"
    tickers_input = st.text_input(
        "Liste des tickers (séparés par des virgules)",
        value=default_tickers,
        key="port_tickers_input",
    )
    
    # Parse en liste propre
    all_tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
    
    if len(all_tickers) < 3:
        st.warning("Veuillez saisir au moins 3 tickers pour le portefeuille.")
        return
    
    # On ne garde que les N_assets premiers tickers saisis
    asset_names = all_tickers[:N_assets]
    
    # 3) Poids configurables pour chaque actif
    st.markdown("### Poids du portefeuille (Long/Short possibles)")
    
    weights_input = []
    cols = st.columns(len(asset_names))
    
    for i, name in enumerate(asset_names):
        with cols[i]:
            w = st.number_input(
                f"Poids {name}",
                min_value=-1.0,
                max_value=2.0,
                value=1.0 / len(asset_names),  # équipondération par défaut
                step=0.01,
                format="%.2f",
                key=f"port_weight_{i}",
            )
            weights_input.append(w)


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
    st.subheader(f"Exporter l’historique de {ticker}")
    render_export_button(df, filename=f"historique_{ticker}.csv")

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

            # -----------------------------
            # Stratégies appliquées au portefeuille
            # -----------------------------
            port_params = {
                "mom_lookback": port_mom_lb,
                "fast_ma": port_fast_ma,
                "slow_ma": port_slow_ma,
                "breakout_lookback": port_breakout_lb,
            }

            port_strategies = build_strategies(price_portfolio, port_params)
            port_strat = port_strategies[port_strategy_name]
            port_strat_returns = port_strat["returns"]
            port_strat_equity = port_strat["equity"]
            port_bh_equity = port_strategies["Buy & Hold"]["equity"]

            st.markdown("---")
            st.subheader(f"Stratégies appliquées au portefeuille : {port_strategy_name}")

            fig_port_strat = go.Figure()

            fig_port_strat.add_trace(
                go.Scatter(
                    x=price_portfolio.index,
                    y=price_portfolio.values,
                    mode="lines",
                    name="Prix Portefeuille (Base 100)",
                    yaxis="y1",
                    line=dict(color="#00c3ff"),
                )
            )

            fig_port_strat.add_trace(
                go.Scatter(
                    x=port_strat_equity.index,
                    y=port_strat_equity.values,
                    mode="lines",
                    name="Equity stratégie Portefeuille",
                    yaxis="y2",
                    line=dict(color="#ff9900"),
                )
            )

            fig_port_strat.update_layout(
                template="plotly_dark",
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(domain=[0.0, 1.0]),
                yaxis=dict(title="Prix portefeuille (Base 100)", side="left"),
                yaxis2=dict(
                    title="Equity stratégie (normalisée)",
                    overlaying="y",
                    side="right",
                ),
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_port_strat, use_container_width=True)

            st.subheader("Métriques de la stratégie sur le portefeuille")

            port_metrics = compute_strategy_metrics(port_strat_returns, port_strat_equity)
            port_bh_metrics = compute_strategy_metrics(
                port_strategies["Buy & Hold"]["returns"], port_bh_equity
            )

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total return (strat port.)", f"{port_metrics['total_return']*100:.2f} %")
            with c2:
                st.metric("Max drawdown (strat port.)", f"{port_metrics['max_drawdown']*100:.2f} %")
            with c3:
                st.metric("Sharpe (strat port.)", f"{port_metrics['sharpe']:.2f}")
            with c4:
                st.metric("Vol annualisée (strat port.)", f"{port_metrics['vol']*100:.2f} %")

            c5, c6, c7 = st.columns(3)
            with c5:
                st.metric("Total return (BH port.)", f"{port_bh_metrics['total_return']*100:.2f} %")
            with c6:
                st.metric("Max drawdown (BH port.)", f"{port_bh_metrics['max_drawdown']*100:.2f} %")
            with c7:
                st.metric("Sharpe (BH port.)", f"{port_bh_metrics['sharpe']:.2f}")

            st.subheader("Distribution des rendements de la stratégie (Portefeuille)")
            fig_port_hist = go.Figure(
                data=[
                    go.Histogram(
                        x=port_strat_returns.values,
                        nbinsx=50,
                        marker_color="#ff9900",
                    )
                ]
            )
            fig_port_hist.update_layout(
                template="plotly_dark",
                xaxis_title="Rendement",
                yaxis_title="Fréquence",
                margin=dict(l=10, r=10, t=30, b=10),
            )

            st.plotly_chart(fig_port_hist, use_container_width=True)



