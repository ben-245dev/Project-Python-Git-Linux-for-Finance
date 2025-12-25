import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from backend.simulation import run_monte_carlo, run_historical_crash_test, compute_risk_metrics
from backend.data_loader import load_batch_data
from backend.optimization import optimize_portfolio_weights
from backend.simulation import run_monte_carlo
from backend.metrics import compute_strategy_metrics

def page_portfolio():
    st.title("🧠 Optimisation & Analyse Avancée")
    
    # --- 1. Configuration ---
    with st.expander("🛠️ Configuration du Portefeuille", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            tickers_input = st.text_input("Actifs (séparés par virgules)", "AAPL, MSFT, GOOGL, NVDA, GLD, BTC-USD")
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        with col2:
            start_date = st.date_input("Date début historique", pd.to_datetime("2022-01-01"))
            amount = st.number_input("Capital (€)", value=10000)

    if not tickers or len(tickers) < 2:
        st.warning("Il faut au moins 2 actifs pour faire un portefeuille.")
        st.stop()

    # --- Chargement ---
    prices = load_batch_data(tickers, start_date, pd.to_datetime("today"))
    if prices.empty:
        st.error("Erreur de récupération des données.")
        return
    
    returns = prices.pct_change().dropna()

    # --- 2. Optimisation avec Contraintes ---
    st.markdown("### 🤖 Allocation Intelligente (IA)")
    
    col_opt1, col_opt2 = st.columns([1, 2])
    
    with col_opt1:
        st.subheader("Paramètres")
        obj = st.radio("Objectif", ["Maximiser Sharpe (Rentabilité/Risque)", "Minimiser Volatilité (Sécurité)"])
        target = "sharpe" if "Sharpe" in obj else "min_vol"
        
        st.markdown("**Contraintes de poids :**")
        min_w = st.slider("Poids Minimum par actif", 0.0, 0.5, 0.01, 0.01)
        max_w = st.slider("Poids Maximum par actif", 0.1, 1.0, 1.0, 0.05)

        if st.button("⚡ Optimiser"):
            # Vérification basique des contraintes
            if min_w * len(tickers) > 1.0:
                st.error("Impossible : Le poids minimum cumulé dépasse 100%. Réduisez le Min.")
            else:
                opt_weights, perf = optimize_portfolio_weights(prices, target, min_weight=min_w, max_weight=max_w)
                if opt_weights:
                    st.session_state['opt_weights'] = opt_weights
                    st.session_state['opt_perf'] = perf
                else:
                    st.error("L'optimiseur n'a pas trouvé de solution avec ces contraintes.")

    with col_opt2:
        if 'opt_weights' in st.session_state:
            w = st.session_state['opt_weights']
            p = st.session_state['opt_perf']
            
            # Pie Chart
            df_w = pd.DataFrame({"Actif": w.keys(), "Poids": w.values()})
            df_w = df_w[df_w["Poids"] > 0.001]
            
            fig = px.pie(df_w, values="Poids", names="Actif", title="Allocation Optimale", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
            
            # Métriques Optimiseur
            c1, c2, c3 = st.columns(3)
            c1.metric("Rendement Attendu", f"{p[0]:.2%}")
            c2.metric("Volatilité", f"{p[1]:.2%}")
            c3.metric("Ratio de Sharpe", f"{p[2]:.2f}")

    st.markdown("---")

    # --- 3. Analyse Détaillée (Métriques & Corrélations) ---
    st.markdown("### 📊 Analyse Approfondie du Portefeuille")

    # Si pas optimisé, on utilise l'équipondéré
    if 'opt_weights' in st.session_state:
        final_weights = np.array([st.session_state['opt_weights'].get(t, 0) for t in prices.columns])
    else:
        final_weights = np.array([1/len(tickers)] * len(tickers))
        st.info("Affichage basé sur un portefeuille Équipondéré (lancez l'optimisation pour voir la différence).")

    # Création indice synthétique
    port_ret = returns.dot(final_weights)
    port_cum = (1 + port_ret).cumprod()
    
    # Calcul des métriques complètes
    m = compute_strategy_metrics(port_ret, port_cum)

    # --- A. Affichage Métriques Avancées ---
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    col_m1.metric("CAGR (Annuel)", f"{m['cagr']:.2%}")
    col_m2.metric("Sortino Ratio", f"{m['sortino']:.2f}", help="Performance ajustée du risque de baisse uniquement.")
    col_m3.metric("Calmar Ratio", f"{m['calmar']:.2f}", help="Rendement annuel / Max Drawdown.")
    col_m4.metric("Skewness", f"{m['skew']:.2f}", help="Asymétrie : < 0 signifie risque de krach fréquent.")
    col_m5.metric("Kurtosis", f"{m['kurtosis']:.2f}", help="Queue de distribution : > 3 signifie événements extrêmes fréquents.")

    # --- B. Corrélations ---
    st.subheader("🔗 Corrélations")
    
    tab_corr1, tab_corr2 = st.tabs(["Matrice (Heatmap)", "Dynamique (Rolling)"])
    
    with tab_corr1:
        corr_matrix = returns.corr()
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="Matrice de Corrélation Statique")
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab_corr2:
        st.caption("Visualisez comment la corrélation entre deux actifs évolue dans le temps.")
        c_a, c_b, c_w = st.columns([1, 1, 2])
        asset_a = c_a.selectbox("Actif A", prices.columns, index=0)
        # Sélection par défaut du 2ème actif s'il existe
        idx_b = 1 if len(prices.columns) > 1 else 0
        asset_b = c_b.selectbox("Actif B", prices.columns, index=idx_b)
        window = c_w.slider("Fenêtre glissante (jours)", 30, 365, 90)

        if asset_a != asset_b:
            # Calcul Rolling Correlation
            rolling_corr = returns[asset_a].rolling(window).corr(returns[asset_b]).dropna()
            
            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, mode='lines', name=f"Corr {asset_a}/{asset_b}"))
            
            # Ajout ligne zéro
            fig_roll.add_hline(y=0, line_dash="dot", line_color="white")
            fig_roll.add_hline(y=1, line_dash="dot", line_color="gray", opacity=0.3)
            fig_roll.add_hline(y=-1, line_dash="dot", line_color="gray", opacity=0.3)
            
            fig_roll.update_layout(
                title=f"Corrélation Glissante ({window} jours) : {asset_a} vs {asset_b}",
                yaxis_title="Corrélation (-1 à +1)",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig_roll, use_container_width=True)
        else:
            st.info("Sélectionnez deux actifs différents.")

# ... (Le début du fichier reste identique jusqu'à la section Monte Carlo) ...

    # --- 4. Module de Stress Test Avancé ---
    st.markdown("### 🔮 Stress Test & Gestion des Risques")
    
    # Onglets pour séparer les approches
    tab_mc, tab_crash = st.tabs(["🎲 Simulation Monte Carlo (Futur Probable)", "💥 Crash Test (Scénarios Historiques)"])

    # === TAB 1 : MONTE CARLO ===
    with tab_mc:
        st.caption("Projection de 500 scénarios possibles sur 1 an, basée sur la volatilité actuelle.")
        
        col_sim1, col_sim2 = st.columns([1, 3])
        
        with col_sim1:
            sim_years = st.slider("Horizon (Années)", 1, 5, 1)
            n_sims = 500 # Fixe pour la performance
            
            if st.button("🚀 Lancer Simulation"):
                with st.spinner("Calcul des trajectoires..."):
                    # On utilise l'indice synthétique calculé plus haut
                    current_capital = port_cum.iloc[-1] * amount
                    
                    # Simulation sur les rendements
                    # On recrée une série fictive partant du capital actuel
                    sim_df = run_monte_carlo(port_cum * amount, days=252*sim_years, simulations=n_sims)
                    
                    # Stockage en session pour ne pas recalculer à chaque interaction
                    st.session_state['mc_results'] = sim_df
                    st.session_state['mc_capital'] = current_capital

        with col_sim2:
            if 'mc_results' in st.session_state:
                sim_df = st.session_state['mc_results']
                start_cap = st.session_state['mc_capital']
                
                # --- GRAPHIQUE CÔNE D'INCERTITUDE ---
                # Au lieu de 500 lignes, on affiche des zones de centiles (5%, 25%, 50%, 75%, 95%)
                
                # Calcul des centiles jour par jour
                p05 = sim_df.quantile(0.05, axis=1)
                p25 = sim_df.quantile(0.25, axis=1)
                p50 = sim_df.median(axis=1) # Scénario Central
                p75 = sim_df.quantile(0.75, axis=1)
                p95 = sim_df.quantile(0.95, axis=1)
                
                fig_mc = go.Figure()
                
                # Zone extrême (5% - 95%)
                fig_mc.add_trace(go.Scatter(
                    x=sim_df.index, y=p95, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
                ))
                fig_mc.add_trace(go.Scatter(
                    x=sim_df.index, y=p05, mode='lines', line=dict(width=0), fill='tonexty', 
                    fillcolor='rgba(255, 0, 0, 0.1)', name='Intervalle 90%'
                ))
                
                # Zone centrale (25% - 75%)
                fig_mc.add_trace(go.Scatter(
                    x=sim_df.index, y=p75, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
                ))
                fig_mc.add_trace(go.Scatter(
                    x=sim_df.index, y=p25, mode='lines', line=dict(width=0), fill='tonexty', 
                    fillcolor='rgba(0, 195, 255, 0.2)', name='Intervalle 50%'
                ))
                
                # Médiane
                fig_mc.add_trace(go.Scatter(
                    x=sim_df.index, y=p50, mode='lines', line=dict(color='white', width=2), name='Scénario Médian'
                ))
                
                fig_mc.update_layout(
                    template="plotly_dark", 
                    title=f"Projection du Capital ({sim_years} ans)", 
                    yaxis_title="Valeur (€)",
                    height=400,
                    hovermode="x unified"
                )
                st.plotly_chart(fig_mc, use_container_width=True)
                
                # --- ANALYSE DES RÉSULTATS ---
                final_values = sim_df.iloc[-1]
                metrics = compute_risk_metrics(final_values, start_cap) # Fonction à importer du backend !
                
                # Import local si besoin, ou assurez-vous de l'importer en haut du fichier
                # from backend.simulation import compute_risk_metrics
                
                st.markdown("#### 📊 Analyse des Risques (Fin de période)")
                m1, m2, m3, m4 = st.columns(4)
                
                m1.metric("Gain Médian", f"{((p50.iloc[-1] - start_cap)/start_cap):+.2%}", f"{p50.iloc[-1]:,.0f} €")
                
                m2.metric(
                    "VaR 95% (Mauvais cas)", 
                    f"{metrics['VaR_95']:.2%}", 
                    f"Capital : {(start_cap * (1+metrics['VaR_95'])):,.0f} €",
                    delta_color="inverse"
                )
                
                m3.metric(
                    "CVaR 95% (Pire cas moyen)", 
                    f"{metrics['CVaR_95']:.2%}",
                    help="Si le marché s'effondre au-delà de la VaR, voici la perte moyenne attendue.",
                    delta_color="inverse"
                )
                
                m4.metric(
                    "Proba. Perte > 20%", 
                    f"{metrics['Prob_Crash']:.1%}",
                    delta_color="inverse"
                )

    # === TAB 2 : CRASH TEST ===
    with tab_crash:
        st.caption("Si une crise historique se reproduisait demain, comment votre portefeuille réagirait-il ?")
        
        # Import local de la fonction crash test
        from backend.simulation import run_historical_crash_test
        
        crash_scenario = st.selectbox("Choisir un scénario catastrophe", ["Subprimes (2008)", "Covid-19 (2020)", "Dotcom (2000)"])
        
        current_capital = port_cum.iloc[-1] * amount
        crash_curve = run_historical_crash_test(current_capital, crash_scenario)
        
        # Calcul de la perte max sur ce scénario
        min_val = crash_curve.min()
        drawdown_crash = (min_val - current_capital) / current_capital
        
        c_crash1, c_crash2 = st.columns([3, 1])
        
        with c_crash1:
            fig_crash = go.Figure()
            fig_crash.add_trace(go.Scatter(
                y=crash_curve, mode='lines', name='Votre Portefeuille', 
                line=dict(color='#ff4b4b', width=3)
            ))
            fig_crash.add_annotation(
                x=crash_curve.idxmin(), y=min_val,
                text=f"Point Bas: {min_val:,.0f} €",
                showarrow=True, arrowhead=1
            )
            fig_crash.update_layout(template="plotly_dark", title=f"Simulation : Impact {crash_scenario}", xaxis_title="Jours de crise")
            st.plotly_chart(fig_crash, use_container_width=True)
            
        with c_crash2:
            st.error(f"Impact Max : {drawdown_crash:.2%}")
            st.write(f"Votre capital tomberait temporairement à **{min_val:,.0f} €**.")
            st.info("💡 Ce test suppose que vos actifs sont corrélés au marché global lors d'un krach.")