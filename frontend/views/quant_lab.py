import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import pandas_ta as ta

from backend.data_loader import load_ticker_data, load_batch_data
from backend.quant_stats import calculate_cointegration, calculate_zscore, calculate_kelly_criterion

def page_quant_lab():
    st.title("🧪 Laboratoire Quantitatif")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["⚡ Arbitrage Statistique (Pairs)", "🔍 Screener Algo", "📐 Money Management (Kelly)"])

    # --- TAB 1 : PAIRS TRADING ---
    with tab1:
        st.subheader("Stratégie de retour à la moyenne (Mean Reversion)")
        st.caption("Cherche à exploiter l'écart temporaire entre deux actifs corrélés.")
        
        c1, c2, c3 = st.columns(3)
        asset_a = c1.text_input("Actif A", "KO")
        asset_b = c2.text_input("Actif B", "PEP")
        period = c3.selectbox("Période", ["1y", "2y", "5y"], index=0)

        if st.button("Analyser la Paire"):
            df_a = load_ticker_data(asset_a, period)
            df_b = load_ticker_data(asset_b, period)

            if not df_a.empty and not df_b.empty:
                # Calculs
                series_a = df_a["Close"]
                series_b = df_b["Close"]
                
                # Cointégration
                pvalue, hedge_ratio = calculate_cointegration(series_a, series_b)
                
                # Calcul du Spread (Écart)
                # Spread = Y - (HedgeRatio * X)
                spread = series_b - (hedge_ratio * series_a)
                zscore = calculate_zscore(spread, window=30)

                # Affichage Résultats
                col_res1, col_res2 = st.columns(2)
                col_res1.metric("Hedge Ratio", f"{hedge_ratio:.4f}")
                
                is_coint = pvalue < 0.05
                col_res2.metric("P-Value (Cointégration)", f"{pvalue:.4f}", 
                                delta="Cointégré (Trading possible)" if is_coint else "Pas de relation stable",
                                delta_color="normal" if is_coint else "inverse")

                # Graphique du Z-Score (Signal de trading)
                fig_z = go.Figure()
                fig_z.add_trace(go.Scatter(x=zscore.index, y=zscore, name="Z-Score Spread", line=dict(color="#00c3ff")))
                
                # Lignes de signal (+2 / -2 écarts types)
                fig_z.add_hline(y=2, line_dash="dot", line_color="red", annotation_text="Vente Spread")
                fig_z.add_hline(y=-2, line_dash="dot", line_color="green", annotation_text="Achat Spread")
                fig_z.add_hline(y=0, line_color="gray", opacity=0.5)
                
                fig_z.update_layout(title="Signaux de Trading (Z-Score)", template="plotly_dark", height=400)
                st.plotly_chart(fig_z, use_container_width=True)

            else:
                st.error("Impossible de récupérer les données pour ces tickers.")

    # --- TAB 2 : SCREENER ---
    with tab2:
        st.subheader("Scanner de Marché Automatisé")
        st.caption("Détecte les opportunités basées sur RSI et SMA.")
        
        # Liste par défaut (Secteur Tech US)
        default_list = "AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, NFLX, AMD, INTC, QCOM"
        tickers_scan = st.text_area("Liste de surveillance (séparés par virgules)", default_list)
        
        col_scan_btn, _ = st.columns([1, 4])
        
        if col_scan_btn.button("Lancer le Scan"):
            ticker_list = [t.strip().upper() for t in tickers_scan.split(",") if t.strip()]
            
            results = []
            progress_bar = st.progress(0)
            
            for i, ticker in enumerate(ticker_list):
                try:
                    df = load_ticker_data(ticker, "6mo")
                    if not df.empty:
                        # Calcul Indicateurs
                        close = df["Close"]
                        current_price = close.iloc[-1]
                        
                        # RSI (via pandas-ta si dispo ou manuel)
                        # Pour faire simple sans dépendance complexe ici, calcul manuel rapide ou via ta
                        rsi_series = df.ta.rsi(length=14) if hasattr(df, 'ta') else pd.Series()
                        sma_200 = close.rolling(200).mean().iloc[-1] if len(close) > 200 else 0
                        
                        last_rsi = rsi_series.iloc[-1] if not rsi_series.empty else 50
                        
                        # Logique du signal
                        signal = "NEUTRE"
                        if last_rsi < 30: signal = "SURVENDU (Oversold)"
                        elif last_rsi > 70: signal = "SURACHETÉ (Overbought)"
                        
                        trend = "HAUSSIER" if (sma_200 > 0 and current_price > sma_200) else "BAISSIER"
                        
                        results.append({
                            "Ticker": ticker,
                            "Prix": round(current_price, 2),
                            "RSI (14)": round(last_rsi, 2),
                            "Tendance (SMA200)": trend,
                            "Signal RSI": signal
                        })
                except Exception:
                    pass
                
                progress_bar.progress((i + 1) / len(ticker_list))
            
            # Affichage DataFrame stylisé
            if results:
                df_res = pd.DataFrame(results)
                
                # Coloration conditionnelle
                def highlight_signal(val):
                    color = 'green' if 'SURVENDU' in val else 'red' if 'SURACHETÉ' in val else 'white'
                    return f'color: {color}'

                st.dataframe(df_res.style.map(highlight_signal, subset=['Signal RSI']), use_container_width=True)
            else:
                st.warning("Aucun résultat trouvé.")

    # --- TAB 3 : KELLY CRITERION ---
    with tab3:
        st.subheader("Optimisation de la taille de position")
        
        col_k1, col_k2 = st.columns(2)
        win_rate = col_k1.slider("Taux de réussite (Win Rate %)", 0, 100, 50) / 100
        risk_reward = col_k2.number_input("Ratio Gain/Risque (Reward to Risk)", 0.1, 10.0, 2.0, 0.1)
        
        kelly_pct = calculate_kelly_criterion(win_rate, risk_reward)
        safe_kelly = kelly_pct / 2  # "Half Kelly" est souvent préféré par les pros
        
        st.markdown("---")
        
        c_res1, c_res2 = st.columns(2)
        c_res1.metric("Kelly Complet (Théorique)", f"{kelly_pct:.2%}")
        c_res2.metric("Half Kelly (Prudent)", f"{safe_kelly:.2%}", help="Recommandé pour réduire la volatilité")
        
        if kelly_pct <= 0:
            st.error("⚠️ Espérance négative. Ne prenez pas ce trade !")
        else:
            st.success(f"✅ Vous devriez risquer environ {safe_kelly*100:.1f}% de votre capital sur ce trade.")