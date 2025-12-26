import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from backend.data_loader import load_ticker_data
from backend.strategies import build_strategies
from backend.metrics import compute_strategy_metrics
from backend.forecasting import arima_forecast, ml_predict_direction

def page_strategy():
    st.title("🧪 Laboratoire de Stratégies & Prédictions")
    
    # --- Sidebar Configuration ---
    st.sidebar.header("Configuration")
    ticker = st.sidebar.text_input("Ticker", value="BTC-USD").upper()
    period = st.sidebar.selectbox("Période", ["1y", "2y", "5y", "max"], index=1)
    
    st.sidebar.markdown("---")
    
    # SÉLECTION DE LA STRATÉGIE
    selected_strat = st.sidebar.selectbox(
        "Stratégie à tester", 
        [
            "Adaptive Trend (Long/Short)",
            "MA Crossover", 
            "RSI Reversion", 
            "MACD Trend", 
            "Bollinger Breakout", 
            "Buy & Hold"
        ]
    )
    
    st.sidebar.markdown("### Paramètres de la Stratégie")
    
    # Dictionnaire de paramètres par défaut
    params = {}
    
    # --- Affichage conditionnel des curseurs ---
    
    if selected_strat in ["MA Crossover", "Adaptive Trend (Long/Short)", "Bollinger Breakout"]:
        params["fast_ma"] = st.sidebar.number_input("EMA Rapide (Tendance court terme)", 5, 100, 20)
        params["slow_ma"] = st.sidebar.number_input("EMA Lente (Tendance de fond)", 10, 300, 50)

    if selected_strat == "RSI Reversion":
        params["rsi_len"] = st.sidebar.number_input("Période RSI", 5, 30, 14)
        col_rsi1, col_rsi2 = st.sidebar.columns(2)
        params["rsi_buy"] = col_rsi1.number_input("Seuil Achat (<)", 10, 45, 30)
        params["rsi_sell"] = col_rsi2.number_input("Seuil Vente (>)", 55, 90, 70)
        
    if selected_strat == "MACD Trend":
        params["macd_fast"] = st.sidebar.number_input("MACD Rapide", 5, 50, 12)
        params["macd_slow"] = st.sidebar.number_input("MACD Lent", 10, 100, 26)
        params["macd_sig"] = st.sidebar.number_input("MACD Signal", 5, 50, 9)
        
    if selected_strat == "Bollinger Breakout":
        params["bb_len"] = st.sidebar.number_input("Période Bollinger", 10, 50, 20)
        params["bb_std"] = st.sidebar.number_input("Ecart-type (Std Dev)", 1.0, 4.0, 2.0, 0.1)
        
    if selected_strat == "Adaptive Trend (Long/Short)":
        params["target_vol"] = st.sidebar.slider("Volatilité Cible (%)", 0.5, 5.0, 2.0, 0.1) / 100

    if not ticker:
        st.stop()

    # --- Chargement ---
    with st.spinner("Chargement et calculs..."):
        df = load_ticker_data(ticker, period)
        if df.empty:
            st.error("Données introuvables.")
            st.stop()

        # Calculs
        strategies, df_indicators = build_strategies(df["Close"], params)
        strat_data = strategies[selected_strat]
        metrics = compute_strategy_metrics(strat_data["returns"], strat_data["equity"])

    # Mise à l'échelle
    initial_price = df["Close"].iloc[0]
    strat_equity_scaled = strat_data["equity"] * initial_price
    bh_equity_scaled = strategies["Buy & Hold"]["equity"] * initial_price

    # --- AFFICHAGE ---
    tab_backtest, tab_forecast, tab_ai = st.tabs(["📊 Backtest", "🔮 Prévision (ARIMA)", "🤖 IA"])

    with tab_backtest:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Rendement Total", f"{metrics['total_return']:.2%}")
        col_m2.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
        col_m3.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
        col_m4.metric("Volatilité", f"{metrics['vol']:.2%}")

        # Configuration Graphique
        is_complex = (selected_strat == "Adaptive Trend (Long/Short)")
        rows = 3 if is_complex else 2
        row_heights = [0.6, 0.2, 0.2] if is_complex else [0.7, 0.3]
        
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=row_heights)
        
        # 1. Courbes Equity
        fig.add_trace(go.Scatter(x=strat_equity_scaled.index, y=strat_equity_scaled, name="Stratégie", line=dict(color="#00c3ff", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=bh_equity_scaled.index, y=bh_equity_scaled, name="Buy & Hold", line=dict(color="gray", dash="dot")), row=1, col=1)

        # 2. Signaux Achat/Vente
        signals = strat_data.get("signals")
        if signals is not None:
            buys = df["Close"][signals == 1]
            sells = df["Close"][signals == -1]
            if not buys.empty:
                fig.add_trace(go.Scatter(x=buys.index, y=buys, mode="markers", marker=dict(symbol="triangle-up", color="green", size=12), name="Achat"), row=1, col=1)
            if not sells.empty:
                fig.add_trace(go.Scatter(x=sells.index, y=sells, mode="markers", marker=dict(symbol="triangle-down", color="red", size=12), name="Vente"), row=1, col=1)

        # 3. Indicateurs Spécifiques (Correction ici pour MACD)
        if selected_strat == "RSI Reversion" and "RSI" in df_indicators.columns:
            fig.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators["RSI"], name="RSI", line=dict(color="purple")), row=2, col=1)
            fig.add_hline(y=params.get("rsi_sell", 70), line_dash="dot", row=2, col=1, line_color="red")
            fig.add_hline(y=params.get("rsi_buy", 30), line_dash="dot", row=2, col=1, line_color="green")
            fig.update_yaxes(range=[0, 100], row=2, col=1)

        elif selected_strat == "MACD Trend" and "MACD_Line" in df_indicators.columns:
            fig.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators["MACD_Line"], name="MACD"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators["MACD_Signal"], name="Signal"), row=2, col=1)
            # CORRECTION DU BUG ICI : Utilisation de add_trace(go.Bar)
            fig.add_trace(go.Bar(x=df_indicators.index, y=df_indicators["MACD_Hist"], name="Hist"), row=2, col=1)

        elif selected_strat == "Bollinger Breakout":
            bbu_col = next((c for c in df_indicators.columns if c.startswith("BBU")), None)
            bbl_col = next((c for c in df_indicators.columns if c.startswith("BBL")), None)
            if bbu_col and bbl_col:
                fig.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators[bbu_col], line=dict(width=1, color='rgba(255,255,255,0.3)'), name="Bande Sup"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators[bbl_col], line=dict(width=1, color='rgba(255,255,255,0.3)'), fill='tonexty', fillcolor='rgba(255, 255, 255, 0.05)', name="Bande Inf"), row=1, col=1)
            
            if "Volume" in df.columns:
                fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color='rgba(100,100,100,0.5)', name="Volume"), row=2, col=1)
            else:
                 dd = strat_data["equity"] / strat_data["equity"].cummax() - 1
                 fig.add_trace(go.Scatter(x=dd.index, y=dd, fill='tozeroy', line=dict(color='red', width=0), name="Drawdown"), row=2, col=1)

        elif is_complex:
             if "leverage" in strat_data:
                lev = strat_data["leverage"]
                fig.add_trace(go.Scatter(x=lev.index, y=lev, fill='tozeroy', name="Levier", line=dict(color="#ff9900")), row=2, col=1)
                fig.add_hline(y=1.0, line_dash="dot", line_color="white", row=2, col=1)
             
             if "RSI" in df_indicators.columns:
                fig.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators["RSI"], name="RSI", line=dict(color="purple")), row=3, col=1)
                fig.add_hline(y=50, line_color="white", row=3, col=1)
                fig.update_yaxes(range=[0, 100], row=3, col=1)

        else: # MA Crossover
            if "EMA_Fast" in df_indicators.columns:
                fig.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators["EMA_Fast"], line=dict(color="orange", width=1), name="EMA Fast"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_indicators.index, y=df_indicators["EMA_Slow"], line=dict(color="blue", width=1), name="EMA Slow"), row=1, col=1)
            
            dd = strat_data["equity"] / strat_data["equity"].cummax() - 1
            fig.add_trace(go.Scatter(x=dd.index, y=dd, fill='tozeroy', line=dict(color='red', width=0), name="Drawdown"), row=2, col=1)

        fig.update_layout(height=700 if is_complex else 600, template="plotly_dark", title=f"Backtest : {selected_strat}")
        st.plotly_chart(fig, use_container_width=True)

    # === TAB 2 : FORECAST (ARIMA) ===
    with tab_forecast:
        st.subheader("Prévision Statistique (ARIMA)")
        horizon = st.slider("Horizon (jours)", 7, 90, 30)
        try:
            with st.spinner("Calcul ARIMA..."):
                forecast_df = arima_forecast(df["Close"], horizon=horizon)
            
            if forecast_df is not None:
                last_real_price = df["Close"].iloc[-1]
                target_price = forecast_df["y_pred"].iloc[-1]
                exp_return = (target_price - last_real_price) / last_real_price
                low_return = (forecast_df["ci_low"].iloc[-1] - last_real_price) / last_real_price
                high_return = (forecast_df["ci_high"].iloc[-1] - last_real_price) / last_real_price

                st.markdown(f"**Projections à {horizon} jours** (Prix : {last_real_price:,.2f} $)")
                c1, c2, c3 = st.columns(3)
                c1.metric("Pessimiste", f"{low_return:+.2%}", f"{forecast_df['ci_low'].iloc[-1]:,.2f} $", delta_color="off")
                c2.metric("🎯 Central", f"{exp_return:+.2%}", f"{target_price:,.2f} $", delta_color="normal")
                c3.metric("Optimiste", f"{high_return:+.2%}", f"{forecast_df['ci_high'].iloc[-1]:,.2f} $", delta_color="normal")
                
                fig_f = go.Figure()
                recent_df = df["Close"].tail(90)
                fig_f.add_trace(go.Scatter(x=recent_df.index, y=recent_df, name="Historique", line=dict(color="rgba(255,255,255,0.5)")))
                fig_f.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["y_pred"], name="Prévision", line=dict(color="#00c3ff", dash="dash")))
                fig_f.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["ci_high"], line=dict(width=0), showlegend=False))
                fig_f.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["ci_low"], fill='tonexty', fillcolor='rgba(0, 195, 255, 0.1)', line=dict(width=0), name="Intervalle 95%"))
                fig_f.update_layout(template="plotly_dark", height=450)
                st.plotly_chart(fig_f, use_container_width=True)
            else:
                st.warning("Pas assez de données pour ARIMA.")
        except Exception as e:
            st.error(f"Erreur modèle : {e}")

    # === TAB 3 : IA ===
    with tab_ai:
        st.subheader("🤖 Oracle IA")
        prob_up, accuracy, feature_imp = ml_predict_direction(df)
        if prob_up is not None:
            c_ai1, c_ai2 = st.columns(2)
            with c_ai1:
                fig_g = go.Figure(go.Indicator(mode = "gauge+number", value = prob_up * 100, title = {'text': "Proba Hausse"}, gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#00c3ff"}}))
                fig_g.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_g, use_container_width=True)
            with c_ai2:
                st.metric("Précision", f"{accuracy:.1%}")
                if feature_imp: st.bar_chart(pd.Series(feature_imp))