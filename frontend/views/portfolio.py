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
    st.title("🧠 Portfolio Optimization & Advanced Analysis")
    
    # --- 1. Configuration ---
    with st.expander("🛠️ Portfolio Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            tickers_input = st.text_input("Assets (separated by commas)", "AAPL, MSFT, GOOGL, NVDA, GLD, BTC-USD")
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        with col2:
            start_date = st.date_input("Start ", pd.to_datetime("2022-01-01"))
            amount = st.number_input("Capital (€)", value=10000)

    if not tickers or len(tickers) < 2:
        st.warning("Minimum 2 assets required.")
        st.stop()

    # --- Loading ---
    prices = load_batch_data(tickers, start_date, pd.to_datetime("today"))
    if prices.empty:
        st.error("Error loading data.")
        return
    
    returns = prices.pct_change().dropna()

    # --- 2. Optimization under Constraints ---
    st.markdown("### 🤖 Automatic allocation")
    
    col_opt1, col_opt2 = st.columns([1, 2])
    
    with col_opt1:
        st.subheader("Parameters")
        obj = st.radio("Objective", ["Maximize Sharpe", "Minimize Volatility (Safety)"])
        target = "sharpe" if "Sharpe" in obj else "min_vol"
        
        st.markdown("**Weight Constraints:**")
        min_w = st.slider("Minimum weight", 0.0, 0.5, 0.01, 0.01)
        max_w = st.slider("Maximum weight per asset", 0.1, 1.0, 1.0, 0.05)

        if st.button("⚡ Optimiser"):
            # Basic constraint check
            if min_w * len(tickers) > 1.0:
                st.error("Impossible the minimum weights exceed 100%.")
            else:
                opt_weights, perf = optimize_portfolio_weights(prices, target, min_weight=min_w, max_weight=max_w)
                if opt_weights:
                    st.session_state['opt_weights'] = opt_weights
                    st.session_state['opt_perf'] = perf
                else:
                    st.error("Didn't find a solution with these constraints.")

    with col_opt2:
        if 'opt_weights' in st.session_state:
            w = st.session_state['opt_weights']
            p = st.session_state['opt_perf']
            
            # Pie Chart
            df_w = pd.DataFrame({"Actif": w.keys(), "Poids": w.values()})
            df_w = df_w[df_w["Poids"] > 0.001]
            
            fig = px.pie(df_w, values="Poids", names="Asset", title="Optimal Allocation", hole=0.4)
            st.plotly_chart(fig, width='stretch')
            
            # Optimizer Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Expected returns", f"{p[0]:.2%}")
            c2.metric("Volatility", f"{p[1]:.2%}")
            c3.metric("Sharpe Ratio", f"{p[2]:.2f}")

    st.markdown("---")

    # --- 3. Detailed Analysis (Metrics & Correlations) ---
    st.markdown("### 📊 In-depth Portfolio Analysis")

    # If not optimized, use equal weighting
    if 'opt_weights' in st.session_state:
        final_weights = np.array([st.session_state['opt_weights'].get(t, 0) for t in prices.columns])
    else:
        final_weights = np.array([1/len(tickers)] * len(tickers))
        st.info("Equiponderated portfolio.")

    port_ret = returns.dot(final_weights)
    port_cum = (1 + port_ret).cumprod()
    
    # Complete Metrics
    m = compute_strategy_metrics(port_ret, port_cum)

    # --- Advanced Metrics Display ---
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    col_m1.metric("CAGR (Annuel)", f"{m['cagr']:.2%}")
    col_m2.metric("Sortino Ratio", f"{m['sortino']:.2f}", help="Performance adjusted for downside risk only.")
    col_m3.metric("Calmar Ratio", f"{m['calmar']:.2f}", help="Annual return / Max Drawdown.")
    col_m4.metric("Skewness", f"{m['skew']:.2f}", help="Asymmetry: < 0 indicates frequent crash risk.")
    col_m5.metric("Kurtosis", f"{m['kurtosis']:.2f}", help="Tail of distribution: > 3 indicates frequent extreme events.")

    # --- B. Correlations ---
    st.subheader("🔗 Correlations")
    
    tab_corr1, tab_corr2 = st.tabs(["Matrix (Heatmap)", "Dynamic (Rolling)"])
    
    with tab_corr1:
        corr_matrix = returns.corr()
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="Static Correlation Matrix")
        st.plotly_chart(fig_corr, width='stretch')

    with tab_corr2:
        st.caption("How correlations between two assets evolve over time.")
        c_a, c_b, c_w = st.columns([1, 1, 2])
        asset_a = c_a.selectbox("Asset A", prices.columns, index=0)
        # Default selection of the 2nd asset if it exists
        idx_b = 1 if len(prices.columns) > 1 else 0
        asset_b = c_b.selectbox("Actif B", prices.columns, index=idx_b)
        window = c_w.slider("sliding window (days)", 30, 365, 90)

        if asset_a != asset_b:
            # Calcul Rolling Correlation
            rolling_corr = returns[asset_a].rolling(window).corr(returns[asset_b]).dropna()
            
            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, mode='lines', name=f"Corr {asset_a}/{asset_b}"))
            
            fig_roll.add_hline(y=0, line_dash="dot", line_color="white")
            fig_roll.add_hline(y=1, line_dash="dot", line_color="gray", opacity=0.3)
            fig_roll.add_hline(y=-1, line_dash="dot", line_color="gray", opacity=0.3)
            
            fig_roll.update_layout(
                title=f"Correlation ({window} days) : {asset_a} vs {asset_b}",
                yaxis_title="Correlation (-1 to +1)",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig_roll, width='stretch')
        else:
            st.info("Select two different assets.")


    # --- 4. Advanced Stress Test Module ---
    st.markdown("### 🔮 Stress Test & Risk Management")
    
    # Tabs to separate approaches
    tab_mc, tab_crash = st.tabs(["🎲 Monte Carlo Simulation (Future Probable)", "💥 Crash Test (Historical Scenarios)"])
    
    # === TAB 1 : MONTE CARLO ===
    with tab_mc:
        st.caption("Projection of 500 possible scenarios over 1 year, based on current volatility.")
        
        col_sim1, col_sim2 = st.columns([1, 3])
        
        with col_sim1:
            sim_years = st.slider("Horizon (Years)", 1, 5, 1)
            n_sims = 500 # Fixed for performance
            
            if st.button("🚀 Start Simulation"):
                with st.spinner("Calculating trajectories..."):
                    # We use the synthetic index calculated above
                    current_capital = port_cum.iloc[-1] * amount
                    sim_df = run_monte_carlo(port_cum * amount, days=252*sim_years, simulations=n_sims)
                    
                    st.session_state['mc_results'] = sim_df
                    st.session_state['mc_capital'] = current_capital

        with col_sim2:
            if 'mc_results' in st.session_state:
                sim_df = st.session_state['mc_results']
                start_cap = st.session_state['mc_capital']
                
                # --- GRAPHICS ---
                # quantiles (5%, 25%, 50%, 75%, 95%)
                p05 = sim_df.quantile(0.05, axis=1)
                p25 = sim_df.quantile(0.25, axis=1)
                p50 = sim_df.median(axis=1) # Central line
                p75 = sim_df.quantile(0.75, axis=1)
                p95 = sim_df.quantile(0.95, axis=1)
                
                fig_mc = go.Figure()
                
                # Extremal areas (5% - 95%)
                fig_mc.add_trace(go.Scatter(
                    x=sim_df.index, y=p95, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
                ))
                fig_mc.add_trace(go.Scatter(
                    x=sim_df.index, y=p05, mode='lines', line=dict(width=0), fill='tonexty', 
                    fillcolor='rgba(255, 0, 0, 0.1)', name='Intervalle 90%'
                ))
                
                # Central areas (25% - 75%)
                fig_mc.add_trace(go.Scatter(
                    x=sim_df.index, y=p75, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
                ))
                fig_mc.add_trace(go.Scatter(
                    x=sim_df.index, y=p25, mode='lines', line=dict(width=0), fill='tonexty', 
                    fillcolor='rgba(0, 195, 255, 0.2)', name='Intervalle 50%'
                ))
                
                # Median
                fig_mc.add_trace(go.Scatter(
                    x=sim_df.index, y=p50, mode='lines', line=dict(color='white', width=2), name='Scenario Median'
                ))
                
                fig_mc.update_layout(
                    template="plotly_dark", 
                    title=f"Projection of capital ({sim_years} years)", 
                    yaxis_title="Value (€)",
                    height=400,
                    hovermode="x unified"
                )
                st.plotly_chart(fig_mc, width='stretch')
                
                # --- Results Metrics ---
                final_values = sim_df.iloc[-1]
                metrics = compute_risk_metrics(final_values, start_cap)
                
                st.markdown("#### 📊 Risk Analysis (End of Period)")
                m1, m2, m3, m4 = st.columns(4)
                
                m1.metric("Median Gain", f"{((p50.iloc[-1] - start_cap)/start_cap):+.2%}", f"{p50.iloc[-1]:,.0f} €")
                
                m2.metric(
                    "VaR 95% (Worst case)", 
                    f"{metrics['VaR_95']:.2%}", 
                    f"Capital : {(start_cap * (1+metrics['VaR_95'])):,.0f} €",
                    delta_color="inverse"
                )
                
                m3.metric(
                    "CVaR 95% (Worst case average)", 
                    f"{metrics['CVaR_95']:.2%}",
                    help="If the market crashes beyond the VaR, this is the average loss expected.",
                    delta_color="inverse"
                )
                
                m4.metric(
                    "Proba. Loss > 20%", 
                    f"{metrics['Prob_Crash']:.1%}",
                    delta_color="inverse"
                )

    # === TAB 2 : CRASH TEST ===
    with tab_crash:
        st.caption("If a historical crisis happened tomorrow, how would your portfolio react?")
        
        # Import local
        from backend.simulation import run_historical_crash_test
        
        crash_scenario = st.selectbox("Choose a scenario", ["Subprimes (2008)", "Covid-19 (2020)", "Dotcom (2000)"])
        
        current_capital = port_cum.iloc[-1] * amount
        crash_curve = run_historical_crash_test(current_capital, crash_scenario)
        
        # max drawdown during crisis
        min_val = crash_curve.min()
        drawdown_crash = (min_val - current_capital) / current_capital
        
        c_crash1, c_crash2 = st.columns([3, 1])
        
        with c_crash1:
            fig_crash = go.Figure()
            fig_crash.add_trace(go.Scatter(
                y=crash_curve, mode='lines', name='Your Portfolio', 
                line=dict(color='#ff4b4b', width=3)
            ))
            fig_crash.add_annotation(
                x=crash_curve.idxmin(), y=min_val,
                text=f"Down point: {min_val:,.0f} €",
                showarrow=True, arrowhead=1
            )
            fig_crash.update_layout(template="plotly_dark", title=f"Simulation : Impact {crash_scenario}", xaxis_title="Crisis day")
            st.plotly_chart(fig_crash, width='stretch')
            
        with c_crash2:
            st.error(f"Max Impact : {drawdown_crash:.2%}")
            st.write(f"Your capital would fall to **{min_val:,.0f} €**.")
            st.info("💡 Supposing your assets are correlated with the overall market during a crash.")