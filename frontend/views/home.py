import streamlit as st
import plotly.graph_objs as go
from backend.data_loader import get_live_index_data, load_ticker_data, get_price_series
from backend.metrics import compute_drawdown
from frontend.components import render_trading_clocks, render_banner, render_export_button

def page_home():
    st.markdown("## 🏠 Market Dashboard")
    
    # 1. Live Data & Clocks
    live_data = get_live_index_data()
    render_trading_clocks()
    st.markdown("---")
    render_banner(live_data)
    
    # 2. Configuration Actif Principal
    st.sidebar.header("🔎 Principal securities")
    ticker = st.sidebar.text_input("Symbol (ex: AAPL, BTC-USD)", value="BTC-USD", key="home_ticker").upper()
    period = st.sidebar.selectbox("Period", ["1d", "5d", "1mo", "6mo", "1y", "5y"], index=2, key="home_period")
    
    if ticker:
        st.subheader(f"Analysis: {ticker}")
        
        # Chargement des données sans cache long pour avoir le prix frais
        df = load_ticker_data(ticker, period)
        
        if not df.empty:
            close = get_price_series(df, "Close", ticker)
            
            current_price = close.iloc[-1]
            previous_price = close.iloc[-2] if len(close) > 1 else current_price
            delta = current_price - previous_price
            delta_pct = (delta / previous_price) * 100
            
            # Dashboard style
            col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
            col_kpi1.metric("Current Price", f"{current_price:,.2f} $", f"{delta_pct:+.2f}%")
            
            # Calculus Volatility & Max DD for dashboard
            returns = close.pct_change().dropna()
            volatility = returns.std() * (252**0.5) * 100 # Annualized
            _, max_dd = compute_drawdown(close)
            
            col_kpi2.metric("Volatility (An.)", f"{volatility:.2f}%")
            col_kpi3.metric("Max Drawdown", f"{max_dd*100:.2f}%", delta_color="inverse")

            # Principal time series plot
            # Graphs for price evolution and indicators
            st.markdown("### 📉 Price evolution")
            
            fig = go.Figure()
            
            # 1. Raw Values
            fig.add_trace(go.Scatter(
                x=df.index, y=close, 
                mode='lines', name='Price (Raw)', 
                line=dict(color='#00c3ff', width=2)
            ))
            
            # 2. Simple Strategy Overlay (ex: SMA 50 for trend)
            sma = close.rolling(window=50).mean()
            fig.add_trace(go.Scatter(
                x=df.index, y=sma, 
                mode='lines', name='Moving average 50 (Trend)', 
                line=dict(color='#ff9900', width=1, dash='dash')
            ))

            fig.update_layout(
                template="plotly_dark",
                height=500,
                xaxis_title="Date",
                yaxis_title="Price ($)",
                legend=dict(orientation="h", y=1.02)
            )
            st.plotly_chart(fig, width='stretch')
            
            # Bouton export
            render_export_button(df, f"history_{ticker}.csv")
            
        else:
            st.error(f"No Data found for {ticker}.")