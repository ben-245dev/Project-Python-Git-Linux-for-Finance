import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import yfinance as yf

from backend.orders import init_db, place_order, get_transaction_history, get_portfolio_positions, get_current_prices

def page_paper_trading():
    # Initialisation 
    init_db()

    st.title("🎮 Paper Trading Simulator")
    st.markdown("---")

    # ---------------------------------------------------------
    # 1. Make a new order
    # ---------------------------------------------------------
    with st.expander("📝 Make a new order", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ticker = st.text_input("Ticker", "AAPL").upper()
        with col2:
            action = st.selectbox("Action", ["BUY", "SELL"])
        with col3:
            qty = st.number_input("Quantity", min_value=0.01, value=1.0, step=1.0)
        
        # auto-fill current market price
        current_market_price = 0.0
        if ticker:
            try:
                # last price
                t = yf.Ticker(ticker)
                # fast_info is quicker than history for the last_price
                if hasattr(t, 'fast_info') and t.fast_info.last_price:
                     current_market_price = t.fast_info.last_price
                else:
                    hist = t.history(period="5d") # 5d to cover the weekend
                    if not hist.empty:
                        current_market_price = hist["Close"].iloc[-1]
            except:
                pass

        with col4:
            price_input = st.number_input("Execution price ($)", value=float(current_market_price), min_value=0.0, format="%.2f")

        if st.button("Validate Order"):
            if price_input > 0:
                success, message = place_order(ticker, action, qty, price_input)
                
                if success:
                    st.success(f"✅ {message} ({action} {qty} {ticker} @ {price_input:.2f} $)")
                    st.rerun() # Refresh the page to update portfolio
                else:
                    st.error(f"⛔ Order rejected: {message}")
            else:
                st.error("Price should be over 0.")

    # ---------------------------------------------------------
    # 2. Current Portfolio & P&L
    # ---------------------------------------------------------
    st.subheader("💼 My Portfolio")
    
    positions_df = get_portfolio_positions()

    if not positions_df.empty:
        # take all tickers to get live prices
        tickers = positions_df["Ticker"].tolist()
        
        with st.spinner("Updating valuations..."):
            live_prices = get_current_prices(tickers)

        # Metrics P&L 
        # Mapping live prices. If not found, we keep 0 (or we could use the PRU to avoid -100%)
        positions_df["Prix Actuel"] = positions_df["Ticker"].map(live_prices).fillna(0)
        
        # Visual safety: if current price is 0 (API bug), we avoid showing a 100% loss
        # We can choose to use PRU as a temporary fallback or leave it at 0
        positions_df["Valeur Actuelle"] = positions_df["Quantité"] * positions_df["Prix Actuel"]
        positions_df["P&L ($)"] = positions_df["Valeur Actuelle"] - positions_df["Investi"]
        
        # Avoid division by zero
        positions_df["P&L (%)"] = positions_df.apply(
            lambda x: (x["P&L ($)"] / x["Investi"] * 100) if x["Investi"] > 0 else 0, axis=1
        )

        # Total
        total_invested = positions_df["Investi"].sum()
        total_value = positions_df["Valeur Actuelle"].sum()
        total_pnl = total_value - total_invested
        
        # Plot metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total value", f"{total_value:,.2f} $")
        m2.metric("Total Invested", f"{total_invested:,.2f} $")
        m3.metric("P&L Latent Global", f"{total_pnl:+,.2f} $", 
                  delta_color="normal" if total_pnl >= 0 else "inverse")

        st.markdown("### Position details")
        
        # Styling of DataFrame (Colors for P&L)
        def color_pnl(val):
            color = '#00ff00' if val > 0 else '#ff4b4b' if val < 0 else 'white'
            return f'color: {color}'

        # Formated DataFrame display
        st.dataframe(
            positions_df.style.format({
                "Quantité": "{:.4f}",
                "PRU (Prix Moyen)": "{:.2f} $",
                "Investi": "{:.2f} $",
                "Prix Actuel": "{:.2f} $",
                "Valeur Actuelle": "{:.2f} $",
                "P&L ($)": "{:+.2f} $",
                "P&L (%)": "{:+.2f} %"
            }).map(color_pnl, subset=['P&L ($)', 'P&L (%)']),
            width='stretch'
        )

        # Pie Chart
        if total_value > 0:
            fig = go.Figure(data=[go.Pie(labels=positions_df["Ticker"], values=positions_df["Valeur Actuelle"], hole=.4)])
            fig.update_layout(title="Asset allocation", template="plotly_dark", height=350)
            st.plotly_chart(fig, width='stretch')

    else:
        st.info("Empty portfolio. Make a first order to begin !")

    # ---------------------------------------------------------
    # 3. Historic
    # ---------------------------------------------------------
    st.markdown("---")
    with st.expander("📜 Historical transaction"):
        history_df = get_transaction_history()
        st.dataframe(history_df, width='stretch')