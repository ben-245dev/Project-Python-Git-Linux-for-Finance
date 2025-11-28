import streamlit as st
from strategies.momentum import MomentumStrategy
from strategies.bollinger import BollingerStrategy
from strategies.meanreversion import MeanReversionStrategy
from strategies.breakout import BreakoutStrategy
from strategies.base import TradingStrategy
from core.data import fetch_data, detect_col
from core.perf import summarize_perf, plot_trade_markers, plot_equity_curve, plot_signals_overlay

def run():
    st.title("Backtest Stratégies de Trading")
    tickers = st.sidebar.text_input("Ticker(s) (séparés par ,)", "AAPL")
    start = st.sidebar.date_input("Début", value="2022-01-01")
    end = st.sidebar.date_input("Fin", value="2023-12-31")
    strat_choices = {
        "Momentum": MomentumStrategy,
        "Bollinger": BollingerStrategy,
        "Mean Reversion": MeanReversionStrategy,
        "Breakout": BreakoutStrategy
    }
    strat_name = st.sidebar.selectbox("Stratégie", list(strat_choices.keys()))
    price_type = st.sidebar.selectbox("Type prix", ["Close", "Open", "High", "Low"], 0)

    # Paramètres spécifiques par stratégie
    params = {}
    if strat_name == "Momentum":
        params["sma_fast"] = st.sidebar.number_input("SMA Fast", 2, 100, 20)
        params["sma_slow"] = st.sidebar.number_input("SMA Slow", 2, 200, 50)
        params["threshold"] = st.sidebar.slider("Seuil (%)", -20, 20, 0)
    if strat_name == "Bollinger":
        params["period"] = st.sidebar.number_input("Période", 5, 100, 20)
        params["nb_std"] = st.sidebar.number_input("Ecart-type", 1.0, 5.0, 2.0, 0.5)
    if strat_name == "Mean Reversion":
        params["lookback"] = st.sidebar.number_input("Période Rolling", 10, 100, 20)
        params["z_entry"] = st.sidebar.slider("Entrée", 0.5, 3.0, 1.5, 0.05)
        params["z_exit"] = st.sidebar.slider("Sortie", 0.1, 2.0, 0.5, 0.05)
    if strat_name == "Breakout":
        params["lookback"] = st.sidebar.number_input("Lookback", 5, 100, 20)

    strat = strat_choices[strat_name](**params)
    user_note = st.sidebar.text_area("Notes")

    if st.sidebar.button("Lancer le backtest"):
        tickers_list = [t.strip() for t in tickers.split(',')]
        for ticker in tickers_list:
            df = fetch_data(ticker, start, end)
            price_col = detect_col(df, ticker, price_type)
            df = strat.generate_signals(df, price_col)
            equity = strat.compute_equity_curve(df, price_col)
            equity_bnh = df[price_col] / df[price_col].iloc[0]
            perf, drawdown = summarize_perf(equity)
            perf_bnh, dd_bnh = summarize_perf(equity_bnh)

            st.header(f"{ticker} : {strat_name}")
            # Equity curve
            plot_equity_curve(st, df, equity, equity_bnh, strat_name)
            # Signal map
            plot_signals_overlay(st, df, price_col, strat_name)
            # KPIs
            st.write(f"Performance : `{perf:.2f}%` / Drawdown : `{drawdown:.2f}%` | Buy & Hold `{perf_bnh:.2f}%` / `{dd_bnh:.2f}%`")
            st.write(df.tail(20))
            if user_note.strip():
                st.subheader("Notes")
                st.write(user_note)

# Les autres pages (analyse.py, screeners.py, portfolio.py...) suivent la même logique.
