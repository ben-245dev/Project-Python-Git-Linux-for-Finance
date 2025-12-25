import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Dashboard Financier", layout="wide")
st.title("Dashboard financier mono-actif")

# Sidebar : inputs user
st.sidebar.title("Options")
ticker = st.sidebar.text_input("Ticker ex:(AAPL, EURUSD=X, BTC-USD)", "AAPL")
start_date = st.sidebar.date_input("Début", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("Fin", pd.to_datetime("2023-12-31"))
user_note = st.sidebar.text_area("Notes personnelles (facultatif)")

# Paramètres des indicateurs / stratégies
st.sidebar.markdown("### Indicateurs & Stratégies")
period_sma_fast = st.sidebar.number_input("SMA rapide", min_value=2, max_value=100, value=20)
period_sma_slow = st.sidebar.number_input("SMA lente", min_value=5, max_value=200, value=50)
period_boll = st.sidebar.number_input("Période Bollinger", min_value=5, max_value=100, value=20)
std_boll = st.sidebar.number_input("Écart-type Bollinger", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
rsi_period = st.sidebar.number_input("RSI (période)", min_value=3, max_value=50, value=14)

strategy = st.sidebar.selectbox(
    "Stratégie de trading à backtester",
    ["Buy & Hold", f"Croisement SMA{period_sma_fast}/SMA{period_sma_slow}", f"Bollinger {period_boll}, écart-type {std_boll}"]
)

show_sma = st.sidebar.checkbox(f"Moyenne mobile simple (SMA{period_sma_fast})", value=True)
show_ema = st.sidebar.checkbox(f"Moyenne mobile exp. (EMA{period_sma_fast})")
show_boll = st.sidebar.checkbox(f"Bandes de Bollinger ({period_boll}, {std_boll})", value=True)
show_rsi = st.sidebar.checkbox(f"RSI ({rsi_period} jours)")

if st.sidebar.button("Charger les données"):
    with st.spinner(f"Téléchargement pour {ticker}…"):
        df = yf.download(ticker, start=start_date, end=end_date, group_by="ticker")
        if df.empty:
            st.error("Aucune donnée trouvée pour ce ticker ou cette période.")
            st.stop()
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join([str(l) for l in col if l]) for col in df.columns.values]

        def detect_col(label):
            col_multi = f"{ticker}_{label}"
            if col_multi in df.columns:
                return col_multi
            elif label in df.columns:
                return label
            else:
                return None
        close_col = detect_col("Close")
        open_col = detect_col("Open")
        high_col = detect_col("High")
        low_col = detect_col("Low")
        volume_col = detect_col("Volume")
        if close_col is None:
            st.error("La colonne 'Close' n'a pas été trouvée dans les données.")
            st.stop()
        for col_var in [open_col, high_col, low_col, close_col, volume_col]:
            if col_var is not None:
                df[col_var] = pd.to_numeric(df[col_var], errors='coerce')

        # Indicateurs
        df['SMA_FAST'] = df[close_col].rolling(period_sma_fast).mean()
        df['SMA_SLOW'] = df[close_col].rolling(period_sma_slow).mean()
        df['EMA_FAST'] = df[close_col].ewm(span=period_sma_fast, adjust=False).mean()
        sma = df[close_col].rolling(period_boll).mean()
        std = df[close_col].rolling(period_boll).std()
        df['BollUpper'] = sma + std_boll * std
        df['BollLower'] = sma - std_boll * std
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Backtest Buy & Hold
        df['Backtest_BuyHold'] = df[close_col] / df[close_col].iloc[0]

        # Backtest Croisement SMA
        df['signal_sma'] = 0
        df.loc[df['SMA_FAST'] > df['SMA_SLOW'], 'signal_sma'] = 1
        df['signal_sma'] = df['signal_sma'].shift(1, fill_value=0)
        df['Backtest_SMA'] = 1.0
        df['returns'] = df[close_col].pct_change().fillna(0)
        for i in range(1, len(df)):
            if df['signal_sma'].iloc[i] == 1:
                df['Backtest_SMA'].iloc[i] = df['Backtest_SMA'].iloc[i-1] * (1 + df['returns'].iloc[i])
            else:
                df['Backtest_SMA'].iloc[i] = df['Backtest_SMA'].iloc[i-1]

        # Backtest Bollinger
        df['signal_boll'] = 0  # 1=long, 0=hors marché
        # Entrée si prix < BollLower, sortie si prix > BollUpper
        in_trade = False
        for i in range(1, len(df)):
            if not in_trade and df[close_col].iloc[i] < df['BollLower'].iloc[i]:
                in_trade = True
            elif in_trade and df[close_col].iloc[i] > df['BollUpper'].iloc[i]:
                in_trade = False
            df['signal_boll'].iloc[i] = 1 if in_trade else 0
        df['Backtest_Boll'] = 1.0
        for i in range(1, len(df)):
            if df['signal_boll'].iloc[i] == 1:
                df['Backtest_Boll'].iloc[i] = df['Backtest_Boll'].iloc[i-1] * (1 + df['returns'].iloc[i])
            else:
                df['Backtest_Boll'].iloc[i] = df['Backtest_Boll'].iloc[i-1]

        # KPIs
        st.subheader(f"Indicateurs pour {ticker}")
        total_return = (df[close_col].iloc[-1]/df[close_col].iloc[0]-1) * 100
        st.metric("Rendement total", f"{total_return:.2f} %")
        volatility = df[close_col].pct_change().std() * (252 ** 0.5) * 100
        st.metric("Volatilité annualisée", f"{volatility:.2f} %")
        max_drawdown = ((df[close_col] / df[close_col].cummax()) - 1).min() * 100
        st.metric("Drawdown max.", f"{max_drawdown:.2f} %")
        daily_returns = df[close_col].pct_change()
        risk_free_rate = 0
        sharpe = (daily_returns.mean() - risk_free_rate/252) / daily_returns.std() * np.sqrt(252)
        st.metric("Sharpe annualisé", f"{sharpe:.2f}")
        st.metric("Séances positives", int((daily_returns > 0).sum()))
        st.metric("Séances négatives", int((daily_returns < 0).sum()))

        # Affichage courbes prix, SMA, EMA, Bollinger
        st.subheader("Prix & indicateurs")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['Date'], y=df[close_col], mode='lines', name='Close'))
        if show_sma:
            fig2.add_trace(go.Scatter(x=df['Date'], y=df['SMA_FAST'], mode='lines', name=f"SMA{period_sma_fast}"))
        if show_ema:
            fig2.add_trace(go.Scatter(x=df['Date'], y=df['EMA_FAST'], mode='lines', name=f"EMA{period_sma_fast}"))
        if show_boll:
            fig2.add_trace(go.Scatter(x=df['Date'], y=df['BollUpper'], mode='lines', name='Boll Upper', line=dict(dash='dash')))
            fig2.add_trace(go.Scatter(x=df['Date'], y=df['BollLower'], mode='lines', name='Boll Lower', line=dict(dash='dash')))
        st.plotly_chart(fig2, use_container_width=True)

        # Graphique Chandelier
        st.subheader("Chandelier (OHLC)")
        if all(x is not None for x in [open_col, high_col, low_col, close_col]) and 'Date' in df.columns:
            fig = go.Figure(data=[go.Candlestick(
                x=pd.to_datetime(df['Date']),
                open=df[open_col], high=df[high_col],
                low=df[low_col], close=df[close_col]
            )])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Colonnes OHLC nécessaires pour afficher le chandelier.")

        # RSI
        if show_rsi and 'RSI' in df:
            st.subheader(f"RSI ({rsi_period} jours)")
            st.line_chart(df['RSI'])

        # Volume
        if volume_col is not None:
            st.subheader("Volume échangé")
            st.line_chart(df[volume_col])

        # Backtest stratégie
        st.subheader("Backtest de la stratégie choisie")
        fig_bt = go.Figure()
        if strategy == "Buy & Hold":
            fig_bt.add_trace(go.Scatter(
                x=df['Date'], y=df['Backtest_BuyHold'], mode='lines', name='Buy & Hold'
            ))
        elif strategy.startswith("Croisement"):
            fig_bt.add_trace(go.Scatter(
                x=df['Date'], y=df['Backtest_SMA'], mode='lines', name='SMA Fast > SMA Slow'
            ))
            fig_bt.add_trace(go.Scatter(
                x=df['Date'], y=df['Backtest_BuyHold'], mode='lines', name='Buy & Hold', line=dict(dash='dot')
            ))
        elif strategy.startswith("Bollinger"):
            fig_bt.add_trace(go.Scatter(
                x=df['Date'], y=df['Backtest_Boll'], mode='lines', name='Bollinger Trading'
            ))
            fig_bt.add_trace(go.Scatter(
                x=df['Date'], y=df['Backtest_BuyHold'], mode='lines', name='Buy & Hold', line=dict(dash='dot')
            ))
        st.plotly_chart(fig_bt, use_container_width=True)
        final_perf = df[
            "Backtest_BuyHold" if strategy=="Buy & Hold"
            else "Backtest_SMA" if strategy.startswith("Croisement")
            else "Backtest_Boll"
        ].iloc[-1]
        st.metric("Performance finale stratégie", f"{(final_perf-1)*100:.2f} %")

        # Tableau des principales colonnes + export CSV et notes
        st.subheader("Table des données principales")
        affich_cols = ['Date']
        for col_var in [open_col, high_col, low_col, close_col, volume_col,
                        'SMA_FAST','SMA_SLOW','EMA_FAST','BollUpper','BollLower','RSI','Backtest_BuyHold','Backtest_SMA','Backtest_Boll']:
            if col_var is not None and col_var in df.columns:
                affich_cols.append(col_var)
        st.write(df[affich_cols].tail(20))
        st.download_button("Télécharger le CSV", df.to_csv(index=False), f"{ticker}_data.csv")
        if user_note.strip():
            st.subheader("Notes")
            st.write(user_note)
    st.success("Dashboard généré avec succès !")
else:
    st.markdown("> Utilise la barre latérale pour choisir un ticker, une période et cliquer sur « Charger les données ».")

st.markdown("---\n_Créé avec Streamlit & yfinance pour une analyse et un backtesting trading complet._")
