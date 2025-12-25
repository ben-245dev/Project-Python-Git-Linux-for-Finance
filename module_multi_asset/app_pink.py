import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ------ Palette rose ------
ROSE = "#ea4c89"
ROSE2 = "#f784ab"
ROSE_DARK = "#cf1666"
BG_HIGH = "#ffe3f6"
BG_KPI = "#ffd6eb"
TXT = "#27161d"


st.set_page_config(page_title="Dashboard Trading Quant", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>.block-container{{background: {BG_HIGH};}} .sidebar-content{{background:#fff0fa !important;}}</style>", unsafe_allow_html=True)
st.sidebar.image("logo_dashboard.png", width=180)
st.image("dashboard_cover.png", use_container_width=True)
st.sidebar.title("Navigation")

def fetch_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, group_by="ticker", progress=False)
    if df.empty:
        return df
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(l) for l in col if l]) for col in df.columns.values]
    if "Date" not in df.columns:
        df["Date"] = pd.date_range(start=start_date, periods=len(df), freq="B")
    return df

def detect_col(df, ticker, label):
    label_multi = f"{ticker}_{label}"
    if label_multi in df.columns: return label_multi
    elif label in df.columns: return label
    return None

def summarize_perf(curve):
    vals = pd.Series(curve).dropna()
    if vals.size < 2: return None, None
    last, base = float(vals.iloc[-1]), float(vals.iloc[0])
    perf = (last/base - 1) * 100
    dd = ((vals / vals.cummax()) - 1).min() * 100
    return round(perf, 2), round(dd, 2)

def kpi_block(title, main, sub):
    st.markdown(f"""<div style="padding:10px 12px 8px 12px;
    border-radius:8px;
    background:{BG_KPI};
    border-left:solid 8px {ROSE};
    margin-bottom:10px;">
        <span style="color:{ROSE_DARK};font-size:15px;"><b>{title} :</b></span>
        <span style="font-size:22px;color:{ROSE_DARK};margin-left:10px;">{main}</span>
        <br><span style="font-size:13px;color:#999">{sub}</span>
    </div>""", unsafe_allow_html=True)

class BuyAndHoldStrategy:
    name = "Buy & Hold"
    def generate_signals(self, df, price_col):
        df = df.copy()
        df['signal_bnh'] = 1
        return df

class MomentumStrategy:
    name = "Momentum"
    def __init__(self, sma_fast=20, sma_slow=50, threshold=0):
        self.sma_fast, self.sma_slow, self.threshold = sma_fast, sma_slow, threshold
    def generate_signals(self, df, price_col):
        df = df.copy()
        df['SMA_FAST'] = df[price_col].rolling(self.sma_fast).mean()
        df['SMA_SLOW'] = df[price_col].rolling(self.sma_slow).mean()
        df['signal_momo'] = (df['SMA_FAST'] > df['SMA_SLOW'] + self.threshold).astype(int).shift(1, fill_value=0)
        return df

class BollingerStrategy:
    name = "Bollinger Bands"
    def __init__(self, period=20, nb_std=2): self.period, self.nb_std = period, nb_std
    def generate_signals(self, df, price_col):
        df = df.copy()
        sma = df[price_col].rolling(self.period).mean()
        std = df[price_col].rolling(self.period).std()
        df['BollUpper'] = sma + self.nb_std * std
        df['BollLower'] = sma - self.nb_std * std
        df['signal_boll'] = 0
        in_trade = False
        for i in range(1, len(df)):
            if not in_trade and df[price_col].iloc[i] < df['BollLower'].iloc[i]: in_trade = True
            elif in_trade and df[price_col].iloc[i] > df['BollUpper'].iloc[i]: in_trade = False
            df.at[i, 'signal_boll'] = int(in_trade)
        return df

class MeanReversionStrategy:
    name = "Mean Reversion"
    def __init__(self, lookback=20, z_entry=1.5, z_exit=0.5): self.lookback, self.z_entry, self.z_exit = lookback, z_entry, z_exit
    def generate_signals(self, df, price_col):
        df = df.copy()
        mean = df[price_col].rolling(self.lookback).mean()
        std = df[price_col].rolling(self.lookback).std()
        df['zscore'] = (df[price_col] - mean) / std
        df['signal_mr'] = 0
        in_trade = False
        for i in range(1, len(df)):
            if not in_trade and df['zscore'].iloc[i] < -self.z_entry: in_trade = True
            elif in_trade and abs(df['zscore'].iloc[i]) < self.z_exit: in_trade = False
            df.at[i, 'signal_mr'] = int(in_trade)
        return df

class BreakoutStrategy:
    name = "Breakout"
    def __init__(self, lookback=20): self.lookback = lookback
    def generate_signals(self, df, price_col):
        df = df.copy()
        high = df[price_col].rolling(self.lookback).max()
        low = df[price_col].rolling(self.lookback).min()
        df['signal_bo'] = 0
        in_trade = False
        for i in range(1, len(df)):
            if not in_trade and df[price_col].iloc[i] > high.iloc[i-1]: in_trade = True
            elif in_trade and df[price_col].iloc[i] < low.iloc[i-1]: in_trade = False
            df.at[i, 'signal_bo'] = int(in_trade)
        return df

pages = ["📈 Analyse graphique", "🔍 Backtest Stratégies", "⚡ Screeners", "💼 Portefeuille", "🛡️ Gestion du risque"]
page = st.sidebar.radio("Navigation", pages)
ticker = st.sidebar.text_input("Ticker", "AAPL")
start_date = st.sidebar.date_input("Début", value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("Fin", value=pd.to_datetime("2023-12-31"))
desc_text = st.sidebar.text_area("Commentaire rapide (non sauvegardé)", "")

if page.startswith("📈"):
    #st.header(f"Analyse Chartist & Technique : {ticker}")
    st.markdown(
    f"<h1 style='color:red;'>Analyse Chartist & Technique : {ticker}</h1>",
    unsafe_allow_html=True)
    df = fetch_data(ticker, start_date, end_date)
    close_col = detect_col(df, ticker, "Close")
    open_col  = detect_col(df, ticker, "Open")
    high_col  = detect_col(df, ticker, "High")
    low_col   = detect_col(df, ticker, "Low")
    menu_techs = st.multiselect("Techniques à afficher", ["SMA20", "SMA50", "EMA20", "Bollinger"], default=["SMA20","SMA50"])
    for tech in menu_techs:
        if tech=="SMA20": df["SMA20"] = df[close_col].rolling(20).mean()
        if tech=="SMA50": df["SMA50"] = df[close_col].rolling(50).mean()
        if tech=="EMA20": df["EMA20"] = df[close_col].ewm(span=20).mean()
        if tech=="Bollinger":
            df["BollUpper"] = df[close_col].rolling(20).mean() + 2 * df[close_col].rolling(20).std()
            df["BollLower"] = df[close_col].rolling(20).mean() - 2 * df[close_col].rolling(20).std()
    if not all([close_col, open_col, high_col, low_col]) or df[close_col].dropna().size < 2:
        st.warning("Pas de données complètes pour ce ticker/période.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=pd.to_datetime(df['Date']),
            open=df[open_col], high=df[high_col], low=df[low_col], close=df[close_col],
            name='Bougies',
            increasing_line_color=ROSE2,
            decreasing_line_color=ROSE_DARK
        ))
        tech_colors = {
            "SMA20": ROSE,
            "SMA50": ROSE_DARK,
            "EMA20": "#ae2677",
            "BollUpper": ROSE,
            "BollLower": "#ea9dd8"
        }
        for tech in menu_techs:
            if tech=="Bollinger":
                fig.add_trace(go.Scatter(x=df['Date'], y=df['BollUpper'], name="Boll Upper", line=dict(color=tech_colors["BollUpper"], dash='dot')))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['BollLower'], name="Boll Lower", line=dict(color=tech_colors["BollLower"], dash='dot')))
            elif tech in df:
                fig.add_trace(go.Scatter(x=df['Date'], y=df[tech], name=tech, line=dict(color=tech_colors.get(tech,ROSE))))
        fig.update_layout(margin=dict(l=0,r=0,t=32,b=0), plot_bgcolor=BG_HIGH)
        st.plotly_chart(fig, use_container_width=True)
        perf, dd = summarize_perf(df[close_col].dropna()/df[close_col].dropna().iloc[0])
        kpi_block("Performance", f"{perf} %", f"{start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
        kpi_block("Drawdown", f"{dd} %", "")
        st.download_button("Télécharger le CSV", df.to_csv(index=False), f"{ticker}_data_{start_date}_{end_date}.csv")
        st.write(df.tail(20))
    if desc_text.strip(): st.info("Commentaire : "+desc_text)

if page.startswith("🔍"):
    st.header("Backtest multi-stratégies couplées")
    with st.sidebar:
        st.markdown("**Sélectionne stratégies et paramètres**")
        colz1, colz2 = st.columns(2)
        with colz1:
            use_bnh = st.checkbox("Buy & Hold", True)
            use_momo = st.checkbox("Momentum")
            use_boll = st.checkbox("Bollinger")
        with colz2:
            use_mr = st.checkbox("Mean Rev.")
            use_bo = st.checkbox("Breakout")
        sma_fast = st.slider("SMA rapide Momentum", 2, 100, 20)
        sma_slow = st.slider("SMA lente Momentum", 2, 200, 50)
        threshold = st.slider("Seuil Momentum", -50, 50, 0)
        period_boll = st.slider("Période Bollinger", 5, 100, 20)
        std_boll = st.slider("Ecart-type Bollinger", 1.0, 5.0, 2.0, 0.1)
        lookback_mr = st.slider("Lookback Mean Rev.", 5, 100, 20)
        z_entry = st.slider("z_entry MRev.", 0.5, 5.0, 1.5, 0.1)
        z_exit = st.slider("z_exit MRev.", 0.05, 2.0, 0.5, 0.05)
        lookback_bo = st.slider("Lookback Breakout", 5, 100, 20)
        combine_mode = st.selectbox("Combiner signaux", ["Intersection (ET)", "Union (OU)"], 0)
    if st.button("Lancer backtest"):
        df = fetch_data(ticker, start_date, end_date)
        close_col = detect_col(df, ticker, "Close")
        if close_col is None or df.empty or df[close_col].dropna().size < 2:
            st.error("Aucune donnée exploitable.")
            st.stop()
        signal_cols = []
        if use_bnh: df = BuyAndHoldStrategy().generate_signals(df, close_col); signal_cols.append("signal_bnh")
        if use_momo: df = MomentumStrategy(sma_fast, sma_slow, threshold).generate_signals(df, close_col); signal_cols.append("signal_momo")
        if use_boll: df = BollingerStrategy(period_boll, std_boll).generate_signals(df, close_col); signal_cols.append("signal_boll")
        if use_mr: df = MeanReversionStrategy(lookback_mr, z_entry, z_exit).generate_signals(df, close_col); signal_cols.append("signal_mr")
        if use_bo: df = BreakoutStrategy(lookback_bo).generate_signals(df, close_col); signal_cols.append("signal_bo")
        if signal_cols:
            if combine_mode == "Intersection (ET)": df['signal_all'] = df[signal_cols].fillna(0).astype(int).prod(axis=1)
            else: df['signal_all'] = df[signal_cols].fillna(0).astype(int).max(axis=1)
            returns = df[close_col].pct_change().fillna(0)
            equity_curve = np.ones(len(df))
            for i in range(1, len(df)):
                if df['signal_all'].iloc[i]: equity_curve[i] = equity_curve[i-1] * (1 + returns.iloc[i])
                else: equity_curve[i] = equity_curve[i-1]
            perf, dd = summarize_perf(equity_curve)
            kpi_block("Performance combinée", f"{perf} %", f"DD : {dd} %")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=equity_curve, name="Equity", line=dict(color=ROSE)))
            st.plotly_chart(fig, use_container_width=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df['Date'], y=df[close_col], name="Prix", line=dict(color=ROSE2)))
            entry_dates = df['Date'][df['signal_all'].diff().fillna(0) == 1]
            exit_dates  = df['Date'][df['signal_all'].diff().fillna(0) == -1]
            entry_prices= df[close_col][df['signal_all'].diff().fillna(0) == 1]
            exit_prices = df[close_col][df['signal_all'].diff().fillna(0) == -1]
            if len(entry_dates) > 0:
                fig2.add_trace(go.Scatter(x=entry_dates, y=entry_prices, mode="markers", marker=dict(color="green", size=8, symbol="triangle-up"), name="Entrée"))
            if len(exit_dates) > 0:
                fig2.add_trace(go.Scatter(x=exit_dates, y=exit_prices, mode="markers", marker=dict(color="red", size=8, symbol="triangle-down"), name="Sortie"))
            st.plotly_chart(fig2, use_container_width=True)
        st.write(df[["Date", close_col]+signal_cols+["signal_all"]].tail(20))

if page.startswith("⚡"):
    st.header("Screeners - Multi-Sélection quant")
    tickers = st.sidebar.text_input("Tickers (séparés par ,)", "AAPL,MSFT,TSLA,GSPC,AMZN")
    seuil_perf = st.sidebar.slider("Perf mini (%)", -100, 100, 5)
    seuil_vol = st.sidebar.slider("Volatilité max (%)", 0, 100, 60)
    if st.sidebar.button("Lancer screener"):
        res = []
        for ticker_ in [t.strip() for t in tickers.split(",")]:
            df = fetch_data(ticker_, start_date, end_date)
            close_col = detect_col(df, ticker_, "Close")
            clôtures = df[close_col].dropna() if close_col else pd.Series()
            if clôtures.size < 2: continue
            try:
                perf = float(clôtures.iloc[-1]) / float(clôtures.iloc[0]) - 1
                perf = perf * 100
                vol = float(clôtures.pct_change().std() * (252 ** 0.5) * 100)
            except Exception:
                continue
            if perf >= seuil_perf and vol <= seuil_vol:
                res.append({"Ticker": ticker_, "Performance (%)": perf, "Volatilité (%)": vol})
        if len(res) == 0:
            st.warning("Aucun actif filtré.")
        else:
            st.dataframe(pd.DataFrame(res))

if page.startswith("💼"):
    st.header("Portefeuille pondéré & courbe cumulée")
    tickers = st.sidebar.text_input("Tickers (séparés par ,)", "AAPL,MSFT,TSLA,GSPC")
    poids_text = st.sidebar.text_input("Poids (séparés par ,)", "0.25,0.25,0.25,0.25")
    if st.sidebar.button("Calculer Portefeuille"):
        tickers_list = [t.strip() for t in tickers.split(',')]
        poids = [float(x) for x in poids_text.split(',')] if poids_text else []
        if len(tickers_list) != len(poids): st.error("Autant de tickers que de poids requis."); st.stop()
        curves = []
        for i, ticker_ in enumerate(tickers_list):
            df = fetch_data(ticker_, start_date, end_date)
            close_col = detect_col(df, ticker_, "Close")
            clôtures = df[close_col].dropna() if close_col else pd.Series()
            if clôtures.size < 2: continue
            curve = (clôtures / float(clôtures.iloc[0])) * poids[i]
            curves.append(curve)
            st.line_chart(curve.rename(f"{ticker_} pondéré"))
        if len(curves) > 0:
            base_index = curves[0].index
            port_curve = sum([c.reindex(base_index, fill_value=0) for c in curves])
            st.subheader("Portefeuille agrégé")
            st.line_chart(port_curve.rename("Portefeuille"))
            perf, dd = summarize_perf(port_curve)
            kpi_block("Performance Portefeuille", f"{perf} %", f"DD : {dd} %")

if page.startswith("🛡️"):
    st.header("Gestion du risque - Value-at-Risk")
    horizon = st.sidebar.number_input("Horizon VaR (jours)", 1, 30, 5)
    level = st.sidebar.slider("Quantile VaR (%)", 90, 99, 95)
    if st.sidebar.button("Analyser Risque"):
        df = fetch_data(ticker, start_date, end_date)
        close_col = detect_col(df, ticker, "Close")
        clôtures = df[close_col].dropna() if close_col else pd.Series()
        if clôtures.size < 2:
            st.error("Pas de prix valides trouvés.")
            st.stop()
        returns = clôtures.pct_change().dropna()
        var = np.percentile(returns * np.sqrt(horizon), 100-level)
        kpi_block("VaR", f"{var*100:.2f} %", f"{level}% / {horizon} jours")
        st.line_chart(returns)

st.markdown(f"<hr><span style='color:{ROSE_DARK};font-weight:bold'>Quant Dashboard rose : trading quant, analyse, stratégies et graphique pro.</span>", unsafe_allow_html=True)
