import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime
import pytz
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
# -----------------------------
# Config générale
# -----------------------------
st.set_page_config(
    page_title="Trading Dashboard",
    layout="wide",
)

# Indices principaux
INDICES = {
    "S&P 500": "^GSPC",
    "DAX": "^GDAXI",
    "CAC 40": "^FCHI",
    "Nikkei 225": "^N225",
    "FTSE 100": "^FTSE",
    "Dow Jones": "^DJI",
}

TIMEZONES = {
    "New York": "America/New_York",
    "Londres": "Europe/London",
    "Paris": "Europe/Paris",
    "Tokyo": "Asia/Tokyo",
}

# -----------------------------
# Fonctions utilitaires communes
# -----------------------------
@st.cache_data(ttl=60)
def get_live_index_data():
    data = {}
    for name, ticker in INDICES.items():
        try:
            df = yf.download(ticker, period="1d", interval="1m", progress=False)
            if not df.empty:
                last_row = df.iloc[-1]
                price = float(last_row["Close"])
                ts_utc = last_row.name.tz_convert("UTC") if last_row.name.tzinfo else last_row.name.tz_localize("UTC")
                data[name] = {
                    "price": round(price, 2),
                    "time_utc": ts_utc,
                }
            else:
                data[name] = None
        except Exception:
            data[name] = None
    return data
import streamlit as st
from datetime import datetime
import pytz
import time

TRADING_TIMEZONES = {
    "New York": "America/New_York",
    "London": "Europe/London",
    "Paris": "Europe/Paris",
    "Tokyo": "Asia/Tokyo",
    "Sydney": "Australia/Sydney",
}

def render_trading_clocks():
    st.subheader("Horloges des principaux fuseaux de trading")
    cols = st.columns(len(TRADING_TIMEZONES))
    for col, (name, tz_str) in zip(cols, TRADING_TIMEZONES.items()):
        with col:
            tz = pytz.timezone(tz_str)
            now_local = datetime.now(tz)
            st.markdown(
                f"""
                <div style="background-color:#0e1117;padding:10px;border-radius:8px;">
                    <h4 style="color:#00c3ff;text-align:center;">{name}</h4>
                    <p style="color:#FFFFFF;font-size:22px;text-align:center;margin:0;">
                        {now_local.strftime('%H:%M:%S')}
                    </p>
                    <p style="color:#AAAAAA;font-size:12px;text-align:center;margin:0;">
                        {now_local.strftime('%Y-%m-%d')}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

def render_time_by_timezone(ts_utc):
    if ts_utc is None:
        return "-"
    parts = []
    for label, tz_name in TIMEZONES.items():
        tz = pytz.timezone(tz_name)
        local_time = ts_utc.astimezone(tz).strftime("%Y-%m-%d %H:%M")
        parts.append(f"{label}: {local_time}")
    return " | ".join(parts)

def render_banner(live_data):
    items = []
    for name, v in live_data.items():
        if v is None:
            continue
        price = v["price"]
        time_utc = v["time_utc"].strftime("%H:%M UTC")
        items.append(f"{name}: {price} ({time_utc})")
    text = " | ".join(items) if items else "Données indisponibles"
    st.markdown(
        f"""
        <div style="background-color:#0e1117;padding:5px 0;">
            <marquee style="color:#00c3ff; font-size:18px;">
                {text}
            </marquee>
        </div>
        """,
        unsafe_allow_html=True,
    )

@st.cache_data(ttl=300)
def load_ticker_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True,
                     progress=False, multi_level_index=False)
    return df.dropna()


def get_price_series(df: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    """
    Gère les deux cas :
    - colonnes MultiIndex (Price, Ticker) -> ('Close', 'AAPL')
    - colonnes simples -> 'Close'
    """
    if isinstance(df.columns, pd.MultiIndex):
        # MultiIndex yfinance récent
        if (field, ticker) in df.columns:
            return df[(field, ticker)]
        # fallback si le ticker n'est pas présent dans le niveau 1
        if field in df.columns.get_level_values(0):
            return df[field]
        raise KeyError(f"Colonne {(field, ticker)} ou {field} introuvable dans df.columns")
    else:
        # Colonnes simples: 'Open', 'High', 'Low', 'Close', 'Volume'
        if field not in df.columns:
            raise KeyError(f"Colonne {field} introuvable dans df.columns")
        return df[field]


def compute_drawdown(equity: pd.Series):
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return dd, dd.min()

def compute_sharpe(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0):
    if returns.std() == 0 or returns.empty:
        return np.nan
    excess = returns - rf / periods_per_year
    return np.sqrt(periods_per_year) * excess.mean() / excess.std()

def price_chart(df: pd.DataFrame, ticker: str, chart_type: str):
    if df.empty:
        st.warning("Pas de données disponibles pour ce ticker / cette période.")
        return

    if chart_type == "Bougies":
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df.index,
                    open=get_price_series(df, "Open", ticker),
                    high=get_price_series(df, "High", ticker),
                    low=get_price_series(df, "Low", ticker),
                    close=get_price_series(df, "Close", ticker),
                    name=ticker,
                )
            ]
        )
    else:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=df.index,
                    y=get_price_series(df, "Close", ticker),
                    mode="lines",
                    name=ticker,
                )
            ]
        )

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Date",
        yaxis_title="Prix",
    )
    st.plotly_chart(fig, use_container_width=True)



def _make_series_stationary(close: pd.Series):
    """
    Retourne une série stationnaire (log ou rendements) + flag pour savoir comment revenir au niveau.
    Ici, on travaille sur les rendements log pour ARIMA (classique en finance).
    """
    # On force une fréquence régulière et on remplit les trous
    series = close.asfreq("D").ffill()

    # Rendements log (série souvent déjà stationnaire)
    log_returns = np.log(series).diff().dropna()

    # Petit check ADF pour info (optionnel pour le modèle, utile pour debug)
    try:
        adf_pvalue = adfuller(log_returns)[1]
        # Tu peux logger/adresser cette info si tu veux
    except Exception:
        adf_pvalue = None

    return log_returns, series

def _select_arima_order(y: pd.Series, max_p: int = 3, max_q: int = 3, d: int = 0):
    """
    Grid-search simple sur p,q pour minimiser l'AIC.
    On reste volontairement léger pour ne pas exploser le temps de calcul dans Streamlit.
    """
    best_aic = np.inf
    best_order = None

    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(y, order=(p, d, q))
                    res = model.fit()
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = (p, d, q)
            except Exception:
                continue

    # fallback si rien ne marche
    if best_order is None:
        best_order = (1, d, 1)

    return best_order, best_aic

def arima_forecast(close: pd.Series, horizon: int = 30,
                   max_p: int = 3, max_q: int = 3, d: int = 0):
    """
    Modèle ARIMA(p,d,q) sur une série préparée (log-returns).
    Sélection d'ordre par AIC sur une petite grille.
    Retourne DataFrame: forecast sur le niveau de prix + intervalles de confiance.
    """
    if len(close) < 50:
        raise ValueError("Pas assez de points pour un ARIMA robuste (min ~50).")

    # 1) préparation: log-returns stationnaires + série de prix d'origine
    y, price_series = _make_series_stationary(close)

    # 2) sélection (p,q) par AIC sur la série stationnaire
    best_order, best_aic = _select_arima_order(y, max_p=max_p, max_q=max_q, d=d)

    # 3) fit final ARIMA sur la série stationnaire
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(y, order=best_order)
        fit = model.fit()

    # 4) forecast sur les rendements log, puis reconstruction en niveau
    res = fit.get_forecast(steps=horizon)
    y_pred = res.predicted_mean          # prévision des log-returns
    ci = res.conf_int()                  # IC sur les log-returns

    # On reconstruit un chemin de prix à partir du dernier prix observé
    last_price = float(price_series.iloc[-1])
    # cumul de log-returns forecast -> facteur multiplicatif
    cum_log_ret = y_pred.cumsum()
    price_forecast = last_price * np.exp(cum_log_ret)

    # Pour les IC, on applique la même logique sur les bornes
    cum_low = ci.iloc[:, 0].cumsum()
    cum_high = ci.iloc[:, 1].cumsum()
    price_low = last_price * np.exp(cum_low)
    price_high = last_price * np.exp(cum_high)

    forecast_df = pd.DataFrame(
        {
            "y_pred": price_forecast,
            "ci_low": price_low,
            "ci_high": price_high,
            "best_p": best_order[0],
            "best_d": best_order[1],
            "best_q": best_order[2],
            "aic": best_aic,
        },
        index=y_pred.index,  # mêmes dates que le forecast ARIMA
    )
    return forecast_df


# -----------------------------
# Page Accueil (inchangée)
# -----------------------------
def page_home():
    st.markdown(
        """
        <h1 style="color:#FFFFFF;">Dashboard de trading</h1>
        <p style="color:#AAAAAA;">Surveillance des marchés, visualisation des prix et analyse des risques.</p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.subheader("Indices majeurs en direct")
    live_data = get_live_index_data()
    render_trading_clocks()
    render_banner(live_data)

    # cols = st.columns(len(INDICES))
    # for col, (name, _) in zip(cols, INDICES.items()):
    #     with col:
    #         data = live_data.get(name)
    #         if data is None:
    #             st.metric(name, "N/A", "N/A")
    #         else:
    #             time_str = render_time_by_timezone(data["time_utc"])
    #             st.metric(name, f"{data['price']}", time_str)

    st.markdown("---")

    st.sidebar.header("Paramètres backtest")
    ticker = st.sidebar.text_input("Ticker (ex: AAPL, MSFT, ^GSPC)", value="AAPL")
    period = st.sidebar.selectbox(
        "Période de backtest",
        ["6mo", "1y", "3y", "5y", "max"],
        index=1,
    )

    # Bouton qui déclenche vraiment le recalcul
    run_backtest = st.sidebar.button("Mettre à jour le backtest")

    # On mémorise les derniers paramètres validés
    if "last_ticker" not in st.session_state:
        st.session_state.last_ticker = ticker
    if "last_period" not in st.session_state:
        st.session_state.last_period = period

    if run_backtest:
        st.session_state.last_ticker = ticker
        st.session_state.last_period = period

    # On utilise toujours les valeurs "validées"
    ticker_used = st.session_state.last_ticker
    period_used = st.session_state.last_period

    st.write(f"Backtest en cours sur: {ticker_used} / {period_used}")

    df = load_ticker_data(ticker_used, period_used)

    chart_type = st.sidebar.selectbox("Type de graphique", ["Bougies", "Courbes"])
    quantile = st.sidebar.slider(
        "Quantile pour la VaR (historique)",
        min_value=0.90,
        max_value=0.99,
        value=0.95,
        step=0.01,
    )

    if ticker:
        df = load_ticker_data(ticker, period)

        if df.empty:
            st.warning("Pas de données téléchargées. Vérifie le ticker ou la période.")
            return

        st.subheader(f"Cours de {ticker}")
        price_chart(df, ticker, chart_type)

        close = get_price_series(df, "Close", ticker)
        returns = close.pct_change().dropna()
        if returns.empty:
            st.warning("Pas assez de données pour les stats.")
            return

        cum_ret_series = (1 + returns).cumprod() - 1
        cum_ret = float(cum_ret_series.iloc[-1])
        dd_series, max_dd = compute_drawdown(close)
        mean_ret = float(returns.mean())
        var_value = float(returns.quantile(1 - quantile))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rendement cumulé", f"{cum_ret*100:.2f} %")
        with col2:
            st.metric("Drawdown max", f"{max_dd*100:.2f} %")
        with col3:
            st.metric("Rendement moyen", f"{mean_ret*100:.2f} %")

        st.markdown("---")
        st.subheader("Distribution des rendements")

        fig = go.Figure(
            data=[
                go.Histogram(
                    x=returns.values,
                    nbinsx=50,
                    marker_color="#00c3ff",
                )
            ]
        )
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Rendement",
            yaxis_title="Fréquence",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**VaR historique {quantile*100:.1f}%** : {var_value*100:.2f} %")

# -----------------------------
# Stratégies de backtest
# -----------------------------
def build_strategies(close: pd.Series, params: dict):
    """
    Retourne un dict {strategy_name: dict(dataframe, equity, returns)}
    Stratégies:
      - Buy & Hold
      - Momentum (lookback)
      - Mean Reversion (lookback)
      - Moving Average Crossover (fast/slow)
      - Breakout (n-high)
    """
    prices = close
    rets = prices.pct_change().fillna(0.0)

    strategies = {}

    # 1) Buy & Hold
    bh_returns = rets.copy()
    bh_equity = (1 + bh_returns).cumprod()
    strategies["Buy & Hold"] = {
        "returns": bh_returns,
        "equity": bh_equity,
    }

    # 2) Momentum: long si prix > prix lookback
    lb = params.get("mom_lookback", 20)
    mom_signal = (prices > prices.shift(lb)).astype(int)  # 1 long, 0 cash
    mom_returns = mom_signal.shift(1).fillna(0) * rets
    mom_equity = (1 + mom_returns).cumprod()
    strategies["Momentum"] = {
        "returns": mom_returns,
        "equity": mom_equity,
    }

    # 3) Mean Reversion: long si prix < prix lookback
    mr_signal = (prices < prices.shift(lb)).astype(int)
    mr_returns = mr_signal.shift(1).fillna(0) * rets
    mr_equity = (1 + mr_returns).cumprod()
    strategies["Mean Reversion"] = {
        "returns": mr_returns,
        "equity": mr_equity,
    }

    # 4) Moving Average Crossover
    fast = params.get("fast_ma", 20)
    slow = params.get("slow_ma", 50)
    ma_fast = prices.rolling(fast).mean()
    ma_slow = prices.rolling(slow).mean()
    mac_signal = (ma_fast > ma_slow).astype(int)
    mac_returns = mac_signal.shift(1).fillna(0) * rets
    mac_equity = (1 + mac_returns).cumprod()
    strategies["MA Crossover"] = {
        "returns": mac_returns,
        "equity": mac_equity,
    }

    # 5) Breakout n plus-hauts
    n_break = params.get("breakout_lookback", 50)
    rolling_max = prices.shift(1).rolling(n_break).max()
    bo_signal = (prices > rolling_max).astype(int)
    bo_returns = bo_signal.shift(1).fillna(0) * rets
    bo_equity = (1 + bo_returns).cumprod()
    strategies["Breakout"] = {
        "returns": bo_returns,
        "equity": bo_equity,
    }

    return strategies

def compute_strategy_metrics(strategy_returns: pd.Series, equity: pd.Series):
    total_ret = equity.iloc[-1] - 1.0
    dd_series, max_dd = compute_drawdown(equity)
    sharpe = compute_sharpe(strategy_returns)
    vol = strategy_returns.std() * np.sqrt(252)
    return {
        "total_return": total_ret,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "vol": vol,
    }

# -----------------------------
# Modèle de prévision simple
# -----------------------------
def linear_regression_forecast(close: pd.Series, horizon: int = 30):
    """
    Régression linéaire sur le temps (indice) pour extrapoler les prix.
    Retourne DataFrame avec historique + forecast + intervalle de confiance simple.
    """
    df = pd.DataFrame({"y": close.values})
    df["t"] = np.arange(len(df))

    X = df[["t"]].values
    y = df["y"].values

    model = LinearRegression()
    model.fit(X, y)

    t_future = np.arange(len(df), len(df) + horizon)
    X_future = t_future.reshape(-1, 1)
    y_pred = model.predict(X_future)

    # Intervalle de confiance simple basé sur l'écart-type des résidus
    residuals = y - model.predict(X)
    sigma = residuals.std()
    ci_upper = y_pred + 1.96 * sigma
    ci_lower = y_pred - 1.96 * sigma

    hist_index = close.index
    freq = pd.infer_freq(hist_index) or "D"
    future_index = pd.date_range(
        start=hist_index[-1] + pd.tseries.frequencies.to_offset(freq),
        periods=horizon,
        freq=freq,
    )

    forecast_df = pd.DataFrame(
        {
            "y_pred": y_pred,
            "ci_low": ci_lower,
            "ci_high": ci_upper,
        },
        index=future_index,
    )

    return forecast_df

# -----------------------------
# Page Stratégie & Backtest
# -----------------------------
def page_strategy():
    st.markdown(
        """
        <h1 style="color:#FFFFFF;">Stratégie de trading et backtest</h1>
        <p style="color:#AAAAAA;">
        Définis une stratégie, ajuste ses paramètres, observe les métriques de performance
        et une prévision simple des prix.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Paramètres backtest")
    ticker = st.sidebar.text_input("Ticker (ex: AAPL, MSFT, ^GSPC)", value="AAPL")
    period = st.sidebar.selectbox(
        "Période de backtest",
        ["6mo", "1y", "3y", "5y", "max"],
        index=1,
    )
    freq = st.sidebar.selectbox("Périodicité", ["Daily", "Weekly", "Monthly"], index=0)

    # Paramètres stratégie
    st.sidebar.subheader("Paramètres des stratégies")
    mom_lb = st.sidebar.slider("Momentum lookback (jours)", 5, 120, 20, step=5)
    fast_ma = st.sidebar.slider("MA rapide (jours)", 5, 60, 20, step=5)
    slow_ma = st.sidebar.slider("MA lente (jours)", 20, 200, 50, step=10)
    breakout_lb = st.sidebar.slider("Breakout lookback (jours)", 20, 200, 50, step=10)

    strategy_name = st.sidebar.selectbox(
        "Stratégie",
        ["Buy & Hold", "Momentum", "Mean Reversion", "MA Crossover", "Breakout"],
    )

    horizon_forecast = st.sidebar.slider("Horizon de prévision (jours)", 5, 90, 30, step=5)

    if not ticker:
        st.warning("Renseigne un ticker.")
        return

    df = load_ticker_data(ticker, period)
    if df.empty:
        st.warning("Pas de données pour ce ticker / période.")
        return

    close = get_price_series(df, "Close", ticker)

    # Changement de périodicité
    if freq == "Weekly":
        close = close.resample("W").last()
    elif freq == "Monthly":
        close = close.resample("M").last()

    close = close.dropna()
    if close.empty:
        st.warning("Pas de données après agrégation.")
        return

    params = {
        "mom_lookback": mom_lb,
        "fast_ma": fast_ma,
        "slow_ma": slow_ma,
        "breakout_lookback": breakout_lb,
    }

    strategies = build_strategies(close, params)
    strat = strategies[strategy_name]
    strat_returns = strat["returns"]
    strat_equity = strat["equity"]

    # Benchmark: Buy&Hold equity
    equity_bh = strategies["Buy & Hold"]["equity"]

    # Graphique principal: prix + equity stratégie
    st.subheader(f"Backtest - {strategy_name} sur {ticker}")
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=close.index,
            y=close.values,
            mode="lines",
            name="Prix",
            yaxis="y1",
            line=dict(color="#00c3ff"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=strat_equity.index,
            y=strat_equity.values,
            mode="lines",
            name="Equity stratégie",
            yaxis="y2",
            line=dict(color="#ff9900"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=equity_bh.index,
            y=equity_bh.values,
            mode="lines",
            name="Equity Buy&Hold",
            yaxis="y2",
            line=dict(color="#888888", dash="dash"),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(domain=[0.0, 1.0]),
        yaxis=dict(title="Prix", side="left"),
        yaxis2=dict(
            title="Equity (normalisée)",
            overlaying="y",
            side="right",
        ),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Métriques
    st.subheader("Métriques de performance")
    metrics = compute_strategy_metrics(strat_returns, strat_equity)
    bh_metrics = compute_strategy_metrics(strategies["Buy & Hold"]["returns"], equity_bh)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total return (strat)", f"{metrics['total_return']*100:.2f} %")
    with col2:
        st.metric("Max drawdown (strat)", f"{metrics['max_drawdown']*100:.2f} %")
    with col3:
        st.metric("Sharpe (strat)", f"{metrics['sharpe']:.2f}")
    with col4:
        st.metric("Vol annualisée (strat)", f"{metrics['vol']*100:.2f} %")

    col5, col6, col7 = st.columns(3)
    with col5:
        st.metric("Total return (BH)", f"{bh_metrics['total_return']*100:.2f} %")
    with col6:
        st.metric("Max drawdown (BH)", f"{bh_metrics['max_drawdown']*100:.2f} %")
    with col7:
        st.metric("Sharpe (BH)", f"{bh_metrics['sharpe']:.2f}")

    # Distribution des rendements stratégie
    st.subheader("Distribution des rendements de la stratégie")
    fig_hist = go.Figure(
        data=[
            go.Histogram(
                x=strat_returns.values,
                nbinsx=50,
                marker_color="#ff9900",
            )
        ]
    )
    fig_hist.update_layout(
        template="plotly_dark",
        xaxis_title="Rendement",
        yaxis_title="Fréquence",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Prévision simple
    st.subheader("Prévision (ARIMA)")
    #forecast_df = arima_forecast(close, horizon=horizon_forecast, order=(1,1,1))
    forecast_df = arima_forecast(close, horizon=horizon_forecast)
    st.caption(f"ARIMA(p,d,q) sélectionné: ({int(forecast_df['best_p'].iloc[0])}, "
           f"{int(forecast_df['best_d'].iloc[0])}, {int(forecast_df['best_q'].iloc[0])}) "
           f" | AIC = {forecast_df['aic'].iloc[0]:.2f}")
    #forecast_df = linear_regression_forecast(close, horizon=horizon_forecast)

    fig_forecast = go.Figure()
    fig_forecast.add_trace(
        go.Scatter(
            x=close.index,
            y=close.values,
            mode="lines",
            name="Historique",
            line=dict(color="#00c3ff"),
        )
    )
    fig_forecast.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df["y_pred"],
            mode="lines",
            name="Prévision",
            line=dict(color="#ff0000"),
        )
    )
    fig_forecast.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df["ci_high"],
            mode="lines",
            name="IC haut",
            line=dict(color="rgba(255,0,0,0.3)", dash="dot"),
            showlegend=False,
        )
    )
    fig_forecast.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df["ci_low"],
            mode="lines",
            name="IC bas",
            line=dict(color="rgba(255,0,0,0.3)", dash="dot"),
            fill="tonexty",
            fillcolor="rgba(255,0,0,0.1)",
            showlegend=False,
        )
    )

    fig_forecast.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Date",
        yaxis_title="Prix",
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

# -----------------------------
# Navigation principale
# -----------------------------
def main():
    st.sidebar.title("Navigation")
    if st.sidebar.button("🔄 Recharger l'application"):
        st.rerun()  # relance tout le script
    page = st.sidebar.radio(
        "Aller à",
        ["Accueil", "Stratégie de trading et backtest"],
    )

    if page == "Accueil":
        page_home()
    else:
        page_strategy()

if __name__ == "__main__":
    main()
