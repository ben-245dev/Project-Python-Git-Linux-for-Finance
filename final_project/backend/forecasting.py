import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ... (Gardez les fonctions ARIMA existantes : _make_series_stationary, etc.) ...
# ... (Assurez-vous que arima_forecast est toujours là) ...

def _make_series_stationary(close: pd.Series):
    series = close.asfreq("D").ffill()
    log_returns = np.log(series).diff().dropna()
    return log_returns, series

def _select_arima_order(y: pd.Series, max_p: int = 3, max_q: int = 3, d: int = 0):
    best_aic = np.inf
    best_order = (1, d, 1)
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0: continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(y, order=(p, d, q))
                    res = model.fit()
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = (p, d, q)
            except: continue
    return best_order, best_aic

def arima_forecast(close: pd.Series, horizon: int = 30):
    """Prévision classique ARIMA"""
    if len(close) < 50: return None
    
    y, price_series = _make_series_stationary(close)
    best_order, _ = _select_arima_order(y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(y, order=best_order)
        fit = model.fit()

    res = fit.get_forecast(steps=horizon)
    y_pred = res.predicted_mean
    ci = res.conf_int()

    last_price = float(price_series.iloc[-1])
    cum_log_ret = y_pred.cumsum()
    
    price_forecast = last_price * np.exp(cum_log_ret)
    price_low = last_price * np.exp(ci.iloc[:, 0].cumsum())
    price_high = last_price * np.exp(ci.iloc[:, 1].cumsum())

    return pd.DataFrame({
        "y_pred": price_forecast,
        "ci_low": price_low,
        "ci_high": price_high
    }, index=y_pred.index)

def ml_predict_direction(df_input: pd.DataFrame):
    """
    Utilise un Random Forest pour prédire si le prix va monter demain.
    Retourne : Probabilité de hausse, Précision du modèle (Test), Importance des features
    """
    df = df_input.copy()
    
    # 1. Création de Features (Lagged returns, Volatilité, RSI)
    df['Returns'] = df['Close'].pct_change()
    df['Vol_5'] = df['Returns'].rolling(5).std()
    df['RSI'] = df.ta.rsi(length=14)
    df['SMA_Diff'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close']
    
    # Target : 1 si le prix de DEMAIN monte, 0 sinon
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df = df.dropna()
    
    if len(df) < 100: return None, None, None # Pas assez de données

    features = ['Returns', 'Vol_5', 'RSI', 'SMA_Diff']
    X = df[features]
    y = df['Target']
    
    # Split Train/Test (sans mélanger l'ordre temporel)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Évaluation
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    
    # Prédiction pour demain (basée sur la dernière ligne connue)
    last_row = X.iloc[[-1]]
    prob_up = model.predict_proba(last_row)[0][1] # Proba de la classe 1 (Hausse)
    
    return prob_up, accuracy, dict(zip(features, model.feature_importances_))