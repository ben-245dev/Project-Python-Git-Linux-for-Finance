import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

def calculate_cointegration(series_a: pd.Series, series_b: pd.Series):
    """
    Teste la cointégration entre deux actifs (Pairs Trading).
    Retourne le p-value, le score et le ratio (Hedge Ratio).
    Si p-value < 0.05, les paires sont cointégrées (bon pour le trading).
    """
    # Alignement des données
    df = pd.concat([series_a, series_b], axis=1).dropna()
    S1 = df.iloc[:, 0]
    S2 = df.iloc[:, 1]
    
    # Calcul du Hedge Ratio via OLS (Régression linéaire)
    S1 = sm.add_constant(S1)
    results = sm.OLS(S2, S1).fit()
    hedge_ratio = results.params.iloc[1]
    
    # Test de cointégration
    score, pvalue, _ = coint(df.iloc[:, 0], df.iloc[:, 1])
    
    return pvalue, hedge_ratio

def calculate_zscore(series: pd.Series, window=30):
    """
    Calcul du Z-Score glissant.
    Indique à combien d'écarts-types on se situe de la moyenne.
    """
    r_mean = series.rolling(window=window).mean()
    r_std = series.rolling(window=window).std()
    z_score = (series - r_mean) / r_std
    return z_score

def calculate_kelly_criterion(win_rate, win_loss_ratio):
    """
    Formule de Kelly : f = p - (q / b)
    p = proba de gain, q = proba de perte, b = ratio gain/perte
    Retourne le % du capital à investir.
    """
    if win_loss_ratio == 0: return 0
    kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
    return kelly