import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def compute_drawdown(equity: pd.Series):
    """Calcule le drawdown historique."""
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return dd, dd.min()

def compute_sharpe(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0):
    """Ratio de Sharpe (Rendement / Volatilité totale)."""
    if returns.std() == 0 or returns.empty:
        return np.nan
    excess = returns - rf / periods_per_year
    return np.sqrt(periods_per_year) * excess.mean() / excess.std()

def compute_sortino(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0):
    """Ratio de Sortino (Rendement / Volatilité Négative)."""
    if returns.empty: return np.nan
    
    excess = returns - rf / periods_per_year
    # On ne garde que les rendements négatifs pour le risque
    downside_returns = excess[excess < 0]
    
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)
    
    if downside_std == 0: return np.nan
    return (excess.mean() * periods_per_year) / downside_std

def compute_calmar(total_return_ann: float, max_drawdown: float):
    """Ratio de Calmar (Rendement Annuel / Max Drawdown)."""
    if max_drawdown == 0: return np.nan
    return total_return_ann / abs(max_drawdown)

def compute_strategy_metrics(strategy_returns: pd.Series, equity: pd.Series, rf: float = 0.0):
    """Calcule un ensemble complet de métriques."""
    if strategy_returns.empty: return {}

    # Rendements et Volatilité
    days = len(strategy_returns)
    years = days / 252
    total_ret = equity.iloc[-1] - 1.0
    cagr = (equity.iloc[-1])**(1/years) - 1 if years > 0 else 0
    
    vol = strategy_returns.std() * np.sqrt(252)
    
    # Risque
    _, max_dd = compute_drawdown(equity)
    
    # Ratios
    sharpe = compute_sharpe(strategy_returns, rf=rf)
    sortino = compute_sortino(strategy_returns, rf=rf)
    calmar = compute_calmar(cagr, max_dd)
    
    # Distribution
    dist_skew = skew(strategy_returns)
    dist_kurt = kurtosis(strategy_returns)

    return {
        "total_return": total_ret,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "vol": vol,
        "skew": dist_skew,
        "kurtosis": dist_kurt
    }

def optimize_portfolio(prices: pd.DataFrame, objective="sharpe"):
    """
    Calcule les poids optimaux pour un portefeuille donné.
    objective: 'sharpe' (max Sharpe) ou 'min_vol' (min volatilité)
    """
    # 1. Calculer les rendements attendus et la matrice de covariance
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)

    # 2. Optimisation
    ef = EfficientFrontier(mu, S)
    
    if objective == "sharpe":
        ef.max_sharpe()
    elif objective == "min_vol":
        ef.min_volatility()
    
    # 3. Nettoyage des poids (arrondis)
    cleaned_weights = ef.clean_weights()
    return cleaned_weights, ef.portfolio_performance(verbose=False)