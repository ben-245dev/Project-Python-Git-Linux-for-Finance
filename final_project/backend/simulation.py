import numpy as np
import pandas as pd

def run_monte_carlo(prices: pd.Series, days=252, simulations=1000):
    """
    Simule N trajectoires futures (Mouvement Brownien Géométrique).
    Retourne un DataFrame avec toutes les simulations.
    """
    last_price = prices.iloc[-1]
    returns = prices.pct_change().dropna()
    
    mu = returns.mean()
    sigma = returns.std()
    
    # Génération vectorisée pour la performance (beaucoup plus rapide que des boucles)
    # On simule 'simulations' colonnes sur 'days' lignes
    daily_shocks = np.random.normal(mu, sigma, (days, simulations))
    
    # On calcule les prix cumulés
    price_paths = np.zeros((days, simulations))
    price_paths[0] = last_price
    
    for t in range(1, days):
        price_paths[t] = price_paths[t-1] * np.exp(daily_shocks[t])
        
    sim_df = pd.DataFrame(price_paths)
    return sim_df

def run_historical_crash_test(current_value, crash_name="2008"):
    """
    Simule l'impact d'un krach historique sur le portefeuille actuel.
    Retourne la courbe d'évolution fictive.
    """
    # Données simplifiées des krachs majeurs (S&P 500 approximatif)
    scenarios = {
        "Subprimes (2008)": [-0.01, -0.02, -0.05, 0.01, -0.04, -0.07, -0.02, -0.09, -0.03, 0.02, -0.05, -0.06], # Séquence violente
        "Covid-19 (2020)": [-0.03, -0.04, 0.01, -0.12, -0.09, 0.05, -0.05, -0.02, 0.06, 0.09], # Chute brutale puis remontée
        "Dotcom (2000)": [-0.01]*20 + [-0.03]*10 + [0.01]*5 + [-0.02]*10 # Chute lente et douloureuse
    }
    
    scenario_rets = scenarios.get(crash_name, [-0.01]*10)
    
    # On étend la séquence pour la rendre plus longue si besoin (boucle)
    # Pour un test simple, on applique juste la séquence de chocs
    prices = [current_value]
    for ret in scenario_rets:
        prices.append(prices[-1] * (1 + ret))
        
    return pd.Series(prices)

def compute_risk_metrics(final_values: pd.Series, initial_value: float):
    """
    Calcule VaR, CVaR et Probabilité de Ruine.
    """
    returns = (final_values - initial_value) / initial_value
    
    # VaR 95% (Le seuil des 5% pires scénarios)
    var_95 = np.percentile(returns, 5)
    
    # CVaR 95% (Moyenne des pertes SI on dépasse la VaR - "Expected Shortfall")
    cvar_95 = returns[returns <= var_95].mean()
    
    # Probabilité de perdre plus de 20%
    prob_loss_20 = (returns < -0.20).mean()
    
    # Probabilité de gain
    prob_profit = (returns > 0).mean()
    
    return {
        "VaR_95": var_95,
        "CVaR_95": cvar_95,
        "Prob_Crash": prob_loss_20,
        "Prob_Profit": prob_profit
    }