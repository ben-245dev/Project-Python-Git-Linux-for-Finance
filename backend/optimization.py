import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

def optimize_portfolio_weights(prices: pd.DataFrame, objective="sharpe", min_weight=0.0, max_weight=1.0):
    """
    Optimal weights for a portfolio using Modern Portfolio Theory.
    objective: "sharpe" to maximize Sharpe Ratio, "min_vol" to minimize volatility.
    """
    # 1. Stats calculus
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)

    # 2. Efficient Frontier
    ef = EfficientFrontier(mu, S, weight_bounds=(min_weight, max_weight))
    
    try:
        if objective == "sharpe":
            ef.max_sharpe()
        elif objective == "min_vol":
            ef.min_volatility()
        
        weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False)
        return weights, performance
    except Exception as e:
        print(f"Erreur Optimisation: {e}")
        return None, None