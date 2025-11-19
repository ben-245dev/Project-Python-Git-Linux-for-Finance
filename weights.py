import pandas as pd

def equal_weights(df):
    """
    Returns an equal-weight portfolio vector.
    
    Parameters
    ----------
    df : DataFrame
        Price dataframe with assets as columns.

    Returns
    -------
    Series
        Weights summing to 1.
    """
    n_assets = df.shape[1]
    weights = pd.Series([1 / n_assets] * n_assets, index=df.columns)
    return weights


def custom_weights(weight_dict, df):
    """
    Create a custom weight vector from a dictionary.
    
    Example:
        weight_dict = {"AAPL": 0.5, "MSFT": 0.3, "NVDA": 0.2}

    Parameters
    ----------
    weight_dict : dict
        Keys = asset names, Values = weights.
    df : DataFrame
        Price dataframe with assets as columns.

    Returns
    -------
    Series
        Validated weight vector.
    """
    # Create Series and align with df columns
    weights = pd.Series(weight_dict)

    # Check for missing assets
    missing_assets = set(df.columns) - set(weights.index)
    if missing_assets:
        raise ValueError(f"Missing weights for assets: {missing_assets}")

    # Reorder weights to match dataframe columns
    weights = weights[df.columns]

    # Validate
    return validate_weights(weights)


def validate_weights(weights):
    """
    Validate that:
    - weights sum to 1
    - no negative weights (unless purposely allowed later)

    Parameters
    ----------
    weights : Series

    Returns
    -------
    Series
        Cleaned and validated weights.
    """

    if (weights < 0).any():
        raise ValueError("Negative weights not allowed in this version.")

    total = weights.sum()

    if abs(total - 1) > 1e-6:
        raise ValueError(f"Weights must sum to 1. Current sum = {total}")

    return weights
