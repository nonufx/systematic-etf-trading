import pandas as pd

def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """
    Daily portfolio turnover:
    sum of absolute changes in weights from day to day.
    """
    daily_changes = weights.diff().abs()
    turnover = daily_changes.sum(axis=1)
    return turnover

def smooth_weights(weights: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
    """
    Apply portfolio inertia to reduce turnover.
    alpha = fraction of today's target weights we move toward.
    """
    smoothed = weights.copy()
    smoothed.iloc[0] = weights.iloc[0]

    for t in range(1, len(weights)):
        smoothed.iloc[t] = (
            (1 - alpha) * smoothed.iloc[t - 1] +
            alpha * weights.iloc[t]
        )

    return smoothed

def apply_transaction_costs(
    portfolio_returns: pd.Series,
    turnover: pd.Series,
    cost_rate: float = 0.001
) -> pd.Series:
    """
    Subtract transaction costs from portfolio returns.
    """
    costs = turnover * cost_rate
    net_returns = portfolio_returns - costs
    return net_returns
