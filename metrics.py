import pandas as pd      # to handle tables (dataframes)
import numpy as np       # for math operations

def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns for each ETF.
    """
    log_returns = np.log(price_df / price_df.shift(1))

    # Remove the first row because it will contain NaN
    log_returns = log_returns.dropna()

    return log_returns

def compute_ema_momentum(log_returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    EMA-based momentum
    Uses an exponential moving average (EMA) of log returns over 'span' days.
    More recent days get more weight than older days.
    """
    ema_mom = log_returns.ewm(span=window, adjust=False).mean()
    return ema_mom


def compute_mean_reversion_zscore(log_returns, window=20):
    # Rolling average of returns - what is “normal”
    rolling_mean = log_returns.rolling(window).mean()

    # Rolling standard deviation - how volatile is “normal”
    rolling_std = log_returns.rolling(window).std()

    # How unusual is today compared to normal?
    z_score = (log_returns - rolling_mean) / rolling_std

    return z_score

def compute_rolling_volatility(log_returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Rolling volatility = rolling standard deviation of log returns.
    Higher volatility = riskier ETF.
    """
    rolling_vol = log_returns.rolling(window).std()
    return rolling_vol

def compute_volatility_scaled_momentum(ema_momentum: pd.DataFrame, volatility: pd.DataFrame) -> pd.DataFrame:
    """
    Divide EMA momentum by rolling volatility.
    This makes the signal comparable across ETFs and stabilizes the model.
    """
    volatility_safe = volatility.replace(0, np.nan)
    scaled = ema_momentum / volatility_safe
    return scaled

def compute_rolling_sharpe(log_returns, window=20):
    # Rolling mean of returns
    rolling_mean = log_returns.rolling(window).mean()

    # Rolling volatility (standard deviation)
    rolling_vol = log_returns.rolling(window).std()

    # Avoid dividing by zero
    rolling_vol = rolling_vol.replace(0, np.nan)

    # Sharpe ratio = mean return / volatility
    sharpe = rolling_mean / rolling_vol

    return sharpe


def compute_ar1_phi(log_returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Computes rolling AR(1) phi values for each ETF.
      + momentum regime (phi > 0)
      + mean reversion regime (phi < 0)
    """

    phi_df = pd.DataFrame(index=log_returns.index, columns=log_returns.columns)

    # Loop through each ETF column, computing a rolling AR(1) coefficient.
    # We approximate phi with the rolling correlation between today's return and
    # yesterday's return, which is the standardized version of the AR(1) slope.
    for col in log_returns.columns:
        series = log_returns[col]
        lagged = series.shift(1)

        temp = pd.DataFrame({"r": series, "lag": lagged}).dropna()
        phi_df[col] = temp["r"].rolling(window).corr(temp["lag"])

    # Clean up once, after all columns are filled (this used to run inside the
    # loop, redoing the whole frame on every ETF for no reason).
    phi_df = phi_df.astype(float).fillna(0).clip(-1, 1)
    return phi_df


