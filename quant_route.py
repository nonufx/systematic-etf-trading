import os                # for working with folders/files
import yfinance as yf    # to download ETF price data
import pandas as pd      # to handle tables (dataframes)
import numpy as np       # for math operations
import matplotlib.pyplot as plt
from metrics import (
    compute_log_returns,
    compute_ema_momentum,
    compute_mean_reversion_zscore,
    compute_rolling_volatility,
    compute_volatility_scaled_momentum,
    compute_rolling_sharpe,
    compute_ar1_phi
)
from scoring import zscore, compute_score
from portfolio import compute_turnover, smooth_weights, apply_transaction_costs



def ensure_data_folder():
    """
    Make sure there is a folder called 'data' to save files in.
    If it doesn't exist, create it.
    """
    if not os.path.exists("data"):
        os.makedirs("data")

def download_etf_prices(tickers, start_date, end_date):
    """
    Downloads daily adjusted close prices for a list of ETFs.
    """
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

    # gets just the close price for each etf
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close'].copy()
    else:
        prices = data.copy()
    return prices


def main():
    # Make sure 'data' folder exists
    ensure_data_folder()

    tickers = ["SPY", "QQQ", "IWM", "XLF", "XLI"]
    start_date = "2015-01-01"
    end_date = "2025-01-01"

    print("downloading etf prices")
    prices = download_etf_prices(tickers, start_date, end_date)
    prices.to_csv("data/etf_adj_close.csv")

    print("first 5 rows of prices")
    print(prices.head())

    # 1. Log Returns
    log_returns = compute_log_returns(prices)
    log_returns.to_csv("./data/etf_log_returns.csv")

    # mean reversion (20 day)
    mean_rev_z = compute_mean_reversion_zscore(log_returns, window=20)

    # 2. EMA momentum (20-day)
    ema_mom_20 = compute_ema_momentum(log_returns, window=20)

    # 3. Rolling volatility (20-day)
    vol_20 = compute_rolling_volatility(log_returns, window=20)

    # 4. volatility scaling
    scaled_momentum = compute_volatility_scaled_momentum(ema_mom_20, vol_20)

    # 5. rolling sharpe
    sharpe_20 = compute_rolling_sharpe(log_returns, window = 20)

    # 6. AR(1) PHI — regime detection
    phi_60 = compute_ar1_phi(log_returns, window=60)

    # 8. Z-score the signals
    mom_z = zscore(ema_mom_20)
    scaled_mom_z = zscore(scaled_momentum)
    rev_z = mean_rev_z
    sharpe_clip = sharpe_z = sharpe_20.clip(lower=-2, upper=2)

    # 9. Compute the final score
    scores = compute_score(mom_z, scaled_mom_z, rev_z, sharpe_z, phi_60)

    print("\nFinal Combined Scores (sample):")
    print(scores.tail())
    # takes only positive scores
    scores_pos = scores.clip(lower=0)
    active_w = scores_pos.div(scores_pos.sum(axis=1), axis=0).fillna(0)

    max_weight = 0.45
    active_w = active_w.clip(upper=max_weight)
    active_w = active_w.div(active_w.sum(axis=1), axis=0).fillna(0)

    # tells us how aggressive the quant sleeve should be on a certain day
    phi_market = phi_60.mean(axis=1).clip(-1, 1)
    active_exposure = (phi_market + 1) / 2  # [-1,1] -> [0,1]
    baseline_w = pd.DataFrame(1 / len(tickers), index=active_w.index, columns=active_w.columns)
    weights = active_w.mul(active_exposure, axis=0) + baseline_w.mul(1 - active_exposure, axis=0)


    weights_smooth = smooth_weights(weights, alpha=0.1)

    turnover = compute_turnover(weights_smooth)


    # Portfolio gross returns
    portfolio_returns = (weights_smooth.shift(1) * log_returns).sum(axis=1)

    # Apply transaction costs
    net_returns = apply_transaction_costs(
        portfolio_returns,
        turnover,
        cost_rate=0.001
    )

    # Quant Sleeve Equity Curve
    equity_curve = net_returns.cumsum().apply(np.exp)



    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1


    print("Max drawdown:", drawdown.min())

    sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252)
    print("Sharpe ratio:", sharpe)


if __name__ == "__main__":
    main()

