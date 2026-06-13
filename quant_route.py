"""
Route Engine: a daily, regime-aware allocation model across five liquid US ETFs.

Pipeline:
  1. download (or load cached) adjusted close prices
  2. turn prices into log returns and a set of rolling signals
  3. score each ETF cross-sectionally each day and turn scores into weights
  4. backtest net of transaction costs and compare against simple benchmarks
  5. write the summary table and figures to results/

Run:  python quant_route.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed; we just save figures
import matplotlib.pyplot as plt

from metrics import (
    compute_log_returns,
    compute_ema_momentum,
    compute_mean_reversion_zscore,
    compute_rolling_volatility,
    compute_volatility_scaled_momentum,
    compute_rolling_sharpe,
    compute_ar1_phi,
)
from scoring import cross_sectional_zscore, compute_score
from portfolio import compute_turnover, smooth_weights, apply_transaction_costs

# ---- config ---------------------------------------------------------------
TICKERS = ["SPY", "QQQ", "IWM", "XLF", "XLI"]
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"

TRADING_DAYS = 252
SIGNAL_WINDOW = 20      # lookback for momentum / vol / reversion / sharpe
PHI_WINDOW = 60         # lookback for the AR(1) regime estimate
MAX_WEIGHT = 0.45       # cap so no single ETF dominates
SMOOTHING_ALPHA = 0.10  # portfolio inertia (lower = slower to trade)
COST_RATE = 0.001       # 10 bps per unit of turnover

DATA_DIR = "data"
RESULTS_DIR = "results"
PRICE_CACHE = os.path.join(DATA_DIR, "etf_adj_close.csv")


# ---- data -----------------------------------------------------------------
def load_prices(tickers, start_date, end_date, cache_path=PRICE_CACHE):
    """
    Load adjusted close prices. Uses a cached CSV if one exists so the backtest
    is reproducible offline; otherwise downloads from Yahoo and caches the result.
    """
    if os.path.exists(cache_path):
        print(f"loading cached prices from {cache_path}")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    import yfinance as yf  # imported lazily so offline runs don't need it
    print("downloading etf prices from yahoo")
    data = yf.download(tickers, start=start_date, end=end_date,
                       auto_adjust=True, progress=False)
    prices = data["Close"].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
    prices = prices[tickers]  # keep a stable column order

    os.makedirs(DATA_DIR, exist_ok=True)
    prices.to_csv(cache_path)
    return prices


# ---- signals + weights ----------------------------------------------------
def build_weights(prices):
    """
    Turn prices into the daily target weights. Returns the smoothed weights,
    the daily log returns, and the per-day turnover.
    """
    log_returns = compute_log_returns(prices)

    # rolling signals (all point-in-time: each value uses only past data)
    ema_momentum = compute_ema_momentum(log_returns, SIGNAL_WINDOW)
    volatility = compute_rolling_volatility(log_returns, SIGNAL_WINDOW)
    scaled_momentum = compute_volatility_scaled_momentum(ema_momentum, volatility)
    reversion_z = compute_mean_reversion_zscore(log_returns, SIGNAL_WINDOW)
    rolling_sharpe = compute_rolling_sharpe(log_returns, SIGNAL_WINDOW)
    phi = compute_ar1_phi(log_returns, PHI_WINDOW)

    # standardize momentum signals across the basket, clip the sharpe filter
    mom_z = cross_sectional_zscore(ema_momentum)
    scaled_mom_z = cross_sectional_zscore(scaled_momentum)
    sharpe_z = rolling_sharpe.clip(lower=-2, upper=2)

    # composite score per ETF per day
    scores = compute_score(mom_z, scaled_mom_z, reversion_z, sharpe_z, phi)

    # keep positive scores, normalize to weights, cap, then renormalize
    active = scores.clip(lower=0)
    active = active.div(active.sum(axis=1), axis=0).fillna(0)
    active = active.clip(upper=MAX_WEIGHT)
    active = active.div(active.sum(axis=1), axis=0).fillna(0)

    # how aggressive to be today: blend active bets with an equal-weight baseline,
    # tilting toward the active view when the average regime favors trending
    phi_market = phi.mean(axis=1).clip(-1, 1)
    active_exposure = (phi_market + 1) / 2  # map [-1, 1] -> [0, 1]
    baseline = pd.DataFrame(1 / len(prices.columns), index=active.index, columns=active.columns)
    weights = active.mul(active_exposure, axis=0) + baseline.mul(1 - active_exposure, axis=0)

    weights = smooth_weights(weights, alpha=SMOOTHING_ALPHA)
    turnover = compute_turnover(weights)
    return weights, log_returns, turnover


def backtest(weights, log_returns, turnover):
    """Net daily returns: yesterday's weights applied to today's returns, minus costs."""
    gross = (weights.shift(1) * log_returns).sum(axis=1)
    net = apply_transaction_costs(gross, turnover, cost_rate=COST_RATE)
    return net.dropna()


# ---- performance ----------------------------------------------------------
def performance_stats(returns):
    """Sharpe, Sortino, annualized return/vol, and max drawdown for a return series."""
    ann_return = returns.mean() * TRADING_DAYS
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_return / ann_vol if ann_vol else np.nan
    downside = returns[returns < 0].std()
    sortino = (returns.mean() / downside * np.sqrt(TRADING_DAYS)) if downside else np.nan
    equity = returns.cumsum().apply(np.exp)
    max_dd = (equity / equity.cummax() - 1).min()
    return {
        "Ann. Return": ann_return,
        "Ann. Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "Final Equity": equity.iloc[-1],
    }


def benchmark_returns(log_returns):
    """Two honest yardsticks: holding SPY, and equal-weighting the whole basket."""
    return {
        "SPY (buy & hold)": log_returns["SPY"],
        "Equal weight": log_returns.mean(axis=1),
    }


# ---- figures --------------------------------------------------------------
def plot_equity_curves(strategy_net, benchmarks, path):
    plt.figure(figsize=(10, 5))
    plt.plot(strategy_net.cumsum().apply(np.exp), label="Route Engine", linewidth=2)
    for name, r in benchmarks.items():
        r = r.reindex(strategy_net.index).fillna(0)
        plt.plot(r.cumsum().apply(np.exp), label=name, alpha=0.7, linewidth=1.2)
    plt.title("Equity Curve vs Benchmarks (2015-2025)")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_drawdown(strategy_net, path):
    equity = strategy_net.cumsum().apply(np.exp)
    drawdown = equity / equity.cummax() - 1
    plt.figure(figsize=(10, 4))
    plt.fill_between(drawdown.index, drawdown.values, 0, color="tab:red", alpha=0.4)
    plt.title("Route Engine Drawdown")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_allocation_heatmap(weights, path):
    plt.figure(figsize=(10, 3))
    plt.imshow(weights.T.values, aspect="auto", cmap="viridis",
               extent=[0, len(weights), 0, len(weights.columns)])
    plt.yticks(np.arange(len(weights.columns)) + 0.5, weights.columns[::-1])
    plt.xlabel("Trading day")
    plt.title("Dynamic ETF Allocation Over Time")
    plt.colorbar(label="Weight")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def monte_carlo(strategy_net, path, horizon=TRADING_DAYS, n_paths=500, seed=0):
    """
    Bootstrap the strategy's daily returns into many 1-year paths to see the
    spread of outcomes rather than trusting the single historical run.
    Returns the probability of ending the year positive.
    """
    rng = np.random.default_rng(seed)
    daily = strategy_net.values
    draws = rng.choice(daily, size=(horizon, n_paths), replace=True)
    paths = np.exp(np.cumsum(draws, axis=0))

    plt.figure(figsize=(9, 5))
    plt.plot(paths, color="tab:blue", alpha=0.03)
    plt.plot(np.median(paths, axis=1), color="black", linewidth=2, label="Median path")
    plt.title(f"Monte Carlo Simulated Equity Paths ({n_paths} runs, 1-year)")
    plt.xlabel("Trading day")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    return float((paths[-1] > 1).mean())


# ---- main -----------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    prices = load_prices(TICKERS, START_DATE, END_DATE)
    weights, log_returns, turnover = build_weights(prices)
    net = backtest(weights, log_returns, turnover)

    # align benchmarks to the strategy's valid window
    benchmarks = {k: v.reindex(net.index) for k, v in benchmark_returns(log_returns).items()}

    # summary table: strategy next to its benchmarks
    rows = {"Route Engine": performance_stats(net)}
    for name, r in benchmarks.items():
        rows[name] = performance_stats(r.dropna())
    summary = pd.DataFrame(rows).T
    summary["Avg Turnover"] = [turnover.mean()] + [np.nan] * len(benchmarks)

    pd.set_option("display.float_format", lambda x: f"{x:0.3f}")
    print("\n=== Performance summary (2015-2025, net of costs) ===")
    print(summary)
    summary.to_csv(os.path.join(RESULTS_DIR, "performance_summary.csv"))

    # figures
    plot_equity_curves(net, benchmarks, os.path.join(RESULTS_DIR, "equity_curve.png"))
    plot_drawdown(net, os.path.join(RESULTS_DIR, "drawdown.png"))
    plot_allocation_heatmap(weights.loc[net.index], os.path.join(RESULTS_DIR, "allocation_heatmap.png"))
    prob_positive = monte_carlo(net, os.path.join(RESULTS_DIR, "monte_carlo.png"))

    print(f"\nMonte Carlo: probability of a positive year = {prob_positive:0.1%}")
    print(f"figures written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
