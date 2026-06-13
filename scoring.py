import pandas as pd
import numpy as np


def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize each row across the ETF basket (point-in-time).

    For every day we take that day's values across the 5 ETFs and turn them into
    z-scores, so a score answers "how strong is this ETF relative to its peers
    today." This is done per-row (axis=1), so it only ever uses information from
    that single day. That matters: an earlier version standardized each ETF over
    the *whole* 2015-2025 sample, which leaks future data into past scores
    (look-ahead bias). Cross-sectional scoring avoids that and is also what makes
    the signals comparable across ETFs, which is the whole point of standardizing.
    """
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)  # avoid divide-by-zero on flat days
    return df.sub(mean, axis=0).div(std, axis=0)


def compute_style_weights(phi: pd.DataFrame):
    """
    Convert the AR(1) phi (regime indicator) into momentum vs mean-reversion
    weights in [0, 1].
      phi > 0  -> trends tend to continue   -> lean on momentum
      phi < 0  -> moves tend to reverse      -> lean on mean reversion
    """
    phi_raw = phi.clip(lower=-1, upper=1)
    momentum_weight = (phi_raw + 1) / 2
    reversion_weight = 1 - momentum_weight
    return momentum_weight, reversion_weight


def compute_score(mom_z, scaled_mom_z, rev_z, sharpe_z, phi):
    """
    Combine the individual signals into one composite score per ETF per day:
       1. blend raw and volatility-scaled momentum into a momentum sub-score
       2. flip the mean-reversion z-score (a low recent return = bounce-back
          opportunity, so it should score positively)
       3. weight momentum vs reversion by the regime (phi)
       4. nudge the score up or down by the rolling Sharpe (a confidence filter)
       5. re-standardize the final score across ETFs so weights are comparable
    """
    # 1 + 2. sub-scores
    momentum_score = 0.7 * mom_z + 0.3 * scaled_mom_z
    reversion_score = -rev_z

    # 3. regime-driven blend of the two styles
    momentum_weight, reversion_weight = compute_style_weights(phi)
    style_signal = momentum_weight * momentum_score + reversion_weight * reversion_score

    # 4. Sharpe as a confidence adjustment (clipped so it can't dominate)
    sharpe_adj = sharpe_z.clip(lower=-0.5, upper=1.5)
    raw_score = style_signal * (1 + sharpe_adj)

    # 5. point-in-time standardization across the basket
    final_score = cross_sectional_zscore(raw_score)
    return final_score
