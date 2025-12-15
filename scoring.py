import pandas as pd
import numpy as np

def zscore(df):
    return (df - df.mean()) / df.std()

def compute_style_weights(phi_60):
    """
    Convert phi_z (momentum vs mean reversion indicator)
    into weights in [0, 1].
    """
    phi_raw = phi_60.clip(lower=-1, upper=1)  # use raw AR(1) phi
    momentum_weight = (phi_raw + 1) / 2
    reversion_weight = 1 - momentum_weight
    return momentum_weight, reversion_weight

def compute_score(mom_z, scaled_mom_z, rev_z, sharpe_z, phi_60):
    """
    Full scoring system:
       - Momentum score
       - Reversion score
       - AR(1) regime weighting
       - Sharpe quality adjustment
    """

    # 1. Subscores
    momentum_score = 0.7 * mom_z + 0.3 * scaled_mom_z
    reversion_score = -rev_z
    # 2. Regime weights                 ↓ method to assess weight of momentum & mean-reversion based on φ
    momentum_weight, reversion_weight = compute_style_weights(phi_60)
    # 3. Combine using phi-driven weighting
    style_signal = (
        momentum_weight * momentum_score +
        reversion_weight * reversion_score
    )
    # 4. Sharpe as confidence adjustment
    sharpe_adj = sharpe_z.clip(lower=-0.5, upper=1.5)
    raw_score = style_signal * (1 + sharpe_adj)
    final_score = zscore(raw_score)
    return final_score
