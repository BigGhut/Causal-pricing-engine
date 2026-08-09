"""Statistical power calculations and CUPED variance reduction."""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def cuped_adjust(
    df: pd.DataFrame,
    target_col: str,
    covariate_col: str,
) -> pd.Series:
    """CUPED (Controlled Experiment Using Pre-Experiment Data) variance reduction.

    Y_adjusted = Y - θ * (X - E[X]), where θ = Cov(Y, X) / Var(X).
    """
    y = df[target_col]
    x = df[covariate_col]
    
    cov = np.cov(y, x)[0, 1]
    var_x = np.var(x, ddof=1)
    
    if var_x == 0:
        return y
        
    theta = cov / var_x
    mean_x = np.mean(x)
    
    return y - theta * (x - mean_x)


def sample_size_calculator(
    std_dev: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Calculate minimum required sample size per group for two-sample t-test."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    
    n = 2 * ((z_alpha + z_beta) * std_dev / mde) ** 2
    return int(np.ceil(n))
