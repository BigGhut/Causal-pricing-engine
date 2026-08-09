"""Offline uplift evaluation metrics: Qini, Uplift@k, percentile curves."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_1d(*arrays: np.ndarray | pd.Series | list) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(a).ravel() for a in arrays)


def qini_auc_score(
    y: np.ndarray | pd.Series,
    uplift: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
) -> float:
    """Area under the Qini curve (normalized by ideal area when possible).

    Ranks samples by predicted uplift (descending) and computes cumulative
    incremental outcomes of treatment vs control. The Qini coefficient is
    the area between the Qini curve and the random baseline.

    Args:
        y: Binary or continuous outcomes.
        uplift: Predicted ITE scores.
        treatment: Binary treatment indicators (0/1).

    Returns:
        Qini AUC (area between model curve and random diagonal), float.
    """
    y_arr, u_arr, t_arr = _as_1d(y, uplift, treatment)
    n = len(y_arr)
    if n == 0:
        return 0.0

    order = np.argsort(-u_arr)
    y_s = y_arr[order]
    t_s = t_arr[order]

    # Cumulative incremental gain
    n_t = 0.0
    n_c = 0.0
    sum_y_t = 0.0
    sum_y_c = 0.0
    qini_values = np.zeros(n + 1, dtype=float)

    for i in range(n):
        if t_s[i] == 1:
            n_t += 1
            sum_y_t += y_s[i]
        else:
            n_c += 1
            sum_y_c += y_s[i]

        # Incremental = treated outcomes − control outcomes scaled to treated size
        if n_c > 0:
            qini_values[i + 1] = sum_y_t - sum_y_c * (n_t / n_c)
        else:
            qini_values[i + 1] = sum_y_t

    # Area under Qini curve via trapezoid; subtract random baseline triangle
    x = np.arange(n + 1, dtype=float)
    area_model = float(np.trapezoid(qini_values, x))
    # Random baseline: straight line from 0 to final qini
    area_random = float(np.trapezoid(np.linspace(0.0, qini_values[-1], n + 1), x))
    return area_model - area_random


def uplift_at_k(
    y: np.ndarray | pd.Series,
    uplift: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    k: float = 0.3,
) -> float:
    """Average treatment effect among the top-k fraction ranked by uplift.

    Args:
        y: Outcomes.
        uplift: Predicted ITE scores.
        treatment: Binary treatment (0/1).
        k: Fraction of population to keep (0 < k ≤ 1). If k > 1, treated as
            absolute count of rows.

    Returns:
        Uplift = mean(y|T=1) − mean(y|T=0) within the top-k subset.
    """
    y_arr, u_arr, t_arr = _as_1d(y, uplift, treatment)
    n = len(y_arr)
    if n == 0:
        return 0.0

    if k <= 0:
        raise ValueError("k must be positive")
    n_top = int(k) if k > 1 else max(1, int(np.ceil(k * n)))
    n_top = min(n_top, n)

    order = np.argsort(-u_arr)[:n_top]
    y_top = y_arr[order]
    t_top = t_arr[order]

    mask_t = t_top == 1
    mask_c = t_top == 0
    if mask_t.sum() == 0 or mask_c.sum() == 0:
        return 0.0

    return float(y_top[mask_t].mean() - y_top[mask_c].mean())


def uplift_by_percentile(
    y: np.ndarray | pd.Series,
    uplift: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute per-percentile uplift for Uplift Curve construction.

    Args:
        y: Outcomes.
        uplift: Predicted ITE scores.
        treatment: Binary treatment (0/1).
        n_bins: Number of equal-size percentile bins (default 10 = deciles).

    Returns:
        DataFrame with columns:
        ``percentile``, ``n``, ``n_treatment``, ``n_control``,
        ``mean_y_treatment``, ``mean_y_control``, ``uplift``.
    """
    y_arr, u_arr, t_arr = _as_1d(y, uplift, treatment)
    n = len(y_arr)
    if n == 0 or n_bins <= 0:
        return pd.DataFrame(
            columns=[
                "percentile",
                "n",
                "n_treatment",
                "n_control",
                "mean_y_treatment",
                "mean_y_control",
                "uplift",
            ]
        )

    order = np.argsort(-u_arr)
    y_s = y_arr[order]
    t_s = t_arr[order]

    # Split into roughly equal bins from highest uplift to lowest
    bins = np.array_split(np.arange(n), n_bins)
    rows: list[dict] = []
    for i, idx in enumerate(bins):
        if len(idx) == 0:
            continue
        y_b = y_s[idx]
        t_b = t_s[idx]
        mask_t = t_b == 1
        mask_c = t_b == 0
        mean_t = float(y_b[mask_t].mean()) if mask_t.any() else np.nan
        mean_c = float(y_b[mask_c].mean()) if mask_c.any() else np.nan
        uplift_val = (
            mean_t - mean_c if not (np.isnan(mean_t) or np.isnan(mean_c)) else np.nan
        )
        rows.append(
            {
                "percentile": (i + 1) * (100 // n_bins),
                "n": int(len(idx)),
                "n_treatment": int(mask_t.sum()),
                "n_control": int(mask_c.sum()),
                "mean_y_treatment": mean_t,
                "mean_y_control": mean_c,
                "uplift": uplift_val,
            }
        )

    return pd.DataFrame(rows)
