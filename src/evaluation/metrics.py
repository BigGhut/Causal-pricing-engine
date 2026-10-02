"""Offline uplift evaluation metrics: Qini coefficient, Uplift@k, percentile curves."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_1d(*arrays: np.ndarray | pd.Series | list) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(a).ravel() for a in arrays)


def _qini_cumulative(
    y: np.ndarray,
    treatment: np.ndarray,
    order: np.ndarray,
) -> np.ndarray:
    """Cumulative Qini curve for a given ranking (length n+1, starts at 0).

    At step k (after first k ranked samples):
        Q(k) = sum(y|T=1) - sum(y|T=0) * (n_t / n_c)
    i.e. incremental treated outcomes vs scaled control outcomes.
    """
    y_s = np.asarray(y[order], dtype=float)
    treated = np.asarray(treatment[order]) == 1
    n = len(y_s)
    qini = np.zeros(n + 1, dtype=float)
    if n == 0:
        return qini
    sum_y_t = np.cumsum(np.where(treated, y_s, 0.0))
    sum_y_c = np.cumsum(np.where(treated, 0.0, y_s))
    n_t = np.cumsum(treated)
    n_c = np.cumsum(~treated)
    body = np.where(n_c > 0, sum_y_t - sum_y_c * (n_t / np.maximum(n_c, 1)), sum_y_t)
    qini[1:] = body
    return qini


def _area_above_random(qini: np.ndarray) -> float:
    """∫(Qini_model - Qini_random) over population fraction x ∈ [0, 1].

    Using fraction on the x-axis avoids the old O(n²) raw trapz over sample index
    that produced misleading values like ~10_000 on n≈900.
    """
    n = len(qini) - 1
    if n <= 0:
        return 0.0
    x = np.linspace(0.0, 1.0, n + 1)
    area_model = float(np.trapezoid(qini, x))
    # Random baseline: straight line from 0 to final Qini value
    area_random = float(np.trapezoid(np.linspace(0.0, qini[-1], n + 1), x))
    return area_model - area_random


def qini_auc_score(
    y: np.ndarray | pd.Series,
    uplift: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    *,
    normalize: bool = True,
) -> float:
    """Qini coefficient (default) or unnormalized Qini area.

    Ranks samples by predicted uplift (descending) and builds the cumulative
    Qini curve. The reported score is the area between the model curve and the
    random diagonal, with x = population fraction in ``[0, 1]``.

    When ``normalize=True`` (default), that area is divided by the same area for
    an **oracle ranking** that uses labels only for evaluation (not for training):

        perfect_score_i = y_i * (2 * t_i - 1)

    so treated responders rank high and control responders rank low. The ratio
    is a Qini *coefficient*-style number typically in roughly ``[-1, 1]``
    (can slightly exceed bounds on small samples).

    Args:
        y: Binary or continuous outcomes.
        uplift: Predicted ITE scores (higher = prioritize for treatment).
        treatment: Binary treatment indicators (0/1).
        normalize: If True, return coefficient vs oracle; if False, return
            unnormalized area over population fraction (still O(effect), not O(n²)).

    Returns:
        float score. Higher is better. Random ranking → ≈ 0 when normalized.
    """
    y_arr, u_arr, t_arr = _as_1d(y, uplift, treatment)
    n = len(y_arr)
    if n == 0:
        return 0.0
    if len(u_arr) != n or len(t_arr) != n:
        raise ValueError("y, uplift, and treatment must have the same length")

    order = np.argsort(-u_arr, kind="mergesort")
    qini_model = _qini_cumulative(y_arr, t_arr, order)
    area_model = _area_above_random(qini_model)

    if not normalize:
        return float(area_model)

    # Oracle ranking for denominator (evaluation only — uses y,t not model)
    perfect_scores = y_arr * (2.0 * t_arr - 1.0)
    order_star = np.argsort(-perfect_scores, kind="mergesort")
    qini_star = _qini_cumulative(y_arr, t_arr, order_star)
    area_star = _area_above_random(qini_star)

    # Degenerate: no incremental structure → coefficient 0
    if abs(area_star) < 1e-12:
        return 0.0

    return float(area_model / area_star)


def qini_auc_score_unnormalized(
    y: np.ndarray | pd.Series,
    uplift: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
) -> float:
    """Unnormalized Qini area over population fraction (for debugging)."""
    return qini_auc_score(y, uplift, treatment, normalize=False)


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


def holdout_decision(
    qini_low: float,
    *,
    false_override_rate: float | None = None,
    false_override_rate_max: float | None = None,
) -> dict[str, float | str | bool | None]:
    """Decide, at training time, whether scores may change a fare.

    Pass the Qini lower bound from the calibration slice, not from the untouched test.
    The flag is true only when that bound is above zero and, once a maximum is
    set, the false-override rate is defined and not greater than the maximum.
    A rate equal to the maximum passes. An undefined rate fails.
    """
    qini_ok = float(qini_low) > 0.0
    if false_override_rate_max is None:
        rate_ok = True
        rate_check = "not_applied"
    elif false_override_rate is None:
        rate_ok = False
        rate_check = "undefined"
    else:
        rate_ok = float(false_override_rate) <= float(false_override_rate_max)
        rate_check = "pass" if rate_ok else "fail"
    return {
        "ranking_supports_decision": bool(qini_ok and rate_ok),
        "false_override_rate": None if false_override_rate is None else float(false_override_rate),
        "false_override_rate_max": (
            None if false_override_rate_max is None else float(false_override_rate_max)
        ),
        "false_override_check": rate_check,
    }


def false_override_rate(
    scores: np.ndarray | pd.Series,
    true_uplift: np.ndarray | pd.Series,
    *,
    threshold: float = 0.05,
) -> float | None:
    """Share of fired overrides that hit a row whose true effect is not negative.

    An override fires when ``score < -threshold``. It is false when
    ``true_uplift >= 0``. The rate divides by the number of fired overrides.
    No fired override makes the rate undefined.
    """
    score_arr, true_arr = _as_1d(scores, true_uplift)
    fired = score_arr < -float(threshold)
    n_fired = int(fired.sum())
    if n_fired == 0:
        return None
    false = fired & (true_arr >= 0.0)
    return float(false.sum() / n_fired)


def pehe(
    predicted: np.ndarray | pd.Series,
    true_uplift: np.ndarray | pd.Series,
) -> float:
    """Root mean squared error of the score against the row-level true effect."""
    pred_arr, true_arr = _as_1d(predicted, true_uplift)
    return float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))


def decile_calibration(
    y: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
) -> list[dict[str, float | int | None]]:
    """Equal-count bins of predicted effect, low scores in decile 1.

    ``observed_difference`` is mean outcome in the treated rows of the bin
    minus mean outcome in the control rows. It is None when a bin lacks an arm.
    """
    y_arr, t_arr, p_arr = _as_1d(y, treatment, predicted)
    n = len(p_arr)
    if n == 0 or n_bins <= 0:
        return []
    order = np.argsort(p_arr, kind="mergesort")
    rows: list[dict[str, float | int | None]] = []
    for i, idx in enumerate(np.array_split(order, n_bins)):
        if len(idx) == 0:
            continue
        treated = t_arr[idx] == 1
        control = ~treated
        observed: float | None
        if treated.any() and control.any():
            observed = float(y_arr[idx][treated].mean() - y_arr[idx][control].mean())
        else:
            observed = None
        rows.append(
            {
                "decile": i + 1,
                "n": int(len(idx)),
                "mean_predicted": float(p_arr[idx].mean()),
                "observed_difference": observed,
            }
        )
    return rows


def read_ranking_supports_decision(metrics: dict | None) -> bool:
    """Read the flag written at training. Do not recompute it from ``qini_low``."""
    if not isinstance(metrics, dict):
        return False
    return metrics.get("ranking_supports_decision") is True


def split_train_calibration_test(
    n: int,
    treatment: np.ndarray,
    *,
    random_state: int = 42,
    holdout_size: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train, then split the holdout into a calibration slice and an untouched test.

    The decision flag uses only the calibration slice. Reported Qini uses only
    the test slice. The two slices are the same size and do not overlap.
    """
    from sklearn.model_selection import train_test_split

    treatment = np.asarray(treatment).ravel()
    indices = np.arange(n)
    if len(treatment) != n:
        raise ValueError("treatment length must equal n")
    train_idx, hold_idx = train_test_split(
        indices,
        test_size=holdout_size,
        random_state=random_state,
        stratify=treatment,
    )
    cal_idx, test_idx = train_test_split(
        hold_idx,
        test_size=0.5,
        random_state=random_state,
        stratify=treatment[hold_idx],
    )
    return (
        np.asarray(train_idx),
        np.asarray(cal_idx),
        np.asarray(test_idx),
    )


def qini_bootstrap_interval(
    y: np.ndarray | pd.Series,
    uplift: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    *,
    n_boot: int = 200,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Percentile interval for the Qini coefficient on one holdout.

    Each draw resamples rows with replacement and recomputes the coefficient.
    Draws that miss an arm are skipped.
    """
    y_arr, u_arr, t_arr = _as_1d(y, uplift, treatment)
    n = len(y_arr)
    point = float(qini_auc_score(y_arr, u_arr, t_arr))
    if n == 0 or n_boot <= 0:
        return {"point": point, "low": point, "high": point, "n_boot": 0}

    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(t_arr[idx])) < 2:
            continue
        scores.append(float(qini_auc_score(y_arr[idx], u_arr[idx], t_arr[idx])))
    if not scores:
        return {"point": point, "low": point, "high": point, "n_boot": 0}
    arr = np.asarray(scores, dtype=float)
    tail = alpha / 2.0
    return {
        "point": point,
        "low": float(np.quantile(arr, tail)),
        "high": float(np.quantile(arr, 1.0 - tail)),
        "n_boot": int(len(arr)),
    }


def qini_random_interval(
    y: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    *,
    n_draws: int = 200,
    seed: int = 1,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Distribution of the Qini coefficient under random scores.

    One random score is one draw from this distribution, not a null by itself.
    """
    y_arr, t_arr = _as_1d(y, treatment)
    n = len(y_arr)
    if n == 0 or n_draws <= 0:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "n_draws": 0}
    rng = np.random.default_rng(seed)
    scores = [
        float(qini_auc_score(y_arr, rng.normal(size=n), t_arr))
        for _ in range(int(n_draws))
    ]
    arr = np.asarray(scores, dtype=float)
    tail = alpha / 2.0
    return {
        "mean": float(arr.mean()),
        "low": float(np.quantile(arr, tail)),
        "high": float(np.quantile(arr, 1.0 - tail)),
        "n_draws": int(len(arr)),
    }
