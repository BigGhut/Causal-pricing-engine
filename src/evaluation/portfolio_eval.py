"""One pre-registered portfolio split.

Version 2 chooses the score threshold on the calibration curve only.
The untouched test checks that frozen threshold and does not move it.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from src.causal.uplift_models import TLearner, XLearner
from src.data.synthetic import (
    FEATURE_COLUMNS,
    generate_uplift_dataset,
    summarize_calibration,
)
from src.evaluation.metrics import (
    decile_calibration,
    pehe,
    qini_bootstrap_interval,
    qini_random_interval,
    split_train_calibration_test,
    uplift_at_k,
)
from src.evaluation.protocol import (
    CALIBRATION_BOOT_SEED,
    FALSE_OVERRIDE_RATE_MAX,
    MODEL_RANDOM_STATE,
    N_BOOT,
    N_RANDOM_DRAWS,
    RANDOM_SCORE_SEED,
    SCORE_THRESHOLD,
    SPLIT_SEED,
    TEST_BOOT_SEED,
)

# Fixed before the version-2 run. Not derived from scores.
THETA_GRID = tuple(i / 100 for i in range(51))


def rates_at_theta(
    scores: np.ndarray,
    true_uplift: np.ndarray,
    theta: float,
) -> dict[str, float | int | None]:
    """False-override rate and sleeping-dog catch rate at one frozen threshold."""
    score_arr = np.asarray(scores, dtype=float).ravel()
    true_arr = np.asarray(true_uplift, dtype=float).ravel()
    fired = score_arr < -float(theta)
    dogs = true_arr < 0.0
    n_fired = int(fired.sum())
    n_dogs = int(dogs.sum())
    if n_dogs == 0:
        caught: float | None = None
    else:
        caught = float((fired & dogs).sum() / n_dogs)
    if n_fired == 0:
        false_rate: float | None = None
    else:
        false_rate = float((fired & (true_arr >= 0.0)).sum() / n_fired)
    return {
        "theta": float(theta),
        "n_fired": n_fired,
        "false_override_rate": false_rate,
        "caught_rate": caught,
    }


def operating_curve(
    scores: np.ndarray,
    true_uplift: np.ndarray,
    thetas: tuple[float, ...] = THETA_GRID,
) -> list[dict[str, float | int | None]]:
    """Calibration curve. Each point is one pre-registered theta."""
    return [rates_at_theta(scores, true_uplift, theta) for theta in thetas]


def choose_operating_point(
    curve: list[dict[str, float | int | None]],
    *,
    rate_max: float = FALSE_OVERRIDE_RATE_MAX,
) -> dict[str, float | int | None] | None:
    """Max sleeping-dog catch rate among points with false rate at most rate_max.

    Points that never fire are not eligible. Ties break toward the smaller
    false rate, then the smaller theta. The rule is fixed in PLAN.md version 2.
    """
    feasible: list[dict[str, float | int | None]] = []
    for row in curve:
        n_fired = int(row["n_fired"])
        false_rate = row["false_override_rate"]
        caught = row["caught_rate"]
        if n_fired < 1 or false_rate is None or caught is None:
            continue
        if float(false_rate) <= float(rate_max):
            feasible.append(row)
    if not feasible:
        return None
    feasible.sort(
        key=lambda row: (
            -float(row["caught_rate"]),
            float(row["false_override_rate"]),
            float(row["theta"]),
        )
    )
    return feasible[0]


def _learner(name: str):
    base = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=3,
        random_state=MODEL_RANDOM_STATE,
    )
    if name == "t_learner":
        return TLearner(base_estimator=base)
    if name == "x_learner":
        return XLearner(base_estimator=base)
    raise ValueError(f"unknown learner: {name}")


def run_split(
    *,
    n: int,
    generation_seed: int,
    learner_name: str,
    threshold: float = SCORE_THRESHOLD,
) -> dict:
    """Fit on train, lock the flag on calibration, then score the untouched test.

    Nothing computed on the test is passed back into the flag.
    """
    frame = generate_uplift_dataset(n=n, random_state=generation_seed)
    treatment = frame["treatment"].to_numpy(dtype=int)
    train_idx, cal_idx, test_idx = split_train_calibration_test(
        len(frame), treatment, random_state=SPLIT_SEED
    )
    features = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    accepted = frame["accepted"].to_numpy(dtype=int)
    truth = frame["true_uplift"].to_numpy(dtype=float)
    segment = frame["segment"].to_numpy(dtype=float)

    model = _learner(learner_name)
    model.fit(features[train_idx], accepted[train_idx], treatment[train_idx])
    predicted_cal = np.asarray(model.predict_uplift(features[cal_idx]), dtype=float)
    cal_qini = qini_bootstrap_interval(
        accepted[cal_idx],
        predicted_cal,
        treatment[cal_idx],
        n_boot=N_BOOT,
        seed=CALIBRATION_BOOT_SEED,
    )
    curve = operating_curve(predicted_cal, truth[cal_idx])
    chosen = choose_operating_point(curve)
    predicted_test = np.asarray(model.predict_uplift(features[test_idx]), dtype=float)
    test_qini = qini_bootstrap_interval(
        accepted[test_idx],
        predicted_test,
        treatment[test_idx],
        n_boot=N_BOOT,
        seed=TEST_BOOT_SEED,
    )
    random_qini = qini_random_interval(
        accepted[test_idx],
        treatment[test_idx],
        n_draws=N_RANDOM_DRAWS,
        seed=RANDOM_SCORE_SEED,
    )
    segments = summarize_calibration(
        segment[test_idx], truth[test_idx], predicted_test
    )
    for row in segments:
        row["bias"] = float(row["mean_predicted"]) - float(row["planted"])
    if chosen is None:
        score_threshold = None
        cal_false = None
        cal_caught = None
        cal_fired = 0
        test_at = {
            "n_fired": 0,
            "false_override_rate": None,
            "caught_rate": None,
        }
        flag = False
    else:
        score_threshold = float(chosen["theta"])
        cal_false = chosen["false_override_rate"]
        cal_caught = chosen["caught_rate"]
        cal_fired = int(chosen["n_fired"])
        test_at = rates_at_theta(predicted_test, truth[test_idx], score_threshold)
        flag = (
            int(test_at["n_fired"]) >= 1
            and test_at["false_override_rate"] is not None
            and float(test_at["false_override_rate"]) <= FALSE_OVERRIDE_RATE_MAX
        )
    return {
        "generation_seed": int(generation_seed),
        "learner": learner_name,
        "n_train": int(len(train_idx)),
        "n_calibration": int(len(cal_idx)),
        "n_test": int(len(test_idx)),
        "calibration_qini": float(cal_qini["point"]),
        "calibration_qini_low": float(cal_qini["low"]),
        "calibration_qini_high": float(cal_qini["high"]),
        "calibration_n_boot": int(cal_qini["n_boot"]),
        "score_threshold": score_threshold,
        "calibration_curve": curve,
        "calibration_n_fired": cal_fired,
        "false_override_rate": None if cal_false is None else float(cal_false),
        "calibration_caught_rate": None if cal_caught is None else float(cal_caught),
        "false_override_rate_max": FALSE_OVERRIDE_RATE_MAX,
        "ranking_supports_decision": flag,
        "test_qini": float(test_qini["point"]),
        "test_qini_low": float(test_qini["low"]),
        "test_qini_high": float(test_qini["high"]),
        "test_n_boot": int(test_qini["n_boot"]),
        "random_qini_mean": float(random_qini["mean"]),
        "random_qini_low": float(random_qini["low"]),
        "random_qini_high": float(random_qini["high"]),
        "random_n_draws": int(random_qini["n_draws"]),
        "uplift_at_k": float(
            uplift_at_k(accepted[test_idx], predicted_test, treatment[test_idx], k=0.3)
        ),
        "pehe": pehe(predicted_test, truth[test_idx]),
        "segments": segments,
        "test_n_fired": int(test_at["n_fired"]),
        "test_false_override_rate": test_at["false_override_rate"],
        "test_caught_rate": test_at["caught_rate"],
        "deciles": decile_calibration(
            accepted[test_idx], treatment[test_idx], predicted_test
        ),
        "model": model,
        "test_features": features[test_idx],
        "test_predicted": predicted_test,
        "test_truth": truth[test_idx],
    }
