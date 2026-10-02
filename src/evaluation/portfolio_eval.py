"""One pre-registered portfolio split. The calibration slice sets the flag only."""

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
    false_override_rate,
    holdout_decision,
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
    rate_cal = false_override_rate(
        predicted_cal, truth[cal_idx], threshold=threshold
    )
    decision = holdout_decision(
        float(cal_qini["low"]),
        false_override_rate=rate_cal,
        false_override_rate_max=FALSE_OVERRIDE_RATE_MAX,
    )

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
        "false_override_rate": decision["false_override_rate"],
        "false_override_rate_max": decision["false_override_rate_max"],
        "false_override_check": decision["false_override_check"],
        "ranking_supports_decision": decision["ranking_supports_decision"],
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
        "test_false_override_rate": false_override_rate(
            predicted_test, truth[test_idx], threshold=threshold
        ),
        "deciles": decile_calibration(
            accepted[test_idx], treatment[test_idx], predicted_test
        ),
        "model": model,
        "test_features": features[test_idx],
        "test_predicted": predicted_test,
        "test_truth": truth[test_idx],
    }
