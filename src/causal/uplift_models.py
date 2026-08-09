"""Uplift meta-learners: T-Learner, S-Learner, X-Learner."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone


class EstimatorProtocol(Protocol):
    """Protocol for base machine learning estimators."""

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray) -> EstimatorProtocol: ...

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray: ...

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray: ...


class BaseUpliftModel:
    """Base class for Uplift Meta-Learners."""

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        treatment: np.ndarray,
    ) -> BaseUpliftModel:
        raise NotImplementedError

    def predict_uplift(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        raise NotImplementedError


def _predict_outcome(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """Predict outcome probability (if classifier) or continuous value."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()
    return np.asarray(model.predict(X), dtype=float).ravel()


class TLearner(BaseUpliftModel):
    """Two-Model Learner (T-Learner) for Individual Treatment Effect (ITE).

    Models Y(1) and Y(0) with two separate base estimators:
    τ̂(x) = μ̂₁(x) − μ̂₀(x)
    """

    def __init__(self, base_estimator: BaseEstimator) -> None:
        self.base_estimator = base_estimator
        self.model_control: BaseEstimator | None = None
        self.model_treatment: BaseEstimator | None = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        treatment: np.ndarray,
    ) -> TLearner:
        """Fit separate models for control (treatment=0) and treatment (treatment=1)."""
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        w_arr = np.asarray(treatment)

        mask_control = w_arr == 0
        mask_treatment = w_arr == 1

        self.model_control = clone(self.base_estimator)
        self.model_treatment = clone(self.base_estimator)

        self.model_control.fit(X_arr[mask_control], y_arr[mask_control])
        self.model_treatment.fit(X_arr[mask_treatment], y_arr[mask_treatment])

        return self

    def predict_uplift(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Calculate predicted uplift: μ̂₁(x) − μ̂₀(x)."""
        if self.model_control is None or self.model_treatment is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        X_arr = np.asarray(X)
        pred_treatment = _predict_outcome(self.model_treatment, X_arr)
        pred_control = _predict_outcome(self.model_control, X_arr)
        return pred_treatment - pred_control


class SLearner(BaseUpliftModel):
    """Single-Model Learner (S-Learner).

    One model on features concatenated with treatment indicator:
    τ̂(x) = μ̂(x, W=1) − μ̂(x, W=0)
    """

    def __init__(self, base_estimator: BaseEstimator) -> None:
        self.base_estimator = base_estimator
        self.model: BaseEstimator | None = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        treatment: np.ndarray,
    ) -> SLearner:
        """Fit a single model on (X, W) → Y."""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        w_arr = np.asarray(treatment, dtype=float).reshape(-1, 1)

        X_aug = np.hstack([X_arr, w_arr])
        self.model = clone(self.base_estimator)
        self.model.fit(X_aug, y_arr)
        return self

    def predict_uplift(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Uplift as difference of predictions with W=1 vs W=0."""
        if self.model is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        X_arr = np.asarray(X, dtype=float)
        n = X_arr.shape[0]
        ones = np.ones((n, 1), dtype=float)
        zeros = np.zeros((n, 1), dtype=float)

        pred_t = _predict_outcome(self.model, np.hstack([X_arr, ones]))
        pred_c = _predict_outcome(self.model, np.hstack([X_arr, zeros]))
        return pred_t - pred_c


def _is_classifier(estimator: BaseEstimator) -> bool:
    """Heuristic: classifiers expose predict_proba or declare classifier type."""
    if hasattr(estimator, "_estimator_type"):
        return estimator._estimator_type == "classifier"
    return hasattr(estimator, "predict_proba")


def _make_cate_regressor(base_estimator: BaseEstimator) -> BaseEstimator:
    """Stage-2 CATE models need continuous targets (imputed TE).

    If the injected base is already a regressor, clone it; otherwise fall back
    to LinearRegression so classifiers can still be used for stage-1 outcomes.
    """
    from sklearn.linear_model import LinearRegression

    if not _is_classifier(base_estimator):
        return clone(base_estimator)
    return LinearRegression()


class XLearner(BaseUpliftModel):
    """X-Learner for heterogeneous treatment effect estimation.

    Stage 1: fit outcome models μ̂₀, μ̂₁ on control / treatment.
    Stage 2: impute treatment effects and fit τ̂₀, τ̂₁ on the opposite groups.
    Stage 3: propensity-weighted blend:
    τ̂(x) = ĝ(x)·τ̂₀(x) + (1 − ĝ(x))·τ̂₁(x)
    where ĝ is the propensity score P(W=1|X).

    Note: imputed treatment effects are continuous, so stage-2 CATE models
    always use a regressor (``cate_estimator`` or LinearRegression when the
    base estimator is a classifier).
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        propensity_estimator: BaseEstimator | None = None,
        cate_estimator: BaseEstimator | None = None,
    ) -> None:
        self.base_estimator = base_estimator
        self.propensity_estimator = propensity_estimator
        self.cate_estimator = cate_estimator
        self.model_control: BaseEstimator | None = None
        self.model_treatment: BaseEstimator | None = None
        self.tau_control: BaseEstimator | None = None
        self.tau_treatment: BaseEstimator | None = None
        self.propensity_model: BaseEstimator | None = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        treatment: np.ndarray,
    ) -> XLearner:
        """Two-stage imputed-TE fit with propensity-weighted blending."""
        from sklearn.linear_model import LogisticRegression

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        w_arr = np.asarray(treatment)

        mask_c = w_arr == 0
        mask_t = w_arr == 1

        # Stage 1: outcome models
        self.model_control = clone(self.base_estimator)
        self.model_treatment = clone(self.base_estimator)
        self.model_control.fit(X_arr[mask_c], y_arr[mask_c])
        self.model_treatment.fit(X_arr[mask_t], y_arr[mask_t])

        # Stage 2: imputed treatment effects
        # D¹_i = Y_i(1) − μ̂₀(X_i) for treated
        # D⁰_i = μ̂₁(X_i) − Y_i(0) for control
        mu0_on_t = _predict_outcome(self.model_control, X_arr[mask_t])
        mu1_on_c = _predict_outcome(self.model_treatment, X_arr[mask_c])

        d_t = y_arr[mask_t] - mu0_on_t
        d_c = mu1_on_c - y_arr[mask_c]

        if self.cate_estimator is not None:
            tau_proto = self.cate_estimator
            self.tau_treatment = clone(tau_proto)
            self.tau_control = clone(tau_proto)
        else:
            self.tau_treatment = _make_cate_regressor(self.base_estimator)
            self.tau_control = _make_cate_regressor(self.base_estimator)

        self.tau_treatment.fit(X_arr[mask_t], d_t)
        self.tau_control.fit(X_arr[mask_c], d_c)

        # Propensity model
        prop_est = self.propensity_estimator
        if prop_est is None:
            prop_est = LogisticRegression(max_iter=1000)
        self.propensity_model = clone(prop_est)
        self.propensity_model.fit(X_arr, w_arr)

        return self

    def predict_uplift(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Propensity-weighted blend of control and treatment CATE models."""
        if (
            self.tau_control is None
            or self.tau_treatment is None
            or self.propensity_model is None
        ):
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        X_arr = np.asarray(X, dtype=float)

        # Effect models are trained on continuous imputed TE → use predict
        tau0 = np.asarray(self.tau_control.predict(X_arr), dtype=float).ravel()
        tau1 = np.asarray(self.tau_treatment.predict(X_arr), dtype=float).ravel()

        if hasattr(self.propensity_model, "predict_proba"):
            g = self.propensity_model.predict_proba(X_arr)[:, 1]
        else:
            g = np.asarray(self.propensity_model.predict(X_arr), dtype=float).ravel()
            g = np.clip(g, 0.0, 1.0)

        # τ̂(x) = ĝ(x)·τ̂₀(x) + (1 − ĝ(x))·τ̂₁(x)
        return g * tau0 + (1.0 - g) * tau1
