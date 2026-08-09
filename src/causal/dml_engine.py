"""Double Machine Learning engine (EconML wrapper with sklearn fallback)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

_ECONML_AVAILABLE = False
try:
    from econml.dml import LinearDML  # type: ignore[import-untyped]

    _ECONML_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when econml missing
    LinearDML = None  # type: ignore[misc, assignment]


class DMLEngine:
    """Double ML wrapper for conditional average treatment effect estimation.

    Prefers EconML ``LinearDML`` when installed; otherwise uses a residual-on-
    residual (partialling-out) estimator with sklearn models that exposes the
    same ``fit(Y, T, X, W)`` / ``effect(X)`` interface.
    """

    def __init__(
        self,
        model_y: BaseEstimator | None = None,
        model_t: BaseEstimator | None = None,
        model_final: BaseEstimator | None = None,
        random_state: int = 42,
    ) -> None:
        self.model_y = model_y
        self.model_t = model_t
        self.model_final = model_final
        self.random_state = random_state
        self._estimator: object | None = None
        self._fallback_cate: BaseEstimator | None = None
        self._uses_econml: bool = False

    def fit(
        self,
        Y: np.ndarray | pd.Series,
        T: np.ndarray | pd.Series,
        X: np.ndarray | pd.DataFrame | None = None,
        W: np.ndarray | pd.DataFrame | None = None,
    ) -> DMLEngine:
        """Fit DML: residualize Y and T on controls, then regress residuals.

        Args:
            Y: Outcome vector.
            T: Treatment vector (binary or continuous).
            X: Features for heterogeneous effects (CATE).
            W: Additional controls for nuisance models (optional).

        Returns:
            self
        """
        Y_arr = np.asarray(Y, dtype=float).ravel()
        T_arr = np.asarray(T, dtype=float).ravel()
        X_arr = np.asarray(X, dtype=float) if X is not None else None
        W_arr = np.asarray(W, dtype=float) if W is not None else None

        if X_arr is None:
            raise ValueError("X must be provided for CATE estimation.")

        if _ECONML_AVAILABLE and LinearDML is not None:
            # Use `is not None` — unfitted sklearn estimators are not safely truthy
            model_y = (
                self.model_y
                if self.model_y is not None
                else GradientBoostingRegressor(random_state=self.random_state)
            )
            model_t = (
                self.model_t
                if self.model_t is not None
                else GradientBoostingRegressor(random_state=self.random_state)
            )
            est = LinearDML(
                model_y=model_y,
                model_t=model_t,
                random_state=self.random_state,
            )
            est.fit(Y_arr, T_arr, X=X_arr, W=W_arr)
            self._estimator = est
            self._uses_econml = True
            return self

        # Fallback: residual-on-residual DML-style estimator
        controls = X_arr if W_arr is None else np.hstack([X_arr, W_arr])

        model_y = (
            clone(self.model_y)
            if self.model_y is not None
            else GradientBoostingRegressor(random_state=self.random_state)
        )
        if self.model_t is not None:
            model_t = clone(self.model_t)
        else:
            unique_t = np.unique(T_arr)
            if len(unique_t) <= 2:
                model_t = LogisticRegression(max_iter=1000, random_state=self.random_state)
            else:
                model_t = GradientBoostingRegressor(random_state=self.random_state)

        model_y.fit(controls, Y_arr)
        model_t.fit(controls, T_arr)

        y_hat = _nuisance_predict(model_y, controls)
        t_hat = _nuisance_predict(model_t, controls)
        y_res = Y_arr - y_hat
        t_res = T_arr - t_hat

        # Classic partially linear: y_res = θ(X) * t_res + ε
        # Fit via interaction features [X * t_res, t_res]
        interaction = X_arr * t_res.reshape(-1, 1)
        final_X = np.hstack([interaction, t_res.reshape(-1, 1)])

        final = (
            clone(self.model_final)
            if self.model_final is not None
            else LinearRegression()
        )
        final.fit(final_X, y_res)
        self._fallback_cate = final
        self._estimator = ("fallback", model_y, model_t, final)
        self._uses_econml = False
        return self

    def effect(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Estimate CATE τ̂(x) for each row of X.

        Args:
            X: Feature matrix.

        Returns:
            1d array of treatment effect estimates.
        """
        if self._estimator is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        X_arr = np.asarray(X, dtype=float)

        if self._uses_econml:
            effects = self._estimator.effect(X_arr)  # type: ignore[union-attr]
            return np.asarray(effects, dtype=float).ravel()

        # Fallback: CATE from interaction coefficients applied at t_res=1 scale
        # final_X for effect at unit treatment residual = [X * 1, 1]
        final = self._fallback_cate
        assert final is not None
        ones = np.ones((X_arr.shape[0], 1), dtype=float)
        final_X = np.hstack([X_arr, ones])
        return np.asarray(final.predict(final_X), dtype=float).ravel()


def _nuisance_predict(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """Predict continuous nuisance residual targets."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()
    return np.asarray(model.predict(X), dtype=float).ravel()
