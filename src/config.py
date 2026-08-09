"""Configuration loader for Causal Pricing Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Uplift model hyperparameters."""

    base_learner: str = "logistic_regression"
    n_estimators: int = 100
    learning_rate: float = 0.1
    random_state: int = 42


class ExperimentConfig(BaseModel):
    """Statistical experiment parameters."""

    alpha: float = 0.05
    power: float = 0.80
    mde: float = 0.02


class ApiConfig(BaseModel):
    """FastAPI service settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "artifacts/model.joblib"
    uplift_threshold: float = 0.05


class DataConfig(BaseModel):
    """Synthetic / training data defaults."""

    n_samples: int = 5000
    random_state: int = 42


class AppConfig(BaseModel):
    """Root application configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    data: DataConfig = Field(default_factory=DataConfig)


def _default_config_path() -> Path:
    """Resolve default configs/config.yaml relative to the project root."""
    return Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load application config from a YAML file.

    Args:
        path: Optional path to YAML. Defaults to ``configs/config.yaml``
            under the project root.

    Returns:
        Validated ``AppConfig`` instance.
    """
    config_path = Path(path) if path is not None else _default_config_path()
    if not config_path.exists():
        return AppConfig()

    with config_path.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    return AppConfig.model_validate(raw)
