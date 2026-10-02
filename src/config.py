"""Configuration loader for Causal Pricing Engine.

Uses **pydantic-settings** ``BaseSettings`` so values can be overridden via
environment variables after YAML defaults are applied.

Environment variables
---------------------
Prefix: ``CPE_``
Nested delimiter: ``__``

Examples (override YAML / field defaults)::

    CPE_API__HOST=127.0.0.1
    CPE_API__PORT=8100
    CPE_MODEL__BASE_LEARNER=gradient_boosting
    CPE_MODEL__N_ESTIMATORS=50
    CPE_MODEL__LEARNING_RATE=0.05
    CPE_EXPERIMENT__ALPHA=0.01
    CPE_EXPERIMENT__POWER=0.9
    CPE_EXPERIMENT__MDE=0.01
    CPE_DATA__N_SAMPLES=1000

YAML still supplies file-based defaults from ``configs/config.yaml``.
When both YAML and env set the same field, **env wins**.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict
from pydantic_settings.sources import YamlConfigSettingsSource


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
    port: int = 8100
    model_path: str = "artifacts/model.joblib"
    uplift_threshold: float = 0.05


class DataConfig(BaseModel):
    """Synthetic / training data defaults."""

    n_samples: int = 5000
    random_state: int = 42


class DpeConfig(BaseModel):
    """DPE connector and database settings.

    Empty means "resolve a sibling checkout or ``/app/data``". A drive letter
    is not a default. Override with ``CPE_DPE__DB_PATH`` when the file lives
    somewhere else.
    """

    db_path: str = ""


class AppConfig(BaseSettings):
    """Root application configuration (pydantic-settings ``BaseSettings``).

    Nested sections are overridable via env vars with prefix ``CPE_`` and
    delimiter ``__`` (e.g. ``CPE_API__PORT=8100``).
    """

    model_config = SettingsConfigDict(
        env_prefix="CPE_",
        env_nested_delimiter="__",
        extra="ignore",
        env_file=None,
    )

    model: ModelConfig = Field(default_factory=ModelConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    dpe: DpeConfig = Field(default_factory=DpeConfig)


def _default_config_path() -> Path:
    """Resolve default configs/config.yaml relative to the project root."""
    return Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load application config from YAML, then apply env overrides.

    Args:
        path: Optional path to YAML. Defaults to ``configs/config.yaml``
            under the project root.

    Returns:
        Validated ``AppConfig`` (``BaseSettings``) instance.

    Env overrides use prefix ``CPE_`` and nested delimiter ``__``
    (see module docstring). Env values take precedence over YAML.
    """
    config_path = Path(path) if path is not None else _default_config_path()

    class _LoadedConfig(AppConfig):
        """Bound to a specific YAML path for this load call."""

        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            yaml_source = YamlConfigSettingsSource(
                settings_cls,
                yaml_file=config_path if config_path.exists() else None,
            )
            return (
                init_settings,
                env_settings,
                yaml_source,
                file_secret_settings,
            )

    return _LoadedConfig()
