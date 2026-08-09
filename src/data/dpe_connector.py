"""Connector for reading and transforming Dynamic Pricing Engine (DPE) simulation data."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

DPE_FEATURE_COLUMNS: list[str] = [
    "distance_km",
    "duration_sec",
    "price",
    "surge_bonus",
    "hour_of_day",
    "past_trips",
    "avg_surge",
]

DEFAULT_DPE_DB_PATH: Path = (
    Path("Z:/pet-project/dynamic-pricing-engine/dpe_database.db")
)


def resolve_dpe_db_path(db_path: str | Path | None = None) -> Path:
    """Resolve DPE SQLite database path following priority rules.

    Priority:
    1. Explicitly passed ``db_path`` argument (if not None).
    2. Configured path from ``load_config().dpe.db_path`` (which checks env ``CPE_DPE__DB_PATH`` and YAML).
    3. Candidate fallback locations in order of existence:
       - ``/app/data/dpe_database.db`` (Docker container volume)
       - ``<project_root>/../dynamic-pricing-engine/dpe_database.db`` (Relative workspace path)
       - ``DEFAULT_DPE_DB_PATH`` (Legacy path)
    """
    if db_path is not None:
        return Path(db_path)

    try:
        from src.config import load_config

        cfg = load_config()
        cfg_path = Path(cfg.dpe.db_path)
        if cfg_path.exists():
            return cfg_path
    except Exception:
        cfg_path = DEFAULT_DPE_DB_PATH

    project_root = Path(__file__).resolve().parent.parent.parent
    candidates = [
        Path("/app/data/dpe_database.db"),
        project_root.parent / "dynamic-pricing-engine" / "dpe_database.db",
        DEFAULT_DPE_DB_PATH,
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    return cfg_path


def load_dpe_data(
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load simulation_analytics from DPE SQLite database and transform to CPE format.

    Args:
        db_path: Optional path to dpe_database.db SQLite file.

    Returns:
        DataFrame with columns:
        ``user_id``, ``distance_km``, ``duration_sec``, ``price``, ``surge_bonus``,
        ``hour_of_day``, ``past_trips``, ``avg_surge``, ``treatment``, ``conversion``, ``revenue``.

    Raises:
        FileNotFoundError: If the specified SQLite database file does not exist.
        ValueError: If the database is missing required tables or columns.
    """
    target_path = resolve_dpe_db_path(db_path)

    if not target_path.exists():
        raise FileNotFoundError(f"DPE SQLite database not found at: {target_path}")

    conn = sqlite3.connect(target_path)
    try:
        query = """
            SELECT 
                id,
                timestamp,
                test_group,
                trip_id,
                distance_km,
                duration_sec,
                price,
                surge_bonus,
                accepted,
                driver_utility,
                driver_id
            FROM simulation_analytics
            ORDER BY timestamp ASC, id ASC
        """
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    if df.empty:
        raise ValueError(f"Table 'simulation_analytics' in {target_path} is empty.")

    # Sort to ensure strictly temporal cumulative stats per driver
    df = df.sort_values(by=["timestamp", "id"]).reset_index(drop=True)

    # Transform targets and treatment
    # Treatment: ADDITIVE -> 1, MULTIPLICATIVE -> 0
    df["treatment"] = (df["test_group"] == "ADDITIVE").astype(int)
    df["conversion"] = df["accepted"].astype(int)
    df["revenue"] = df["driver_utility"].astype(float)
    df["user_id"] = df["driver_id"].astype(str)

    # Derived feature: hour_of_day from timestamp
    hours = pd.to_datetime(df["timestamp"], unit="s", errors="coerce").dt.hour
    df["hour_of_day"] = hours.fillna(12.0).astype(float)

    # Derived driver history features (cumulative count & mean surge up to prior trip)
    df["past_trips"] = df.groupby("driver_id").cumcount().astype(float)

    # Cumulative average surge bonus for this driver (expanding mean shifted by 1)
    df["avg_surge"] = (
        df.groupby("driver_id")["surge_bonus"]
        .transform(lambda s: s.shift(1).expanding().mean())
        .fillna(0.0)
        .astype(float)
    )

    # Ensure required feature columns exist and are clean floats
    for col in DPE_FEATURE_COLUMNS:
        df[col] = df[col].astype(float)

    # Final column ordering
    output_cols = (
        ["user_id"]
        + DPE_FEATURE_COLUMNS
        + ["treatment", "conversion", "revenue"]
    )
    return df[output_cols]
