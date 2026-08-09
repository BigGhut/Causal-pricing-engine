"""Tests for DPE database connector."""

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.dpe_connector import (
    DEFAULT_DPE_DB_PATH,
    DPE_FEATURE_COLUMNS,
    load_dpe_data,
    resolve_dpe_db_path,
)


def create_dummy_dpe_db(db_path: Path, n_rows: int = 50) -> Path:
    """Helper to create a temporary SQLite database with simulation_analytics table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE simulation_analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            test_group TEXT,
            trip_id TEXT,
            distance_km REAL,
            duration_sec REAL,
            price REAL,
            surge_bonus REAL,
            accepted INTEGER,
            driver_utility REAL,
            driver_id TEXT
        )
    """)
    rng = np.random.default_rng(42)
    for i in range(n_rows):
        cursor.execute("""
            INSERT INTO simulation_analytics (
                timestamp, test_group, trip_id, distance_km, duration_sec,
                price, surge_bonus, accepted, driver_utility, driver_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            1700000000.0 + i * 100,
            "ADDITIVE" if i % 2 == 0 else "CONTROL",
            f"trip_{i}",
            float(rng.uniform(1.0, 20.0)),
            float(rng.uniform(300.0, 1800.0)),
            float(rng.uniform(10.0, 50.0)),
            float(rng.uniform(0.0, 5.0)),
            int(rng.choice([0, 1])),
            float(rng.uniform(5.0, 30.0)),
            f"drv_{i % 5}"
        ))
    conn.commit()
    conn.close()
    return db_path


def test_load_dpe_data_fixture(tmp_path: Path):
    """Test loading DPE data using an isolated SQLite fixture (no skip)."""
    db_path = tmp_path / "test_dpe.db"
    create_dummy_dpe_db(db_path, n_rows=30)

    df = load_dpe_data(db_path=db_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 30

    expected_cols = ["user_id"] + DPE_FEATURE_COLUMNS + ["treatment", "conversion", "revenue"]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"

    assert set(df["treatment"].unique()).issubset({0, 1})
    assert set(df["conversion"].unique()).issubset({0, 1})
    assert np.isfinite(df["revenue"]).all()
    assert (df["past_trips"] >= 0).all()
    assert (df["avg_surge"] >= 0).all()


def test_resolve_dpe_db_path_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that CPE_DPE__DB_PATH env var overrides resolution when file exists."""
    dummy_db = tmp_path / "custom_env_dpe.db"
    create_dummy_dpe_db(dummy_db, n_rows=5)

    monkeypatch.setenv("CPE_DPE__DB_PATH", str(dummy_db))
    resolved = resolve_dpe_db_path(db_path=None)
    assert resolved == dummy_db

    df = load_dpe_data()
    assert len(df) == 5


def test_load_dpe_data_real_db():
    if not DEFAULT_DPE_DB_PATH.exists():
        pytest.skip(f"DPE DB does not exist at {DEFAULT_DPE_DB_PATH}")

    df = load_dpe_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    expected_cols = ["user_id"] + DPE_FEATURE_COLUMNS + ["treatment", "conversion", "revenue"]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"

    assert set(df["treatment"].unique()).issubset({0, 1})
    assert set(df["conversion"].unique()).issubset({0, 1})
    assert np.isfinite(df["revenue"]).all()
    assert (df["past_trips"] >= 0).all()


def test_load_dpe_data_file_not_found(tmp_path: Path):
    non_existent = tmp_path / "non_existent.db"
    with pytest.raises(FileNotFoundError):
        load_dpe_data(db_path=non_existent)


def test_history_matches_cpe_connector_parity(tmp_path: Path):
    """Verify online DriverHistoryStore replay matches offline load_dpe_data past_trips & avg_surge."""
    import importlib.util

    dpe_file = Path(__file__).resolve().parent.parent.parent / "dynamic-pricing-engine" / "src" / "features" / "driver_history.py"
    spec = importlib.util.spec_from_file_location("dpe_driver_history", dpe_file)
    dpe_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dpe_mod)
    DriverHistoryStore = dpe_mod.DriverHistoryStore

    db_path = tmp_path / "parity_test.db"
    create_dummy_dpe_db(db_path, n_rows=50)

    # Offline stats via connector
    df_offline = load_dpe_data(db_path=db_path)

    # Online replay: process each trip in temporal order
    store = DriverHistoryStore()

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT driver_id, surge_bonus FROM simulation_analytics ORDER BY timestamp ASC, id ASC"
    ).fetchall()
    conn.close()

    assert len(rows) == len(df_offline)

    for i, (driver_id, surge_bonus) in enumerate(rows):
        p_trips, a_surge = store.get_driver_features(driver_id=str(driver_id))

        expected_past_trips = df_offline.iloc[i]["past_trips"]
        expected_avg_surge = df_offline.iloc[i]["avg_surge"]

        assert p_trips == expected_past_trips, f"Row {i} past_trips mismatch: {p_trips} != {expected_past_trips}"
        assert abs(a_surge - expected_avg_surge) < 1e-6, f"Row {i} avg_surge mismatch: {a_surge} != {expected_avg_surge}"

        store.record_trip(driver_id=str(driver_id), surge_bonus=surge_bonus)


