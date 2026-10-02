"""Tests for the DPE switchback log loader."""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.dpe_connector import (
    POST_TREATMENT_COLUMNS,
    default_dpe_db_path,
    describe_switchback,
    load_dpe_data,
    resolve_dpe_db_path,
)


def create_dummy_dpe_db(db_path: Path, n_rows: int = 50, virtual_hour: bool = False) -> Path:
    """SQLite fixture matching the current simulation_analytics columns."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    hour_sql = ", virtual_hour REAL" if virtual_hour else ""
    cursor.execute(f"""
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
            {hour_sql}
        )
    """)
    rng = np.random.default_rng(42)
    for i in range(n_rows):
        # Wall-clock timestamps an hour apart. This must not become the arm clock.
        timestamp = 1700000000.0 + i * 3600
        arm = "ADDITIVE" if i % 2 == 0 else "MULTIPLICATIVE"
        columns = (
            "timestamp, test_group, trip_id, distance_km, duration_sec, "
            "price, surge_bonus, accepted, driver_utility, driver_id"
        )
        values = [
            timestamp,
            arm,
            f"trip_{i}",
            float(rng.uniform(1.0, 20.0)),
            float(rng.uniform(300.0, 1800.0)),
            float(rng.uniform(100.0, 500.0)),
            0.0 if arm == "MULTIPLICATIVE" else float(rng.uniform(10.0, 40.0)),
            int(rng.choice([0, 1])),
            float(rng.uniform(0.05, 0.95)),
            f"drv_{i % 5}",
        ]
        if virtual_hour:
            columns += ", virtual_hour"
            values.append(float(7 + i))
        placeholders = ", ".join("?" for _ in values)
        cursor.execute(
            f"INSERT INTO simulation_analytics ({columns}) VALUES ({placeholders})",
            values,
        )
    conn.commit()
    conn.close()
    return db_path


def test_load_dpe_data_does_not_invent_an_ite_sample(tmp_path: Path):
    db_path = tmp_path / "test_dpe.db"
    create_dummy_dpe_db(db_path, n_rows=30)

    df = load_dpe_data(db_path=db_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 30
    assert "driver_id" in df.columns
    assert "pricing_arm" in df.columns
    assert "accepted" in df.columns
    for banned in ("user_id", "treatment", "conversion", "revenue", "hour_of_day", "price", "surge_bonus"):
        assert banned not in df.columns
    assert df["virtual_hour"].isna().all()
    assert set(df["pricing_arm"].unique()) == {"ADDITIVE", "MULTIPLICATIVE"}
    assert set(df["accepted"].unique()).issubset({0, 1})
    assert (df["past_trips"] >= 0).all()

    design = describe_switchback(df)
    assert design["identified_ite"] is False
    assert design["arm_clock_recorded"] is False


def test_pre_treatment_is_the_default_and_serve_parity_is_explicit(tmp_path: Path):
    db_path = tmp_path / "test_dpe_modes.db"
    create_dummy_dpe_db(db_path, n_rows=20)

    default = load_dpe_data(db_path=db_path)
    for col in POST_TREATMENT_COLUMNS:
        assert col not in default.columns

    served = load_dpe_data(db_path=db_path, feature_mode="serve_parity")
    for col in POST_TREATMENT_COLUMNS:
        assert col in served.columns
    assert "hour_of_day" not in served.columns

    with pytest.raises(ValueError, match="Invalid feature_mode"):
        load_dpe_data(db_path=db_path, feature_mode="invalid_mode")


def test_virtual_hour_is_kept_and_timestamp_is_not_the_arm_clock(tmp_path: Path):
    db_path = tmp_path / "with_hour.db"
    create_dummy_dpe_db(db_path, n_rows=8, virtual_hour=True)
    df = load_dpe_data(db_path=db_path)
    assert df["virtual_hour"].notna().all()
    assert float(df["virtual_hour"].iloc[0]) == pytest.approx(7.0)
    # Timestamps are an hour of wall time apart, so a derived clock hour would
    # not equal the stored virtual hour on every row.
    wall_hour = pd.to_datetime(df["timestamp"], unit="s").dt.hour
    assert not np.allclose(wall_hour.to_numpy(), df["virtual_hour"].to_numpy())
    assert describe_switchback(df)["arm_clock_recorded"] is True


def test_resolve_dpe_db_path_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dummy_db = tmp_path / "custom_env_dpe.db"
    create_dummy_dpe_db(dummy_db, n_rows=5)

    monkeypatch.setenv("CPE_DPE__DB_PATH", str(dummy_db))
    resolved = resolve_dpe_db_path(db_path=None)
    assert resolved == dummy_db

    df = load_dpe_data()
    assert len(df) == 5


def test_default_path_is_the_sibling_checkout():
    path = default_dpe_db_path()
    assert path.name == "dpe_database.db"
    assert path.parent.name == "dynamic-pricing-engine"
    connector = Path(__file__).resolve().parents[1] / "src" / "data" / "dpe_connector.py"
    config = Path(__file__).resolve().parents[1] / "src" / "config.py"
    yaml = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    for file in (connector, config, yaml):
        text = file.read_text(encoding="utf-8")
        assert "Z:/pet-project" not in text
        assert "Z:\\pet-project" not in text


def test_load_dpe_data_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_dpe_data(db_path=tmp_path / "non_existent.db")


def test_history_matches_cpe_connector_parity(tmp_path: Path):
    """Prior trip count and mean surge match DriverHistoryStore."""
    import importlib.util

    dpe_file = (
        Path(__file__).resolve().parent.parent.parent
        / "dynamic-pricing-engine"
        / "src"
        / "features"
        / "driver_history.py"
    )
    spec = importlib.util.spec_from_file_location("dpe_driver_history", dpe_file)
    dpe_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dpe_mod)
    DriverHistoryStore = dpe_mod.DriverHistoryStore

    db_path = tmp_path / "parity_test.db"
    create_dummy_dpe_db(db_path, n_rows=50)
    df_offline = load_dpe_data(db_path=db_path, feature_mode="serve_parity")
    store = DriverHistoryStore()

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT driver_id, surge_bonus FROM simulation_analytics ORDER BY timestamp ASC, id ASC"
    ).fetchall()
    conn.close()

    assert len(rows) == len(df_offline)
    for i, (driver_id, surge_bonus) in enumerate(rows):
        p_trips, a_surge = store.get_driver_features(driver_id=str(driver_id))
        assert p_trips == df_offline.iloc[i]["past_trips"]
        assert abs(a_surge - df_offline.iloc[i]["avg_surge"]) < 1e-6
        store.record_trip(driver_id=str(driver_id), surge_bonus=surge_bonus)
