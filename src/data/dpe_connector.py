"""Read DPE ``simulation_analytics`` as a switchback log, not as an ITE sample.

Current DPE table (after the fare refactor):

``id, timestamp, test_group, trip_id, distance_km, duration_sec, price,
surge_bonus, accepted, driver_utility, driver_id``

``virtual_hour`` is absent from that schema. New simulator writes add it.
``timestamp`` is ``clock.now()``: wall time, or the frozen eval clock that
advances a few seconds per tick. It is not the virtual hour whose parity
picks ADDITIVE versus MULTIPLICATIVE.

The arm is constant inside a virtual hour, so the driver is not randomized
into a pricing arm. ``accepted`` is the driver taking the order.
``driver_id`` is that driver. ``driver_utility`` is the planted acceptance
probability, not revenue.

CPE's served model estimates a different contrast: additive surcharge versus
the base fare, randomized per offer in ``synthetic.py``. This loader does not
rename the switchback arm to that treatment.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

# Quote fields that exist only after the arm has set the price.
POST_TREATMENT_COLUMNS: list[str] = ["price", "surge_bonus"]

# History and trip facts that do not include the current quote.
# ``hour_of_day`` is intentionally absent: the arm clock is ``virtual_hour``
# when the file has it, and it is not invented from ``timestamp``.
LOG_FEATURE_COLUMNS: list[str] = [
    "distance_km",
    "duration_sec",
    "past_trips",
    "avg_surge",
]

# Kept so older call sites can still ask for the post-treatment columns by name.
DPE_FEATURE_COLUMNS: list[str] = LOG_FEATURE_COLUMNS + POST_TREATMENT_COLUMNS
DPE_PRE_TREATMENT_FEATURES: list[str] = list(LOG_FEATURE_COLUMNS)

REQUIRED_COLUMNS = {
    "timestamp",
    "test_group",
    "trip_id",
    "distance_km",
    "duration_sec",
    "price",
    "surge_bonus",
    "accepted",
    "driver_id",
}

DESIGN_NOTE = (
    "DPE assigns the pricing arm by the parity of the virtual hour, not by driver. "
    "Trips in the same hour share an arm, so a row-level T-learner estimates a "
    "difference between hours, not an effect for one driver. "
    "timestamp is the process clock, not that virtual hour. "
    "accepted is whether the driver took the order, and driver_id is the driver. "
    "The served CPE model estimates a different contrast: an additive surcharge "
    "versus the base fare, randomized on synthetic offers. These logs do not "
    "contain that contrast, and no individual effect is fit on them."
)


def default_dpe_db_path() -> Path:
    """Sibling checkout of Dynamic-pricing-engine, not a drive letter."""
    project_root = Path(__file__).resolve().parents[2]
    return project_root.parent / "dynamic-pricing-engine" / "dpe_database.db"


def resolve_dpe_db_path(db_path: str | Path | None = None) -> Path:
    """Resolve the SQLite file.

    An explicit argument wins. Otherwise a non-empty configured path
    (``CPE_DPE__DB_PATH`` or YAML) wins when the file exists. Then
    ``/app/data/dpe_database.db``, then the sibling checkout. A blank config
    is not treated as the current directory.
    """
    if db_path is not None and str(db_path).strip():
        return Path(db_path)

    configured = ""
    try:
        from src.config import load_config

        configured = (load_config().dpe.db_path or "").strip()
    except Exception:
        configured = ""

    if configured:
        cfg_path = Path(configured)
        if cfg_path.exists():
            return cfg_path

    candidates = [
        Path("/app/data/dpe_database.db"),
        default_dpe_db_path(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if configured:
        return Path(configured)
    return default_dpe_db_path()


def _table_columns(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("PRAGMA table_info(simulation_analytics)").fetchall()
    if not rows:
        raise ValueError("Table 'simulation_analytics' was not found.")
    return {row[1] for row in rows}


def load_dpe_data(
    db_path: str | Path | None = None,
    feature_mode: str = "pre_treatment",
) -> pd.DataFrame:
    """Load the switchback log.

    ``feature_mode='pre_treatment'`` (default) leaves out ``price`` and
    ``surge_bonus``. ``serve_parity`` adds those quote fields for inspection.
    Neither mode builds a treatment column for the surcharge model, and neither
    mode derives an hour feature from ``timestamp``.
    """
    if feature_mode not in {"serve_parity", "pre_treatment"}:
        raise ValueError(
            f"Invalid feature_mode '{feature_mode}'. "
            "Must be 'serve_parity' or 'pre_treatment'."
        )

    target_path = resolve_dpe_db_path(db_path)
    if not target_path.exists():
        raise FileNotFoundError(f"DPE SQLite database not found at: {target_path}")

    conn = sqlite3.connect(target_path)
    try:
        present = _table_columns(conn)
        missing = REQUIRED_COLUMNS - present
        if missing:
            raise ValueError(
                "simulation_analytics is missing "
                f"{sorted(missing)}. The current DPE schema is timestamp, "
                "test_group, trip_id, distance_km, duration_sec, price, "
                "surge_bonus, accepted, driver_utility, driver_id, "
                "and virtual_hour when the simulator has written it."
            )
        optional = [
            name
            for name in ("id", "driver_utility", "virtual_hour")
            if name in present
        ]
        selected = [
            "timestamp",
            "test_group",
            "trip_id",
            "distance_km",
            "duration_sec",
            "price",
            "surge_bonus",
            "accepted",
            "driver_id",
            *optional,
        ]
        order = "timestamp ASC, id ASC" if "id" in present else "timestamp ASC"
        frame = pd.read_sql_query(
            f"SELECT {', '.join(selected)} FROM simulation_analytics ORDER BY {order}",
            conn,
        )
    finally:
        conn.close()

    if frame.empty:
        raise ValueError(f"Table 'simulation_analytics' in {target_path} is empty.")

    sort_cols = ["timestamp", "id"] if "id" in frame.columns else ["timestamp"]
    frame = frame.sort_values(by=sort_cols).reset_index(drop=True)

    frame["driver_id"] = frame["driver_id"].astype(str)
    frame["pricing_arm"] = frame["test_group"].astype(str)
    frame["accepted"] = frame["accepted"].astype(int)
    if "driver_utility" in frame.columns:
        frame["accept_probability"] = frame["driver_utility"].astype(float)
    if "virtual_hour" not in frame.columns:
        frame["virtual_hour"] = float("nan")
    else:
        frame["virtual_hour"] = pd.to_numeric(frame["virtual_hour"], errors="coerce")

    frame["past_trips"] = frame.groupby("driver_id").cumcount().astype(float)
    frame["avg_surge"] = (
        frame.groupby("driver_id")["surge_bonus"]
        .transform(lambda series: series.shift(1).expanding().mean())
        .fillna(0.0)
        .astype(float)
    )

    keep = [
        "driver_id",
        "trip_id",
        "pricing_arm",
        "accepted",
        "timestamp",
        "virtual_hour",
        *LOG_FEATURE_COLUMNS,
    ]
    if "accept_probability" in frame.columns:
        keep.append("accept_probability")
    if feature_mode == "serve_parity":
        keep.extend(POST_TREATMENT_COLUMNS)
    return frame[keep]


def describe_switchback(frame: pd.DataFrame) -> dict:
    """Describe the arm assignment. This does not estimate an individual effect."""
    arms = frame["pricing_arm"].value_counts()
    clock = bool(frame["virtual_hour"].notna().any())
    lines = [
        "## DPE switchback log",
        "",
        DESIGN_NOTE,
        "",
        "### Arms",
        "",
    ]
    arm_counts: dict[str, int] = {}
    for arm, count in arms.items():
        arm_name = str(arm)
        arm_counts[arm_name] = int(count)
        accept = float(frame.loc[frame["pricing_arm"] == arm, "accepted"].mean())
        lines.append(f"- `{arm_name}`: n={int(count)}, mean accepted={accept:.3f}")
    lines.append("")
    if not clock:
        lines.append(
            "virtual_hour is not in this file, so the arm clock cannot be "
            "reconstructed. hour_of_day is not filled from timestamp."
        )
    else:
        known = frame.dropna(subset=["virtual_hour"]).copy()
        known["hour_index"] = known["virtual_hour"].astype(int)
        lines.append(
            "virtual_hour is stored. Acceptance by that hour is a contrast "
            "between hours, not an effect for one driver."
        )
        switch = known[known["pricing_arm"].isin(["ADDITIVE", "MULTIPLICATIVE"])]
        if not switch.empty:
            expected = np_where_arm(switch["hour_index"].to_numpy())
            agree = float((switch["pricing_arm"].to_numpy() == expected).mean())
            lines.append(
                f"Among ADDITIVE and MULTIPLICATIVE rows, the stored arm matches "
                f"odd-hour = ADDITIVE on {agree:.1%} of rows."
            )
    lines.append("")
    lines.append("No individual treatment effect is estimated from this table.")
    return {
        "arm_counts": arm_counts,
        "arm_clock_recorded": clock,
        "n": int(len(frame)),
        "identified_ite": False,
        "report": "\n".join(lines),
    }


def np_where_arm(hour_index) -> list[str]:
    """Odd virtual hour is the additive arm in the current DPE switchback."""
    return ["ADDITIVE" if int(hour) % 2 == 1 else "MULTIPLICATIVE" for hour in hour_index]
