"""Spatiotemporal switchback assignment for marketplace A/B tests."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import md5
from typing import Literal

Assignment = Literal["control", "treatment"]


class SwitchbackSplitter:
    """Assign units to control/treatment via space–time blocks.

    Each observation is mapped to a block key
    ``(grid_cell, time_bucket)``. The block is hashed to a stable
    control/treatment label so that nearby units in the same time window
    share the same arm (reducing interference / spillover).

    Args:
        time_bucket_minutes: Width of temporal buckets in minutes.
        n_grid_cells: Number of spatial cells when deriving cell from unit_id
            (used when ``grid_cell`` is not passed explicitly).
        treatment_ratio: Target fraction of blocks assigned to treatment.
        salt: Salt string for stable hashing.
    """

    def __init__(
        self,
        time_bucket_minutes: int = 60,
        n_grid_cells: int = 16,
        treatment_ratio: float = 0.5,
        salt: str = "switchback",
    ) -> None:
        if time_bucket_minutes <= 0:
            raise ValueError("time_bucket_minutes must be positive")
        if n_grid_cells <= 0:
            raise ValueError("n_grid_cells must be positive")
        if not 0.0 < treatment_ratio < 1.0:
            raise ValueError("treatment_ratio must be in (0, 1)")

        self.time_bucket_minutes = time_bucket_minutes
        self.n_grid_cells = n_grid_cells
        self.treatment_ratio = treatment_ratio
        self.salt = salt

    def grid_cell_from_unit(self, unit_id: str | int) -> int:
        """Deterministically map a unit id to a spatial grid cell."""
        digest = md5(f"{self.salt}:cell:{unit_id}".encode()).hexdigest()
        return int(digest[:8], 16) % self.n_grid_cells

    def time_bucket(self, timestamp: datetime | float | int | str) -> int:
        """Map a timestamp to a discrete time bucket index."""
        ts = _to_unix(timestamp)
        bucket_seconds = self.time_bucket_minutes * 60
        return int(ts // bucket_seconds)

    def assign(
        self,
        unit_id: str | int,
        timestamp: datetime | float | int | str,
        grid_cell: int | None = None,
    ) -> Assignment:
        """Assign control or treatment for (unit_id, timestamp).

        Args:
            unit_id: Spatial / marketplace unit identifier.
            timestamp: Event time (datetime, unix seconds, or ISO string).
            grid_cell: Optional explicit spatial cell; if omitted, derived
                from ``unit_id``.

        Returns:
            ``"control"`` or ``"treatment"``.
        """
        cell = grid_cell if grid_cell is not None else self.grid_cell_from_unit(unit_id)
        bucket = self.time_bucket(timestamp)
        block_key = f"{self.salt}:{cell}:{bucket}"
        digest = md5(block_key.encode()).hexdigest()
        score = int(digest[:8], 16) / 0xFFFFFFFF
        return "treatment" if score < self.treatment_ratio else "control"


def _to_unix(timestamp: datetime | float | int | str) -> float:
    """Normalize various timestamp types to unix seconds."""
    if isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.timestamp()
    if isinstance(timestamp, (int, float)):
        return float(timestamp)
    if isinstance(timestamp, str):
        # Support ISO-8601 and pure numeric strings
        try:
            return float(timestamp)
        except ValueError:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
    raise TypeError(f"Unsupported timestamp type: {type(timestamp)!r}")
