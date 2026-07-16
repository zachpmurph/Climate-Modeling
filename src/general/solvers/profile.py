import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from general.solvers.contract import Domain

MIN_DEPTH = 1e-10


@dataclass(frozen=True)
class RiverProfile:
    """Cell-centered river profile inputs for the 1D kinematic wave model."""

    station_m: np.ndarray
    dx_m: np.ndarray
    slope: np.ndarray
    manning_n: np.ndarray
    initial_depth_m: np.ndarray | None = None
    rainfall_rate_m_per_min: np.ndarray | None = None
    labels: tuple[str, ...] = ()

    @property
    def length_m(self):
        return float(np.sum(self.dx_m))


def _as_float(row, key, *, required=True, default=None):
    value = row.get(key, default)
    if value in (None, ""):
        if required:
            raise ValueError(f"Missing required profile field: {key}")
        return default
    return float(value)


def _cell_widths_from_stations(station_m):
    station_m = np.asarray(station_m, dtype=float)
    if station_m.ndim != 1 or len(station_m) == 0:
        raise ValueError("station_m must contain at least one station")
    if len(station_m) == 1:
        raise ValueError("At least two station_m values are required to infer cell widths")
    if np.any(np.diff(station_m) <= 0):
        raise ValueError("station_m values must be strictly increasing")

    edges = np.empty(len(station_m) + 1, dtype=float)
    edges[1:-1] = 0.5 * (station_m[:-1] + station_m[1:])
    edges[0] = station_m[0] - 0.5 * (station_m[1] - station_m[0])
    edges[-1] = station_m[-1] + 0.5 * (station_m[-1] - station_m[-2])
    return np.diff(edges)


def _optional_array(values, expected_len, name, *, minimum=None):
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if len(arr) != expected_len:
        raise ValueError(f"{name} must have one value per station")
    if minimum is not None and np.any(arr < minimum):
        raise ValueError(f"{name} values must be >= {minimum}")
    return arr


def make_profile(station_m, slope, manning_n, initial_depth_m=None, rainfall_rate_m_per_min=None, labels=None):
    station_m = np.asarray(station_m, dtype=float)
    slope = np.asarray(slope, dtype=float)
    manning_n = np.asarray(manning_n, dtype=float)

    if not (len(station_m) == len(slope) == len(manning_n)):
        raise ValueError("station_m, slope, and manning_n must have the same length")
    if np.any(slope <= 0):
        raise ValueError("slope values must be positive")
    if np.any(manning_n <= 0):
        raise ValueError("manning_n values must be positive")

    initial = _optional_array(initial_depth_m, len(station_m), "initial_depth_m", minimum=0.0)
    if initial is not None:
        initial = np.maximum(initial, MIN_DEPTH)

    rainfall = _optional_array(rainfall_rate_m_per_min, len(station_m), "rainfall_rate_m_per_min", minimum=0.0)

    if labels is None:
        labels = tuple("" for _ in station_m)
    else:
        labels = tuple(labels)
        if len(labels) != len(station_m):
            raise ValueError("labels must have one value per station")

    return RiverProfile(
        station_m=station_m,
        dx_m=_cell_widths_from_stations(station_m),
        slope=slope,
        manning_n=manning_n,
        initial_depth_m=initial,
        rainfall_rate_m_per_min=rainfall,
        labels=labels,
    )


def load_profile_csv(path):
    """Load a river profile CSV.

    Required columns: station_m, slope, manning_n.
    Optional columns: initial_depth_m, rainfall_rate_m_per_min, label.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("River profile CSV is empty")

    initial_values = [row.get("initial_depth_m", "") for row in rows]
    has_initial = any(value not in (None, "") for value in initial_values)
    rainfall_values = [row.get("rainfall_rate_m_per_min", "") for row in rows]
    has_rainfall = any(value not in (None, "") for value in rainfall_values)

    return make_profile(
        station_m=[_as_float(row, "station_m") for row in rows],
        slope=[_as_float(row, "slope") for row in rows],
        manning_n=[_as_float(row, "manning_n") for row in rows],
        initial_depth_m=[_as_float(row, "initial_depth_m", required=False, default=MIN_DEPTH) for row in rows]
        if has_initial
        else None,
        rainfall_rate_m_per_min=[_as_float(row, "rainfall_rate_m_per_min", required=False, default=0.0) for row in rows]
        if has_rainfall
        else None,
        labels=[row.get("label", "") for row in rows],
    )


def load_profile_json(path):
    """Load a river profile JSON file.

    Accepted forms are either a list of segment objects or an object with a
    ``segments`` list. Segment fields match the CSV columns.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("segments", data) if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise ValueError("River profile JSON must contain a non-empty segment list")

    has_initial = any(row.get("initial_depth_m") is not None for row in rows)
    has_rainfall = any(row.get("rainfall_rate_m_per_min") is not None for row in rows)
    return make_profile(
        station_m=[_as_float(row, "station_m") for row in rows],
        slope=[_as_float(row, "slope") for row in rows],
        manning_n=[_as_float(row, "manning_n") for row in rows],
        initial_depth_m=[_as_float(row, "initial_depth_m", required=False, default=MIN_DEPTH) for row in rows]
        if has_initial
        else None,
        rainfall_rate_m_per_min=[_as_float(row, "rainfall_rate_m_per_min", required=False, default=0.0) for row in rows]
        if has_rainfall
        else None,
        labels=[str(row.get("label", "")) for row in rows],
    )


def load_profile(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_profile_csv(path)
    if suffix == ".json":
        return load_profile_json(path)
    raise ValueError(f"Unsupported river profile format: {suffix}")


def domain_from_profile(profile: RiverProfile) -> Domain:
    """Build a Domain from a RiverProfile (uses per-cell slope and Manning n)."""
    return Domain(
        x_m=profile.station_m,
        dx_m=profile.dx_m,
        slope=profile.slope,
        manning_n=profile.manning_n,
    )
