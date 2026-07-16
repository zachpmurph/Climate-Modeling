import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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


def q(depth_m, slope, manning_n):
    depth_m = np.maximum(depth_m, 0.0)
    return (1.0 / manning_n) * (depth_m ** (5.0 / 3.0)) * np.sqrt(slope)


def c(depth_m, slope, manning_n):
    depth_m = np.maximum(depth_m, 0.0)
    return (5.0 / (3.0 * manning_n)) * (depth_m ** (2.0 / 3.0)) * np.sqrt(slope)


def _initial_depth(profile, base_depth_m, wave_center_m, wave_amplitude_m, wave_width_m):
    if profile.initial_depth_m is not None:
        depth = profile.initial_depth_m.copy()
    else:
        depth = np.full_like(profile.station_m, base_depth_m, dtype=float)

    if wave_amplitude_m != 0.0:
        if wave_center_m is None:
            wave_center_m = float(profile.station_m[0] + 0.25 * (profile.station_m[-1] - profile.station_m[0]))
        if wave_width_m is None:
            wave_width_m = max(float(profile.length_m) / 20.0, float(np.min(profile.dx_m)))
        depth += wave_amplitude_m * np.exp(-((profile.station_m - wave_center_m) ** 2) / (2.0 * wave_width_m ** 2))

    return np.maximum(depth, MIN_DEPTH)


def _rainfall_source(profile, rainfall_rate_m_per_min, rainfall_start_min, rainfall_end_min, t_current):
    if rainfall_end_min is not None and rainfall_end_min < rainfall_start_min:
        raise ValueError("rainfall_end_min must be greater than or equal to rainfall_start_min")
    if t_current < rainfall_start_min:
        return np.zeros_like(profile.station_m, dtype=float)
    if rainfall_end_min is not None and t_current >= rainfall_end_min:
        return np.zeros_like(profile.station_m, dtype=float)

    source = np.zeros_like(profile.station_m, dtype=float)
    if profile.rainfall_rate_m_per_min is not None:
        source += profile.rainfall_rate_m_per_min
    source += rainfall_rate_m_per_min
    return source


def run_model(
    profile,
    t_final_min,
    left_inflow_flux,
    record_interval_min=1.0,
    base_depth_m=0.01,
    wave_center_m=None,
    wave_amplitude_m=0.0,
    wave_width_m=None,
    rainfall_rate_m_per_min=0.0,
    rainfall_start_min=0.0,
    rainfall_end_min=None,
    cfl=0.5,
):
    """Run a 1D river kinematic wave model with upstream inflow and rainfall.

    ``left_inflow_flux`` uses the same unit-width flux convention as the
    existing solver: depth-area flux in square meters per minute. Rainfall
    source terms are depth added per minute. The model state is water depth
    in meters.
    """
    if t_final_min < 0:
        raise ValueError("t_final_min must be non-negative")
    if record_interval_min <= 0:
        raise ValueError("record_interval_min must be positive")
    if left_inflow_flux < 0:
        raise ValueError("left_inflow_flux must be non-negative")
    if rainfall_rate_m_per_min < 0:
        raise ValueError("rainfall_rate_m_per_min must be non-negative")
    if rainfall_start_min < 0:
        raise ValueError("rainfall_start_min must be non-negative")
    if rainfall_end_min is not None and rainfall_end_min < rainfall_start_min:
        raise ValueError("rainfall_end_min must be greater than or equal to rainfall_start_min")
    if not (0 < cfl <= 1):
        raise ValueError("cfl must be in the interval (0, 1]")

    depth = _initial_depth(profile, base_depth_m, wave_center_m, wave_amplitude_m, wave_width_m)
    initial_depth = depth.copy()

    n_marks = int(np.floor(t_final_min / record_interval_min + 1e-9))
    record_times = [i * record_interval_min for i in range(n_marks + 1)]
    if not record_times or record_times[-1] < t_final_min - 1e-9:
        record_times.append(float(t_final_min))

    times = [0.0]
    history = [initial_depth.copy()]
    next_record_idx = 1
    t_current = 0.0

    mass_inflow = 0.0
    mass_source = 0.0
    mass_outflow = 0.0

    while t_current < t_final_min - 1e-12:
        wave_speed = c(depth, profile.slope, profile.manning_n)
        c_max = float(np.max(wave_speed))
        if c_max > 0:
            dt = cfl * float(np.min(profile.dx_m)) / c_max
        else:
            dt = t_final_min - t_current

        dt = min(dt, t_final_min - t_current)
        if next_record_idx < len(record_times):
            dt = min(dt, record_times[next_record_idx] - t_current)
        if rainfall_end_min is not None and t_current < rainfall_end_min < t_current + dt:
            dt = rainfall_end_min - t_current
        if t_current < rainfall_start_min < t_current + dt:
            dt = rainfall_start_min - t_current
        if dt <= 1e-12:
            dt = min(t_final_min - t_current, 1e-12)

        cell_flux = q(depth, profile.slope, profile.manning_n)
        interface_flux = np.empty(len(depth) + 1, dtype=float)
        interface_flux[0] = left_inflow_flux
        interface_flux[1:] = cell_flux

        source = _rainfall_source(profile, rainfall_rate_m_per_min, rainfall_start_min, rainfall_end_min, t_current)
        depth = depth - (dt / profile.dx_m) * (interface_flux[1:] - interface_flux[:-1])
        depth = depth + dt * source
        depth = np.maximum(depth, MIN_DEPTH)

        mass_inflow += left_inflow_flux * dt
        mass_source += float(np.sum(source * profile.dx_m) * dt)
        mass_outflow += cell_flux[-1] * dt
        t_current += dt

        if next_record_idx < len(record_times) and t_current >= record_times[next_record_idx] - 1e-9:
            times.append(record_times[next_record_idx])
            history.append(depth.copy())
            next_record_idx += 1

    return {
        "station_m": profile.station_m,
        "dx_m": profile.dx_m,
        "slope": profile.slope,
        "manning_n": profile.manning_n,
        "times": np.array(times),
        "depth_history": np.array(history),
        "depth_initial": initial_depth,
        "depth_final": depth,
        "mass_inflow": mass_inflow,
        "mass_source": mass_source,
        "mass_outflow": mass_outflow,
        "left_inflow_flux": left_inflow_flux,
        "rainfall_rate_m_per_min": rainfall_rate_m_per_min,
        "rainfall_start_min": rainfall_start_min,
        "rainfall_end_min": rainfall_end_min,
    }


def save_time_series_csv(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_min"] + [f"{station:.6f}" for station in result["station_m"]])
        for t, depth_row in zip(result["times"], result["depth_history"]):
            writer.writerow([f"{t:.6f}"] + [f"{depth:.10g}" for depth in depth_row])


def save_summary_json(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    storage_initial = float(np.sum(result["depth_initial"] * result["dx_m"]))
    storage_final = float(np.sum(result["depth_final"] * result["dx_m"]))
    expected_delta = result["mass_inflow"] + result["mass_source"] - result["mass_outflow"]
    summary = {
        "t_start_min": float(result["times"][0]),
        "t_final_min": float(result["times"][-1]),
        "cells": int(len(result["station_m"])),
        "river_length_m": float(np.sum(result["dx_m"])),
        "left_inflow_flux_m2_per_min": float(result["left_inflow_flux"]),
        "rainfall_rate_m_per_min": float(result["rainfall_rate_m_per_min"]),
        "rainfall_start_min": float(result["rainfall_start_min"]),
        "rainfall_end_min": None if result["rainfall_end_min"] is None else float(result["rainfall_end_min"]),
        "mass_inflow_m2": float(result["mass_inflow"]),
        "mass_source_m2": float(result["mass_source"]),
        "mass_outflow_m2": float(result["mass_outflow"]),
        "storage_initial_m2": storage_initial,
        "storage_final_m2": storage_final,
        "storage_delta_m2": storage_final - storage_initial,
        "mass_balance_error_m2": (storage_final - storage_initial) - expected_delta,
        "max_depth_final_m": float(np.max(result["depth_final"])),
    }
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
