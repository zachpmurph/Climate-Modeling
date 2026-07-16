import csv
import json
from pathlib import Path

import numpy as np

from general.solvers.contract import Domain, Scenario, SimulationResult
from general.solvers.profile import (
    RiverProfile,
    domain_from_profile,
    make_profile,
    load_profile,
    load_profile_csv,
    load_profile_json,
)

MIN_DEPTH = 1e-10


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


class _RiverKinematicWaveSolver:
    name = "river_kinematic_wave"
    supports = frozenset({"initial_depth", "left_inflow", "rainfall", "cfl"})

    def run(self, domain: Domain, scenario: Scenario) -> SimulationResult:
        init_depth = scenario.initial_depth_m
        profile = RiverProfile(
            station_m=domain.x_m,
            dx_m=domain.dx_m,
            slope=domain.slope,
            manning_n=domain.manning_n,
            initial_depth_m=init_depth if isinstance(init_depth, np.ndarray) else None,
        )

        left_inflow = scenario.left_inflow
        if callable(left_inflow):
            left_inflow = float(left_inflow(0.0))

        rainfall_rate = 0.0
        if scenario.rainfall is not None:
            sample = scenario.rainfall(domain.x_m, 0.0)
            rainfall_rate = float(np.mean(sample))

        base_depth_m = float(init_depth) if not isinstance(init_depth, np.ndarray) else 0.01

        result = run_model(
            profile,
            t_final_min=scenario.t_final_min,
            left_inflow_flux=float(left_inflow),
            record_interval_min=scenario.record_interval_min,
            rainfall_rate_m_per_min=rainfall_rate,
            cfl=scenario.cfl,
            base_depth_m=base_depth_m,
        )

        return SimulationResult(
            domain=domain,
            times=result["times"],
            depth_history=result["depth_history"],
            depth_initial=result["depth_initial"],
            depth_final=result["depth_final"],
            mass_inflow=result["mass_inflow"],
            mass_source=result["mass_source"],
            mass_outflow=result["mass_outflow"],
        )


SOLVER = _RiverKinematicWaveSolver()
