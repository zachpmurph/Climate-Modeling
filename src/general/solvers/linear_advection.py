"""1-D river kinematic wave model.

This is the project's kinematic wave solver. It runs directly on a
``RiverProfile`` (per-cell slope and Manning n), with optional constant upstream
inflow and a rainfall source term, and integrates water depth forward with a
conservative finite-volume upwind scheme and an adaptive (CFL-limited) time step.

Two ways to run it:

* **Standalone:** ``python src/general/solvers/linear_advection.py [profile]`` runs
  a profile (CSV/JSON) and writes a depth time-series CSV and a summary JSON. With
  no profile it runs a built-in uniform demo and also writes ``data/linear_advection.png``.
* **Harness:** the module-level ``SOLVER`` (name ``"kinematic_wave"``) plugs into
  ``src/rivers/simulations/run_simulation.py`` via the solver registry.

Units are meters and minutes throughout.
"""

import csv
import json
import sys
from pathlib import Path

# Allow running this file directly (python src/general/solvers/linear_advection.py):
# put src/ on the path so the general.* / rivers.* packages import either way.
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import numpy as np

from general.solvers.contract import Domain, Scenario, SimulationResult
from general.solvers.profile import (
    RiverProfile,
    domain_from_profile,
    load_profile,
    load_profile_csv,
    load_profile_json,
    make_profile,
)

MIN_DEPTH = 1e-10


def q(depth_m, slope, manning_n):
    """Manning discharge per unit width (m^2/min in this repo's convention)."""
    depth_m = np.maximum(depth_m, 0.0)
    return (1.0 / manning_n) * (depth_m ** (5.0 / 3.0)) * np.sqrt(slope)


def c(depth_m, slope, manning_n):
    """Kinematic wave speed dq/dh."""
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


def _evaluate_rainfall(rainfall, stations_m, t_current):
    if rainfall is None:
        return np.zeros_like(stations_m, dtype=float)
    values = np.asarray(rainfall(stations_m, t_current), dtype=float)
    if values.ndim == 0:
        values = np.full_like(stations_m, float(values), dtype=float)
    if values.shape != stations_m.shape:
        raise ValueError("rainfall must return one value per cell")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("rainfall must return finite, non-negative rates")
    return values


def _rainfall_source(
    profile,
    rainfall_rate_m_per_min,
    rainfall_start_min,
    rainfall_end_min,
    rainfall,
    t_current,
):
    if rainfall_end_min is not None and rainfall_end_min < rainfall_start_min:
        raise ValueError("rainfall_end_min must be greater than or equal to rainfall_start_min")

    source = np.zeros_like(profile.station_m, dtype=float)
    if profile.rainfall_rate_m_per_min is not None:
        source += profile.rainfall_rate_m_per_min
    uniform_is_active = t_current >= rainfall_start_min and (
        rainfall_end_min is None or t_current < rainfall_end_min
    )
    if uniform_is_active:
        source += rainfall_rate_m_per_min
    source += _evaluate_rainfall(rainfall, profile.station_m, t_current)
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
    rainfall=None,
):
    """Run a 1D river kinematic wave model with upstream inflow and rainfall.

    ``left_inflow_flux`` is the depth-area flux entering the left boundary in
    square meters per minute. Rainfall source terms are depth added per minute.
    ``rainfall`` may be a callable ``rainfall(station_m, t_min)`` returning one
    non-negative rate per cell. The model state is water depth in meters.
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
        # Adaptive time step from the CFL condition against the current max wave
        # speed -- c(h) is nonlinear, so a fixed dt can go unstable as h grows.
        wave_speed = c(depth, profile.slope, profile.manning_n)
        c_max = float(np.max(wave_speed))
        if c_max > 0:
            dt = cfl * float(np.min(profile.dx_m)) / c_max
        else:
            dt = t_final_min - t_current

        dt = min(dt, t_final_min - t_current)
        if next_record_idx < len(record_times):
            dt = min(dt, record_times[next_record_idx] - t_current)
        # Land exactly on rainfall on/off transitions so the source integral is exact.
        if rainfall_end_min is not None and t_current < rainfall_end_min < t_current + dt:
            dt = rainfall_end_min - t_current
        if t_current < rainfall_start_min < t_current + dt:
            dt = rainfall_start_min - t_current
        if dt <= 1e-12:
            dt = min(t_final_min - t_current, 1e-12)

        # Conservative upwind flux update: left interface carries the upstream
        # inflow, interior interfaces carry the upwind cell's Manning flux.
        cell_flux = q(depth, profile.slope, profile.manning_n)
        interface_flux = np.empty(len(depth) + 1, dtype=float)
        interface_flux[0] = left_inflow_flux
        interface_flux[1:] = cell_flux

        source = _rainfall_source(
            profile,
            rainfall_rate_m_per_min,
            rainfall_start_min,
            rainfall_end_min,
            rainfall,
            t_current,
        )
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
    """Write the recorded (t, depth(x)) table: one row per recorded time, one
    column per cell. Read back by src/general/viz/animate_depth.py."""
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


class _KinematicWaveSolver:
    name = "kinematic_wave"
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

        base_depth_m = float(init_depth) if not isinstance(init_depth, np.ndarray) else 0.01

        result = run_model(
            profile,
            t_final_min=scenario.t_final_min,
            left_inflow_flux=float(left_inflow),
            record_interval_min=scenario.record_interval_min,
            rainfall=scenario.rainfall,
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


SOLVER = _KinematicWaveSolver()


def _demo_profile():
    """A uniform reach used when the script is run with no profile argument."""
    n_cells = 101
    return make_profile(
        station_m=np.linspace(0.0, 10.0, n_cells),
        slope=np.full(n_cells, 0.05),
        manning_n=np.full(n_cells, 0.05),
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv:
        profile = load_profile(argv[0])
        run_name = Path(argv[0]).stem
        result = run_model(profile, t_final_min=300.0, left_inflow_flux=0.0, rainfall_rate_m_per_min=0.00002,
                           rainfall_end_min=50.0)
    else:
        profile = _demo_profile()
        run_name = "linear_advection"
        result = run_model(profile, t_final_min=300.0, left_inflow_flux=0.0, rainfall_rate_m_per_min=0.00002,
                           rainfall_end_min=50.0)

    out_dir = Path("data")
    save_time_series_csv(result, out_dir / f"{run_name}_timeseries.csv")
    summary = save_summary_json(result, out_dir / f"{run_name}_summary.json")

    if not argv:
        # Preserve the historical no-argument demo artifact.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.plot(result["station_m"], result["depth_initial"], label="Initial")
        plt.plot(result["station_m"], result["depth_final"], label="Final", ls="--")
        plt.legend(); plt.xlabel("station (m)"); plt.ylabel("depth (m)")
        plt.savefig(out_dir / "linear_advection.png")

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
