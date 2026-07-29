"""Run a reproducible two-gauge river validation case.

The upstream observed discharge drives the model boundary.  The downstream
observations are held out and scored against the simulated downstream
hydrograph.  Configuration and observations are plain JSON/CSV so a committed
case runs offline and retains its provenance.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from general.solvers import saint_venant_1d
from rivers.validation.compare import evaluate_series


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_two_gauge_observations(path):
    """Return minutes relative to the first held-out downstream observation."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Validation observation CSV is empty")

    parsed = []
    for row in rows:
        role = row["role"].strip().lower()
        if role not in {"upstream", "downstream"}:
            raise ValueError("Observation role must be upstream or downstream")
        timestamp = _timestamp(row["observed_at"])
        discharge = float(row["discharge_m3_per_min"])
        if not math.isfinite(discharge) or discharge < 0:
            raise ValueError("Observed discharge must be finite and non-negative")
        parsed.append((role, timestamp, discharge))

    downstream_timestamps = [
        timestamp for role, timestamp, _ in parsed if role == "downstream"
    ]
    if not downstream_timestamps:
        raise ValueError("Validation case needs downstream observations")
    origin = min(downstream_timestamps)
    result = {}
    for role in ("upstream", "downstream"):
        selected = sorted(
            (
                ((timestamp - origin).total_seconds() / 60.0, discharge)
                for row_role, timestamp, discharge in parsed
                if row_role == role
            ),
            key=lambda item: item[0],
        )
        if len(selected) < 2:
            raise ValueError(f"Validation case needs at least two {role} observations")
        times = np.asarray([item[0] for item in selected], dtype=float)
        values = np.asarray([item[1] for item in selected], dtype=float)
        if np.any(np.diff(times) <= 0):
            raise ValueError(f"{role} observation times must be strictly increasing")
        result[role] = (times, values)
    return result


def discharge_boundary(times_min, discharge_m3_per_min):
    """Create a whole-channel upstream hydrograph callable in m³/min."""
    times = np.asarray(times_min, dtype=float)
    discharge = np.asarray(discharge_m3_per_min, dtype=float)

    def boundary(time_min):
        return float(np.interp(time_min, times, discharge))

    boundary.breakpoints_min = tuple(float(value) for value in times)
    return boundary


def shifted_boundary(boundary, offset_min):
    """Shift a forcing onto a simulation clock while retaining its breakpoints."""
    offset = float(offset_min)

    def shifted(time_min):
        return boundary(float(time_min) + offset)

    shifted.breakpoints_min = tuple(
        float(value - offset)
        for value in getattr(boundary, "breakpoints_min", ())
    )
    return shifted


def fractional_lateral_inflow(boundary, fraction, length_m):
    """Distribute a fraction of upstream flow uniformly along the reach."""
    fraction = float(fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("lateral_inflow_fraction must be between 0 and 1")
    length = float(length_m)

    def lateral(x_m, time_min):
        flow = boundary(time_min) if callable(boundary) else float(boundary)
        rate = fraction * flow / length
        return np.full_like(x_m, rate, dtype=float)

    lateral.breakpoints_min = getattr(boundary, "breakpoints_min", ())
    return lateral


def rectangular_normal_depth(discharge, width, manning_n, slope):
    """Solve Manning's relation for depth in a rectangular cross-section."""
    if min(discharge, width, manning_n, slope) <= 0:
        raise ValueError("Normal-depth inputs must be positive")

    def residual(depth):
        area = width * depth
        hydraulic_radius = area / (width + 2.0 * depth)
        predicted = (
            area
            * hydraulic_radius ** (2.0 / 3.0)
            * math.sqrt(slope)
            / manning_n
        )
        return predicted - discharge

    lower, upper = 0.0, 1.0
    while residual(upper) < 0.0:
        upper *= 2.0
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if residual(midpoint) < 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def run_validation_case(config_path, *, output_path=None, overrides=None):
    """Run one configured Saint-Venant case and return its validation evidence."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if overrides:
        config = copy.deepcopy(config)
        config["reach"].update(overrides.get("reach", {}))
        for key, value in overrides.items():
            if key != "reach":
                config[key] = value
    observation_path = Path(config["observations"])
    if not observation_path.is_absolute():
        observation_path = config_path.parent / observation_path
    observations = load_two_gauge_observations(observation_path)
    upstream_times, upstream_flow = observations["upstream"]
    downstream_times, downstream_flow = observations["downstream"]

    reach = config["reach"]
    length_m = float(reach["length_m"])
    cells = int(reach["cells"])
    slope = float(reach["slope"])
    manning_n = float(reach["manning_n"])
    upstream_width = float(reach["upstream_width_m"])
    downstream_width = float(reach["downstream_width_m"])
    if length_m <= 0 or cells < 2 or slope <= 0 or manning_n <= 0:
        raise ValueError("Reach length, cells, slope, and Manning n must be positive")

    duration = min(float(upstream_times[-1]), float(downstream_times[-1]))
    if duration <= 0:
        raise ValueError("Upstream and downstream observations have no positive overlap")
    x_m = np.linspace(0.0, length_m, cells)
    dx_m = np.full(cells, length_m / cells)
    bed_slope = np.full(cells, slope)
    roughness = np.full(cells, manning_n)
    channel_width = np.linspace(upstream_width, downstream_width, cells)
    boundary = discharge_boundary(upstream_times, upstream_flow)
    lateral_fraction = float(config.get("lateral_inflow_fraction", 0.0))
    lateral_inflow = fractional_lateral_inflow(
        boundary, lateral_fraction, length_m
    )
    warmup_config = config.get("warmup", {})
    warmup_min = float(
        warmup_config.get("duration_min", config.get("warmup_min", 0.0))
    )
    warmup_forcing = warmup_config.get("upstream_forcing", "constant_initial")
    if warmup_forcing not in {"constant_initial", "observed"}:
        raise ValueError(
            "warmup upstream_forcing must be 'constant_initial' or 'observed'"
        )
    warmup_start = -warmup_min
    if warmup_forcing == "observed" and upstream_times[0] > warmup_start + 1e-9:
        raise ValueError(
            "Observed warm-up requires upstream observations through the warm-up start"
        )
    initial_q = boundary(warmup_start if warmup_forcing == "observed" else 0.0)
    normal_depth = rectangular_normal_depth(
        initial_q, upstream_width, manning_n, slope
    )
    spatial_order = int(config.get("spatial_order", 1))
    initial_depth = np.full(cells, normal_depth)
    initial_discharge = np.full(cells, initial_q)
    if warmup_min > 0:
        warmup_boundary = (
            shifted_boundary(boundary, warmup_start)
            if warmup_forcing == "observed"
            else initial_q
        )
        warmup_lateral = fractional_lateral_inflow(
            warmup_boundary, lateral_fraction, length_m
        )
        warmup = saint_venant_1d.run_model(
            length_m,
            warmup_min,
            record_interval=warmup_min,
            h_init=initial_depth,
            q_init=initial_discharge,
            left_inflow=warmup_boundary,
            rainfall=lambda x, time: np.zeros_like(x),
            lateral_inflow=warmup_lateral,
            x_m=x_m,
            dx_m=dx_m,
            slope=bed_slope,
            manning_n=roughness,
            channel_width_m=channel_width,
            cfl=float(config.get("cfl", 0.4)),
            spatial_order=spatial_order,
        )
        initial_depth = warmup["h_final"]
        initial_discharge = warmup["q_final"]

    result = saint_venant_1d.run_model(
        length_m,
        duration,
        record_interval=float(config.get("record_interval_min", 5.0)),
        h_init=initial_depth,
        q_init=initial_discharge,
        left_inflow=boundary,
        rainfall=lambda x, time: np.zeros_like(x),
        lateral_inflow=lateral_inflow,
        x_m=x_m,
        dx_m=dx_m,
        slope=bed_slope,
        manning_n=roughness,
        channel_width_m=channel_width,
        cfl=float(config.get("cfl", 0.4)),
        spatial_order=spatial_order,
    )

    target = (downstream_times >= 0.0) & (downstream_times <= duration)
    target_times = downstream_times[target]
    target_flow = downstream_flow[target]
    predicted_flow = result["q_history"][:, -1]
    scores = evaluate_series(
        target_times,
        target_flow,
        result["times"],
        predicted_flow,
    )
    evidence = {
        "schema_version": 1,
        "case": config["case"],
        "solver": "saint_venant",
        "status": config.get("validation_status", "uncalibrated_baseline"),
        "observations": {
            "path": str(observation_path),
            "upstream_count": int(len(upstream_times)),
            "downstream_count": int(len(target_times)),
            "duration_min": duration,
        },
        "assumptions": {
            **reach,
            "initial_condition": (
                "observed-upstream dynamic warm-up from uniform Manning normal flow"
                if warmup_min > 0 and warmup_forcing == "observed"
                else "constant-boundary hydraulic warm-up from uniform Manning normal flow"
                if warmup_min > 0
                else "uniform rectangular-section Manning normal depth and discharge"
            ),
            "warmup_min": warmup_min,
            "warmup_upstream_forcing": warmup_forcing,
            "spatial_order": spatial_order,
            "lateral_inflow": (
                "zero"
                if lateral_fraction == 0.0
                else "uniformly distributed fraction of observed upstream flow"
            ),
            "lateral_inflow_fraction": lateral_fraction,
            "rainfall": "zero",
        },
        "scores": {
            key: (
                int(value)
                if key == "n"
                else None if not math.isfinite(float(value)) else float(value)
            )
            for key, value in scores.items()
            if key != "predicted_on_obs"
        },
        "mass": {
            "inflow_m3": float(result["mass_inflow"]),
            "source_m3": float(result["mass_source"]),
            "outflow_m3": float(result["mass_outflow"]),
            "floor_correction_m3": float(result["mass_floor_correction"]),
            "rainfall_m3": float(result["mass_rainfall"]),
            "lateral_inflow_m3": float(result["mass_lateral_inflow"]),
        },
        "series": {
            "observed_times_min": target_times.tolist(),
            "observed_downstream_m3_per_min": target_flow.tolist(),
            "predicted_downstream_m3_per_min": scores["predicted_on_obs"].tolist(),
        },
    }
    destination = (
        Path(output_path)
        if output_path is not None
        else config_path.with_suffix(".results.json")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run an offline two-gauge real-river validation case."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    evidence = run_validation_case(args.config, output_path=args.output)
    print(json.dumps({"case": evidence["case"], "scores": evidence["scores"]}, indent=2))


if __name__ == "__main__":
    main()
