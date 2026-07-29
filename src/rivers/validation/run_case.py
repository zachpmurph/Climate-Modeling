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
from general.solvers.profile import (
    load_channel_geometry,
    load_compound_cross_sections,
    load_surveyed_cross_sections,
)
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


def shifted_spatial_forcing(forcing, offset_min):
    """Shift an ``f(x, time)`` forcing onto a new simulation clock."""
    offset = float(offset_min)

    def shifted(x_m, time_min):
        return forcing(x_m, float(time_min) + offset)

    shifted.breakpoints_min = tuple(
        float(value - offset)
        for value in getattr(forcing, "breakpoints_min", ())
    )
    return shifted


def _event_relative_time(row, event_start):
    relative = row.get("t_min")
    observed = row.get("observed_at")
    if relative is not None and str(relative).strip():
        return float(relative)
    if observed is not None and str(observed).strip():
        return (
            _timestamp(observed) - event_start
        ).total_seconds() / 60.0
    raise ValueError("Control rows require t_min or observed_at")


def load_event_control_series(path, value_column, event_start):
    """Load a timestamped or event-relative scalar control series."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError(f"{path} must contain at least two control rows")
    try:
        times = np.asarray(
            [_event_relative_time(row, event_start) for row in rows],
            dtype=float,
        )
        values = np.asarray(
            [float(row[value_column]) for row in rows], dtype=float
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{path} must contain numeric {value_column} and either t_min "
            "or observed_at"
        ) from exc
    if (
        np.any(~np.isfinite(times))
        or np.any(~np.isfinite(values))
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError(
            f"{path} control times must be finite and strictly increasing, "
            f"with finite {value_column}"
        )

    def control(time_min):
        return float(np.interp(time_min, times, values))

    control.breakpoints_min = tuple(float(value) for value in times)
    control.coverage_min = (float(times[0]), float(times[-1]))
    return control


def load_event_point_flows(path, x_m, dx_m, event_start):
    """Map signed, timestamped point flows to conservative model-cell rates."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} must contain point-flow rows")
    grouped = {}
    try:
        for row in rows:
            station = float(row["station_m"])
            time = _event_relative_time(row, event_start)
            flow = float(row["discharge_m3_per_min"])
            grouped.setdefault(station, []).append((time, flow))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{path} must contain numeric station_m, "
            "discharge_m3_per_min, and either t_min or observed_at"
        ) from exc

    stations = np.asarray(x_m, dtype=float)
    cell_lengths = np.asarray(dx_m, dtype=float)
    series = []
    all_breakpoints = set()
    for station, observations in grouped.items():
        observations.sort()
        times = np.asarray([item[0] for item in observations], dtype=float)
        flows = np.asarray([item[1] for item in observations], dtype=float)
        if (
            len(times) < 2
            or not math.isfinite(station)
            or station < stations[0]
            or station > stations[-1]
            or np.any(~np.isfinite(times))
            or np.any(~np.isfinite(flows))
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError(
                f"Each point flow in {path} needs a station inside the "
                "domain and at least two finite values at strictly increasing "
                "times"
            )
        cell = int(np.argmin(np.abs(stations - station)))
        series.append((cell, times, flows))
        all_breakpoints.update(float(value) for value in times)

    def lateral_flow(x, time_min):
        rates = np.zeros_like(x, dtype=float)
        for cell, times, flows in series:
            rates[cell] += (
                float(np.interp(time_min, times, flows))
                / cell_lengths[cell]
            )
        return rates

    lateral_flow.breakpoints_min = tuple(sorted(all_breakpoints))
    lateral_flow.coverage_min = (
        min(item[1][0] for item in series),
        max(item[1][-1] for item in series),
    )
    lateral_flow.series_coverage_min = tuple(
        (float(times[0]), float(times[-1])) for _, times, _ in series
    )
    lateral_flow.point_count = len(series)
    return lateral_flow


def _require_control_coverage(control, start_min, end_min, label):
    coverages = getattr(
        control,
        "series_coverage_min",
        (getattr(control, "coverage_min"),),
    )
    if any(
        start > start_min + 1e-9 or end < end_min - 1e-9
        for start, end in coverages
    ):
        raise ValueError(
            f"{label} must cover the full simulation from "
            f"{start_min:g} to {end_min:g} event minutes"
        )


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
    return manning_normal_depth(
        discharge,
        width,
        0.0,
        manning_n,
        slope,
    )


def manning_normal_depth(
    discharge,
    bottom_width,
    side_slope,
    manning_n,
    slope,
    *,
    table_depth=None,
    table_width=None,
    table_perimeter=None,
):
    """Solve Manning's relation for an arbitrary supported cross-section."""
    inputs = (
        float(discharge),
        float(bottom_width),
        float(side_slope),
        float(manning_n),
        float(slope),
    )
    if (
        not all(math.isfinite(value) for value in inputs)
        or min(discharge, bottom_width, manning_n, slope) <= 0
        or side_slope < 0
    ):
        raise ValueError("Normal-depth inputs must be positive")

    def residual(depth):
        area = saint_venant_1d._cross_section_area(
            depth,
            bottom_width,
            side_slope,
            table_depth,
            table_width,
        )
        hydraulic_radius = saint_venant_1d._hydraulic_radius(
            depth,
            bottom_width,
            side_slope,
            table_depth,
            table_width,
            table_perimeter,
        )
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


def _configured_path(config_path, value):
    path = Path(value)
    return path if path.is_absolute() else config_path.parent / path


def _geometry_source(config_path, reach, key):
    if not reach.get(key):
        raise ValueError(
            f"{reach.get('cross_section_shape')} validation geometry requires {key}"
        )
    source = _configured_path(config_path, reach[key])
    if not source.is_file():
        raise ValueError(f"Validation geometry does not exist: {source}")
    return source


def load_validation_geometry(config_path, reach, x_m):
    """Load the configured cross-section model onto validation cells."""
    shape = reach.get("cross_section_shape", "rectangular")
    geometry = {
        "channel_width_m": None,
        "channel_bottom_width_m": None,
        "side_slope_h_to_v": None,
        "cross_section_depth_m": None,
        "cross_section_top_width_m": None,
        "cross_section_wetted_perimeter_m": None,
    }
    provenance = {"cross_section_shape": shape}
    if shape == "rectangular":
        geometry["channel_width_m"] = np.linspace(
            float(reach["upstream_width_m"]),
            float(reach["downstream_width_m"]),
            len(x_m),
        )
        return geometry, provenance
    if shape == "trapezoidal":
        source = _geometry_source(config_path, reach, "hydraulic_geometry")
        width, bankfull = load_channel_geometry(source, x_m)
        fraction = float(reach.get("bottom_width_fraction", 0.5))
        if not 0.0 < fraction <= 1.0:
            raise ValueError("bottom_width_fraction must be in (0, 1]")
        bottom = fraction * width
        geometry.update(
            {
                "channel_width_m": width,
                "channel_bottom_width_m": bottom,
                "side_slope_h_to_v": (width - bottom) / (2.0 * bankfull),
            }
        )
        provenance.update(
            {
                "hydraulic_geometry": str(reach["hydraulic_geometry"]),
                "bottom_width_fraction": fraction,
            }
        )
        return geometry, provenance
    if shape == "compound":
        source = _geometry_source(
            config_path, reach, "compound_cross_sections"
        )
        depth, width = load_compound_cross_sections(source, x_m)
        geometry.update(
            {
                "channel_width_m": width[:, -1],
                "cross_section_depth_m": depth,
                "cross_section_top_width_m": width,
            }
        )
        provenance["compound_cross_sections"] = str(
            reach["compound_cross_sections"]
        )
        return geometry, provenance
    if shape == "surveyed":
        source = _geometry_source(
            config_path, reach, "surveyed_cross_sections"
        )
        depth, width, perimeter = load_surveyed_cross_sections(source, x_m)
        geometry.update(
            {
                "channel_width_m": width[:, -1],
                "cross_section_depth_m": depth,
                "cross_section_top_width_m": width,
                "cross_section_wetted_perimeter_m": perimeter,
            }
        )
        provenance["surveyed_cross_sections"] = str(
            reach["surveyed_cross_sections"]
        )
        return geometry, provenance
    raise ValueError(
        "cross_section_shape must be rectangular, trapezoidal, compound, or surveyed"
    )


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
    if length_m <= 0 or cells < 2 or slope <= 0 or manning_n <= 0:
        raise ValueError("Reach length, cells, slope, and Manning n must be positive")

    duration = min(float(upstream_times[-1]), float(downstream_times[-1]))
    if duration <= 0:
        raise ValueError("Upstream and downstream observations have no positive overlap")
    x_m = np.linspace(0.0, length_m, cells)
    dx_m = np.full(cells, length_m / cells)
    bed_slope = np.full(cells, slope)
    roughness = np.full(cells, manning_n)
    geometry, geometry_provenance = load_validation_geometry(
        config_path, reach, x_m
    )
    channel_width = geometry["channel_width_m"]
    boundary = discharge_boundary(upstream_times, upstream_flow)
    lateral_fraction = float(config.get("lateral_inflow_fraction", 0.0))
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
    event_start = _timestamp(config["case"]["observation_window"][0])
    point_flow_source = config.get("point_flow_series")
    if point_flow_source is not None and lateral_fraction != 0.0:
        raise ValueError(
            "Measured point_flow_series cannot be combined with a uniform "
            "lateral_inflow_fraction"
        )
    if point_flow_source is None:
        lateral_inflow = fractional_lateral_inflow(
            boundary, lateral_fraction, length_m
        )
        point_flow_path = None
    else:
        point_flow_path = _configured_path(config_path, point_flow_source)
        if not point_flow_path.is_file():
            raise ValueError(
                f"Point-flow control does not exist: {point_flow_path}"
            )
        lateral_inflow = load_event_point_flows(
            point_flow_path, x_m, dx_m, event_start
        )
        _require_control_coverage(
            lateral_inflow,
            warmup_start,
            duration,
            "point_flow_series",
        )

    downstream_stage_source = config.get("downstream_stage_series")
    if downstream_stage_source is None:
        downstream_boundary = "outflow"
        downstream_stage = None
        downstream_stage_path = None
    else:
        downstream_stage_path = _configured_path(
            config_path, downstream_stage_source
        )
        if not downstream_stage_path.is_file():
            raise ValueError(
                f"Downstream-stage control does not exist: "
                f"{downstream_stage_path}"
            )
        downstream_stage = load_event_control_series(
            downstream_stage_path,
            "downstream_stage_m",
            event_start,
        )
        _require_control_coverage(
            downstream_stage,
            warmup_start,
            duration,
            "downstream_stage_series",
        )
        downstream_boundary = "stage"

    initial_q = boundary(warmup_start if warmup_forcing == "observed" else 0.0)
    spatial_order = int(config.get("spatial_order", 1))
    bottom_width = geometry["channel_bottom_width_m"]
    if bottom_width is None:
        bottom_width = channel_width
    side_slope = geometry["side_slope_h_to_v"]
    if side_slope is None:
        side_slope = np.zeros(cells)
    table_depth = geometry["cross_section_depth_m"]
    table_width = geometry["cross_section_top_width_m"]
    table_perimeter = geometry["cross_section_wetted_perimeter_m"]
    initial_depth = np.asarray(
        [
            manning_normal_depth(
                initial_q,
                bottom_width[cell],
                side_slope[cell],
                roughness[cell],
                bed_slope[cell],
                table_depth=table_depth,
                table_width=(
                    None if table_width is None else table_width[cell]
                ),
                table_perimeter=(
                    None if table_perimeter is None else table_perimeter[cell]
                ),
            )
            for cell in range(cells)
        ]
    )
    initial_discharge = np.full(cells, initial_q)
    geometry_arguments = {
        key: value for key, value in geometry.items() if value is not None
    }
    if warmup_min > 0:
        warmup_boundary = (
            shifted_boundary(boundary, warmup_start)
            if warmup_forcing == "observed"
            else initial_q
        )
        warmup_lateral = (
            fractional_lateral_inflow(
                warmup_boundary, lateral_fraction, length_m
            )
            if point_flow_path is None
            else shifted_spatial_forcing(lateral_inflow, warmup_start)
        )
        warmup_stage = (
            None
            if downstream_stage is None
            else shifted_boundary(downstream_stage, warmup_start)
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
            **geometry_arguments,
            cfl=float(config.get("cfl", 0.4)),
            spatial_order=spatial_order,
            downstream_boundary=downstream_boundary,
            downstream_stage_m=warmup_stage,
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
        **geometry_arguments,
        cfl=float(config.get("cfl", 0.4)),
        spatial_order=spatial_order,
        downstream_boundary=downstream_boundary,
        downstream_stage_m=downstream_stage,
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
            **geometry_provenance,
            "initial_condition": (
                "observed-upstream dynamic warm-up from per-cell Manning normal flow"
                if warmup_min > 0 and warmup_forcing == "observed"
                else "constant-boundary hydraulic warm-up from per-cell Manning normal flow"
                if warmup_min > 0
                else "per-cell cross-section Manning normal depth and discharge"
            ),
            "warmup_min": warmup_min,
            "warmup_upstream_forcing": warmup_forcing,
            "spatial_order": spatial_order,
            "lateral_inflow": (
                "measured signed point flows"
                if point_flow_path is not None
                else (
                    "zero"
                    if lateral_fraction == 0.0
                    else "uniformly distributed fraction of observed upstream flow"
                )
            ),
            "lateral_inflow_fraction": lateral_fraction,
            "point_flow_series": (
                None
                if point_flow_path is None
                else str(point_flow_source)
            ),
            "point_flow_count": (
                0
                if point_flow_path is None
                else lateral_inflow.point_count
            ),
            "downstream_boundary": downstream_boundary,
            "downstream_stage_series": (
                None
                if downstream_stage_path is None
                else str(downstream_stage_source)
            ),
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
