"""Run an observed event through the 2-D solver on an explicit screening terrain.

The committed observations do not include continuous topobathymetry.  This
runner therefore supports only auditable idealized corridor representations:

``ribbon``
    A laterally uniform, wall-bounded channel.  This tests the 2-D numerical
    pathway without claiming floodplain geometry.

``shelf``
    A central channel with raised lateral shelves.  This tests sensitivity to
    lateral storage/connectivity, not a mapped inundation prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from general.solvers import saint_venant_2d
from rivers.validation.compare import evaluate_series
from rivers.validation.skill import benchmark_skill, skill_scores
from rivers.validation.datasets import validate_case_policy
from rivers.validation.diagnose import diagnose_hydrograph, diagnose_reach_routing
from rivers.validation.case_inputs import (
    _configured_path,
    _require_control_coverage,
    _timestamp,
    discharge_boundary,
    load_event_control_series,
    load_event_point_flows,
    load_two_gauge_observations,
    load_validation_geometry,
    shifted_boundary,
)


def _bed_from_slope(x_m, slope):
    bed = np.zeros_like(x_m)
    if len(x_m) > 1:
        bed[1:] = -np.cumsum(
            0.5 * (slope[:-1] + slope[1:]) * np.diff(x_m)
        )
    return bed


def _rectangular_normal_depth(discharge, width, manning_n, slope):
    """Solve whole-channel Manning flow without invoking a 1-D solver."""
    if discharge <= 0.0:
        return 0.0

    def flow(depth):
        area = width * depth
        radius = area / (width + 2.0 * depth)
        return area * radius ** (2.0 / 3.0) * np.sqrt(slope) / manning_n

    lower, upper = 0.0, 1.0
    while flow(upper) < discharge:
        upper *= 2.0
        if upper > 1e5:
            raise ValueError("Could not bracket 2-D initial normal depth")
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if flow(middle) < discharge:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def _whole_flow_boundary(boundary, mask, dy):
    wet_width = float(np.sum(dy[mask]))
    if wet_width <= 0:
        raise ValueError("2-D upstream terrain has no active channel cells")

    def forcing(time):
        values = np.zeros(len(dy))
        values[mask] = float(boundary(time)) / wet_width
        return values

    if hasattr(boundary, "breakpoints_min"):
        forcing.breakpoints_min = boundary.breakpoints_min
    return forcing


def _point_flows_2d(point_flows, channel, dy):
    """Distribute each whole-flow hydrograph over its mapped wet channel row."""
    channel = np.asarray(channel, dtype=bool)
    dy = np.asarray(dy, dtype=float)
    row_widths = np.sum(channel * dy[None, :], axis=1)
    if np.any(row_widths <= 0.0):
        raise ValueError("Every point-flow row needs at least one channel cell")

    def forcing(x, y, time):
        del y
        per_length = np.asarray(point_flows(x, time), dtype=float)
        if per_length.shape != (len(x),):
            raise ValueError("Mapped point flows must return one rate per x cell")
        rates = np.zeros(channel.shape, dtype=float)
        rates[channel] = np.broadcast_to(
            (per_length / row_widths)[:, None], channel.shape
        )[channel]
        return rates

    for attribute in (
        "breakpoints_min",
        "coverage_min",
        "series_coverage_min",
        "point_count",
    ):
        if hasattr(point_flows, attribute):
            setattr(forcing, attribute, getattr(point_flows, attribute))
    return forcing


def _shifted_2d_forcing(forcing, offset_min):
    """Shift an ``f(x, y, time)`` forcing onto a warm-up clock."""
    offset = float(offset_min)

    def shifted(x, y, time):
        return forcing(x, y, float(time) + offset)

    shifted.breakpoints_min = tuple(
        float(value - offset)
        for value in getattr(forcing, "breakpoints_min", ())
    )
    return shifted


def _terrain(
    x_m,
    width_m,
    bed_center_m,
    normal_depth_m,
    *,
    representation,
    y_cells,
    floodplain_width_factor,
    bank_height_factor,
):
    if representation not in {"ribbon", "shelf"}:
        raise ValueError("terrain representation must be 'ribbon' or 'shelf'")
    if y_cells < 1:
        raise ValueError("y_cells must be positive")
    reference_width = float(np.max(width_m))
    total_width = (
        reference_width
        if representation == "ribbon"
        else floodplain_width_factor * reference_width
    )
    dy = np.full(y_cells, total_width / y_cells)
    y_m = (np.arange(y_cells) + 0.5) * dy[0] - 0.5 * total_width
    if representation == "ribbon":
        channel = np.ones((len(x_m), y_cells), dtype=bool)
        bank_height = 0.0
    else:
        channel = np.abs(y_m[None, :]) <= 0.5 * width_m[:, None]
        channel[:, y_cells // 2] = True
        bank_height = bank_height_factor * float(np.median(normal_depth_m))
    bed = bed_center_m[:, None] + np.where(channel, 0.0, bank_height)
    return y_m, dy, bed, channel, bank_height


def run_validation_case_2d(
    config_path,
    *,
    representation="ribbon",
    x_cells=None,
    y_cells=3,
    floodplain_width_factor=3.0,
    bank_height_factor=1.25,
    output_path=None,
):
    """Run one fixed observed case through the 2-D screening pathway."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hydraulic_dataset = validate_case_policy(config)
    observations_path = _configured_path(config_path, config["observations"])
    observations = load_two_gauge_observations(observations_path)
    upstream_times, upstream_flow = observations["upstream"]
    downstream_times, downstream_flow = observations["downstream"]

    reach = config["reach"]
    if reach.get("cross_section_shape", "rectangular") != "rectangular":
        raise ValueError(
            "2-D screening validation currently supports rectangular source "
            "sections only; converting compound/surveyed curves into terrain "
            "requires reviewed topobathymetric preprocessing"
        )
    length_m = float(reach["length_m"])
    cells = int(x_cells or reach["cells"])
    if cells < 2:
        raise ValueError("2-D validation needs at least two longitudinal cells")
    x_m = np.linspace(0.0, length_m, cells)
    dx_m = np.full(cells, length_m / cells)
    slope = np.full(cells, float(reach["slope"]))
    roughness = np.full(cells, float(reach["manning_n"]))
    event_start = _timestamp(config["case"]["observation_window"][0])
    geometry, geometry_provenance = load_validation_geometry(
        config_path, reach, x_m, event_start=event_start
    )
    width = np.asarray(geometry["channel_width_m"], dtype=float)
    if geometry["bed_elevation_m"] is None:
        bed_center = _bed_from_slope(x_m, slope)
    else:
        bed_center = np.asarray(geometry["bed_elevation_m"], dtype=float)
        slope = -np.gradient(bed_center, x_m)
    if geometry["field_manning_n"] is not None:
        roughness = np.asarray(geometry["field_manning_n"], dtype=float)
    if np.any(slope <= 0.0):
        raise ValueError("2-D screening terrain requires positive longitudinal slope")

    duration = min(float(upstream_times[-1]), float(downstream_times[-1]))
    boundary = discharge_boundary(upstream_times, upstream_flow)
    warmup_config = config.get("warmup", {})
    warmup_min = float(
        warmup_config.get("duration_min", config.get("warmup_min", 0.0))
    )
    warmup_start = -warmup_min
    warmup_forcing = warmup_config.get("upstream_forcing", "constant_initial")
    initial_q = boundary(warmup_start if warmup_forcing == "observed" else 0.0)
    normal_depth = np.asarray(
        [
            _rectangular_normal_depth(initial_q, width[i], roughness[i], slope[i])
            for i in range(cells)
        ]
    )
    y_m, dy_m, bed, channel, bank_height = _terrain(
        x_m,
        width,
        bed_center,
        normal_depth,
        representation=representation,
        y_cells=int(y_cells),
        floodplain_width_factor=float(floodplain_width_factor),
        bank_height_factor=float(bank_height_factor),
    )
    surface = bed_center[:, None] + normal_depth[:, None]
    initial_depth = np.maximum(surface - bed, 0.0)
    initial_hu = np.zeros_like(initial_depth)
    for i in range(cells):
        wet_width = float(np.sum(dy_m[channel[i]]))
        initial_hu[i, channel[i]] = initial_q / wet_width
    initial_hv = np.zeros_like(initial_depth)
    boundary_2d = _whole_flow_boundary(boundary, channel[0], dy_m)

    stage_source = config.get("downstream_stage_series")
    if stage_source is None:
        boundary_x = "inflow_outflow"
        downstream_stage = None
    else:
        stage_path = _configured_path(config_path, stage_source)
        downstream_stage = load_event_control_series(
            stage_path, "downstream_stage_m", event_start
        )
        _require_control_coverage(
            downstream_stage, warmup_start, duration, "downstream_stage_series"
        )
        boundary_x = "inflow_stage"
    point_flow_source = config.get("point_flow_series")
    if point_flow_source is None:
        point_flow_path = None
        lateral_inflow = None
    else:
        point_flow_path = _configured_path(config_path, point_flow_source)
        if not point_flow_path.is_file():
            raise ValueError(
                f"Point-flow control does not exist: {point_flow_path}"
            )
        point_flows = load_event_point_flows(
            point_flow_path, x_m, dx_m, event_start
        )
        _require_control_coverage(
            point_flows,
            warmup_start,
            duration,
            "point_flow_series",
        )
        lateral_inflow = _point_flows_2d(point_flows, channel, dy_m)
    if float(config.get("lateral_inflow_fraction", 0.0)) != 0.0:
        raise ValueError("Calibrated/uniform lateral gain is forbidden in 2-D validation")

    slope_x = np.broadcast_to(slope[:, None], bed.shape).copy()
    slope_y = np.zeros_like(bed)
    manning = np.broadcast_to(roughness[:, None], bed.shape).copy()
    common = {
        "x_m": x_m,
        "y_m": y_m,
        "dx_m": dx_m,
        "dy_m": dy_m,
        "slope_x": slope_x,
        "slope_y": slope_y,
        "manning_n": manning,
        "bed_elevation_m": bed,
        "rainfall": lambda x, y, time: np.zeros((len(x), len(y))),
        "cfl": min(float(config.get("cfl", 0.4)), 0.5),
        "spatial_order": int(config.get("spatial_order", 1)),
        "boundary_x": boundary_x,
        "boundary_y": "wall",
    }

    if warmup_min > 0.0:
        warmup_boundary = (
            shifted_boundary(boundary, warmup_start)
            if warmup_forcing == "observed"
            else lambda time: initial_q
        )
        warmup_stage = (
            None
            if downstream_stage is None
            else shifted_boundary(downstream_stage, warmup_start)
        )
        warmup_lateral = (
            None
            if lateral_inflow is None
            else _shifted_2d_forcing(lateral_inflow, warmup_start)
        )
        warmup = saint_venant_2d.run_model(
            T_final=warmup_min,
            record_interval=warmup_min,
            h_init=initial_depth,
            hu_init=initial_hu,
            hv_init=initial_hv,
            left_inflow=_whole_flow_boundary(
                warmup_boundary, channel[0], dy_m
            ),
            lateral_inflow=warmup_lateral,
            downstream_stage_m=warmup_stage,
            **common,
        )
        initial_depth = warmup["h_final"]
        initial_hu = warmup["hu_final"]
        initial_hv = warmup["hv_final"]

    result = saint_venant_2d.run_model(
        T_final=duration,
        record_interval=float(config.get("record_interval_min", 5.0)),
        h_init=initial_depth,
        hu_init=initial_hu,
        hv_init=initial_hv,
        left_inflow=boundary_2d,
        lateral_inflow=lateral_inflow,
        downstream_stage_m=downstream_stage,
        **common,
    )
    target = (downstream_times >= 0.0) & (downstream_times <= duration)
    target_times = downstream_times[target]
    target_flow = downstream_flow[target]
    scores = evaluate_series(
        target_times,
        target_flow,
        result["times"],
        result["downstream_flux_history"],
    )
    predicted = scores["predicted_on_obs"]
    upstream_on_target = np.interp(target_times, upstream_times, upstream_flow)
    upstream_passthrough_scores = skill_scores(target_flow, upstream_on_target)
    reach_diagnosis = diagnose_reach_routing(
        target_times, upstream_on_target, target_flow, predicted
    )
    reach_diagnosis["routing_lag_comparable"] = lateral_inflow is None
    if lateral_inflow is not None:
        reach_diagnosis["routing_lag_limitation"] = (
            "Modeled downstream timing reflects both upstream routing and "
            "internal-source timing, so a single lag relative to the mainstem "
            "upstream gauge is not a like-for-like travel-time metric."
        )
    cell_area = np.asarray(result["dx_m"], dtype=float)[:, None] * np.asarray(
        result["dy_m"], dtype=float
    )[None, :]
    storage_initial = float(np.sum(result["h_initial"] * cell_area))
    storage_final = float(np.sum(result["h_final"] * cell_area))
    expected_storage_change = float(
        result["mass_inflow"]
        + result["mass_source"]
        - result["mass_outflow"]
        + result["mass_floor_correction"]
    )
    mass_residual = storage_final - storage_initial - expected_storage_change
    mass_scale = max(
        abs(storage_initial),
        abs(storage_final),
        abs(expected_storage_change),
        1.0,
    )
    evidence = {
        "schema_version": 1,
        "case": config["case"],
        "solver": "saint_venant_2d",
        "status": config.get(
            "validation_status", "uncalibrated_2d_screening"
        ),
        "solver_policy": "saint_venant_2d_only",
        "validation_policy": config["validation_policy"],
        "hydraulic_dataset": hydraulic_dataset,
        "reach_context": config.get("reach_context"),
        "observations": {
            "path": str(observations_path),
            "upstream_count": int(len(upstream_times)),
            "downstream_count": int(len(target_times)),
            "duration_min": duration,
        },
        "terrain_representation": {
            "name": representation,
            "x_cells": cells,
            "y_cells": int(y_cells),
            "domain_width_m": float(np.sum(dy_m)),
            "bank_height_m": bank_height,
            "floodplain_width_factor": float(floodplain_width_factor),
            "configured_channel_width_m_range": {
                "minimum": float(np.min(width)),
                "maximum": float(np.max(width)),
            },
            "effective_channel_width_m_range": {
                "minimum": float(
                    np.min(np.sum(channel * dy_m[None, :], axis=1))
                ),
                "maximum": float(
                    np.max(np.sum(channel * dy_m[None, :], axis=1))
                ),
            },
            "provenance": geometry_provenance,
            "limitation": (
                "Laterally uniform constant-width numerical ribbon; configured "
                "longitudinal width variation is not represented and this is "
                "not mapped flood terrain."
                if representation == "ribbon"
                else "Idealized raised shelves; tests lateral storage sensitivity "
                "but is not observed topobathymetry."
            ),
        },
        "boundary": {
            "x": boundary_x,
            "downstream_score_observable": (
                "finite-volume 2-D downstream boundary discharge flux"
            ),
            "downstream_stage_series": stage_source,
            "point_flow_series": point_flow_source,
            "point_flow_count": (
                0 if lateral_inflow is None else lateral_inflow.point_count
            ),
            "point_flow_mapping": (
                None
                if lateral_inflow is None
                else (
                    "Nearest longitudinal cell; whole discharge divided by "
                    "cell length and active channel width. Positive flow enters "
                    "without assumed momentum; withdrawals remove local momentum."
                )
            ),
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
        "benchmarks": {
            "upstream_passthrough": {
                key: (
                    int(value)
                    if key == "n"
                    else None
                    if not math.isfinite(float(value))
                    else float(value)
                )
                for key, value in upstream_passthrough_scores.items()
            },
            "squared_error_skill_over_upstream_passthrough": (
                None
                if not math.isfinite(
                    value_added := benchmark_skill(
                        target_flow, predicted, upstream_on_target
                    )
                )
                else float(value_added)
            ),
            "interpretation": (
                "Positive skill means routing reduces squared error relative "
                "to using the simultaneous upstream gauge unchanged; zero or "
                "negative skill means the hydraulic model adds no value over "
                "that naive boundary-hydrograph benchmark for this event."
            ),
        },
        "error_diagnosis": diagnose_hydrograph(
            target_times, target_flow, predicted
        ),
        "reach_diagnosis": reach_diagnosis,
        "mass": {
            "storage_initial_m3": storage_initial,
            "storage_final_m3": storage_final,
            "storage_change_m3": storage_final - storage_initial,
            "inflow_m3": float(result["mass_inflow"]),
            "source_m3": float(result["mass_source"]),
            "rainfall_m3": float(result["mass_rainfall"]),
            "lateral_inflow_m3": float(result["mass_lateral_inflow"]),
            "lateral_requested_m3": float(
                result["mass_lateral_requested"]
            ),
            "unmet_withdrawal_m3": float(
                result["mass_unmet_withdrawal"]
            ),
            "outflow_m3": float(result["mass_outflow"]),
            "floor_correction_m3": float(result["mass_floor_correction"]),
            "balance_residual_m3": mass_residual,
            "relative_balance_residual": mass_residual / mass_scale,
        },
        "series": {
            "times_min": target_times.tolist(),
            "observed_downstream_m3_per_min": target_flow.tolist(),
            "predicted_downstream_m3_per_min": predicted.tolist(),
        },
    }
    destination = (
        Path(output_path)
        if output_path is not None
        else config_path.with_name(
            f"{config_path.stem}.{representation}.2d.results.json"
        )
    )
    destination.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one observed case through an explicit 2-D screening terrain."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--representation", choices=("ribbon", "shelf"), default="ribbon")
    parser.add_argument("--x-cells", type=int)
    parser.add_argument("--y-cells", type=int, default=3)
    parser.add_argument("--floodplain-width-factor", type=float, default=3.0)
    parser.add_argument("--bank-height-factor", type=float, default=1.25)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    evidence = run_validation_case_2d(
        args.config,
        representation=args.representation,
        x_cells=args.x_cells,
        y_cells=args.y_cells,
        floodplain_width_factor=args.floodplain_width_factor,
        bank_height_factor=args.bank_height_factor,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "case": evidence["case"]["name"],
                "terrain": evidence["terrain_representation"],
                "scores": evidence["scores"],
                "reach_diagnosis": evidence["reach_diagnosis"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
