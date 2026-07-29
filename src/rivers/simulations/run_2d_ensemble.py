"""Run reproducible 2-D uncertainty ensembles and save probabilistic outcomes.

The ensemble varies explicitly configured input scales with Latin-hypercube
sampling. It does not fit parameters or inspect observations, so it can be
applied after calibration without leaking held-out outcomes into parameter
selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from general.solvers.profile import (
    domain2d_from_profile,
    load_channel_geometry,
    load_profile,
    resample_profile,
)
from rivers.simulations.ingest_to_simulate import scenario_from_profile
from rivers.simulations.registry import dispatch
from rivers.simulations.run_simulation import (
    _forcing_value,
    _load_temporal_series,
    _portable_path,
)


PARAMETER_NAMES = (
    "manning_scale",
    "longitudinal_slope_scale",
    "channel_width_scale",
    "bankfull_depth_scale",
    "floodplain_slope_scale",
    "inflow_scale",
    "rainfall_scale",
    "downstream_stage_offset_m",
)
DEFAULT_BOUNDS = {
    name: (
        (0.0, 0.0)
        if name == "downstream_stage_offset_m"
        else (1.0, 1.0)
    )
    for name in PARAMETER_NAMES
}
SCHEMA_VERSION = 1


def load_ensemble_config(path):
    """Load and validate an uncertainty configuration JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Ensemble config schema_version must be {SCHEMA_VERSION}"
        )
    explicit_samples = data.get("parameter_samples")
    configured = data.get("parameter_scales", {})
    if not isinstance(configured, dict):
        raise ValueError("parameter_scales must be an object")
    if explicit_samples is not None and configured:
        raise ValueError(
            "Use parameter_scales or parameter_samples, not both"
        )
    unknown = set(configured).difference(PARAMETER_NAMES)
    if unknown:
        raise ValueError(f"Unknown uncertainty parameters: {sorted(unknown)}")
    bounds = {}
    for name in PARAMETER_NAMES:
        values = configured.get(name, DEFAULT_BOUNDS[name])
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError(f"{name} must be a [minimum, maximum] pair")
        try:
            lower, upper = (float(value) for value in values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} bounds must be numeric") from exc
        requires_positive = name != "downstream_stage_offset_m"
        if (
            not np.isfinite(lower)
            or not np.isfinite(upper)
            or (requires_positive and lower <= 0.0)
            or upper < lower
        ):
            raise ValueError(
                f"{name} bounds must be finite and ordered"
                + (", with a positive minimum" if requires_positive else "")
            )
        bounds[name] = (lower, upper)

    samples = None
    if explicit_samples is not None:
        if not isinstance(explicit_samples, list) or len(explicit_samples) < 2:
            raise ValueError(
                "parameter_samples must contain at least two joint samples"
            )
        rows = []
        for index, sample in enumerate(explicit_samples):
            if not isinstance(sample, dict):
                raise ValueError(f"parameter sample {index} must be an object")
            unknown = set(sample).difference(PARAMETER_NAMES)
            if unknown:
                raise ValueError(
                    f"Unknown parameters in sample {index}: {sorted(unknown)}"
                )
            row = []
            for name in PARAMETER_NAMES:
                default = (
                    0.0
                    if name == "downstream_stage_offset_m"
                    else 1.0
                )
                try:
                    value = float(sample.get(name, default))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{name} in sample {index} must be numeric"
                    ) from exc
                if not np.isfinite(value) or (
                    name != "downstream_stage_offset_m"
                    and value <= 0.0
                ):
                    raise ValueError(
                        f"{name} in sample {index} must be finite"
                        + (
                            " and positive"
                            if name != "downstream_stage_offset_m"
                            else ""
                        )
                    )
                row.append(value)
            rows.append(row)
        samples = np.asarray(rows, dtype=float)
        sample_count = len(samples)
        seed = None
        bounds = {
            name: (
                float(np.min(samples[:, column])),
                float(np.max(samples[:, column])),
            )
            for column, name in enumerate(PARAMETER_NAMES)
        }
        method = "explicit_joint_samples"
    else:
        sample_count = data.get("sample_count")
        seed = data.get("seed")
        if (
            not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or sample_count < 2
        ):
            raise ValueError("sample_count must be an integer of at least 2")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        method = "latin_hypercube"

    try:
        quantiles = np.asarray(
            data.get("quantiles", [0.05, 0.5, 0.95]), dtype=float
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("quantiles must be numeric") from exc
    if (
        quantiles.ndim != 1
        or len(quantiles) < 2
        or np.any(~np.isfinite(quantiles))
        or np.any(quantiles < 0.0)
        or np.any(quantiles > 1.0)
        or np.any(np.diff(quantiles) <= 0.0)
    ):
        raise ValueError(
            "quantiles must be a strictly increasing list in [0, 1]"
        )
    try:
        wet_threshold = float(data.get("wet_depth_threshold_m", 0.01))
    except (TypeError, ValueError) as exc:
        raise ValueError("wet_depth_threshold_m must be numeric") from exc
    if not np.isfinite(wet_threshold) or wet_threshold < 0.0:
        raise ValueError("wet_depth_threshold_m must be finite and non-negative")
    retain_member_depth = data.get("retain_member_depth", False)
    if not isinstance(retain_member_depth, bool):
        raise ValueError("retain_member_depth must be true or false")
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_count": sample_count,
        "seed": seed,
        "method": method,
        "bounds": bounds,
        "samples": samples,
        "quantiles": quantiles,
        "wet_depth_threshold_m": wet_threshold,
        "retain_member_depth": retain_member_depth,
    }


def sample_parameter_scales(config):
    """Return deterministic Latin-hypercube parameter scales."""
    if config.get("samples") is not None:
        return np.asarray(config["samples"], dtype=float).copy()
    sample_count = config["sample_count"]
    rng = np.random.default_rng(config["seed"])
    samples = np.empty((sample_count, len(PARAMETER_NAMES)), dtype=float)
    for column, name in enumerate(PARAMETER_NAMES):
        lower, upper = config["bounds"][name]
        strata = (np.arange(sample_count) + rng.random(sample_count)) / sample_count
        samples[:, column] = lower + (upper - lower) * rng.permutation(strata)
    return samples


def _scaled_forcing(forcing, scale):
    if callable(forcing):
        def scaled(time_min):
            return scale * float(forcing(time_min))

        scaled.breakpoints_min = getattr(forcing, "breakpoints_min", ())
        return scaled
    return scale * float(forcing)


def _scaled_rainfall(rainfall, scale):
    if rainfall is None:
        return None

    def scaled(x, time_min):
        return scale * np.asarray(rainfall(x, time_min), dtype=float)

    scaled.breakpoints_min = getattr(rainfall, "breakpoints_min", ())
    return scaled


def _offset_stage(stage, offset_m):
    if stage is None:
        if offset_m != 0.0:
            raise ValueError(
                "downstream_stage_offset_m uncertainty requires a "
                "downstream stage boundary"
            )
        return None
    if callable(stage):
        def shifted(time_min):
            return np.asarray(stage(time_min), dtype=float) + offset_m

        shifted.breakpoints_min = getattr(stage, "breakpoints_min", ())
        return shifted
    return np.asarray(stage, dtype=float) + offset_m


def _initialize_2d_scenario(scenario, domain, inflow):
    """Apply the runner's level-water and wet-width flow initialization."""
    longitudinal_depth = np.asarray(scenario.initial_depth_m, dtype=float)
    if longitudinal_depth.ndim == 0:
        longitudinal_depth = np.full(len(domain.x_m), float(longitudinal_depth))
    channel_bed = np.min(domain.bed_elevation_m, axis=1)
    water_surface = channel_bed + longitudinal_depth
    scenario.initial_depth_m = np.maximum(
        water_surface[:, None] - domain.bed_elevation_m,
        0.0,
    )
    wet = scenario.initial_depth_m > 0.0
    wet_width = np.sum(wet * domain.dy_m[None, :], axis=1)
    initial_flow = _forcing_value(inflow, 0.0)
    initial_unit_flow = np.zeros_like(scenario.initial_depth_m)
    active_rows = wet_width > 0.0
    initial_unit_flow[active_rows] = (
        wet[active_rows]
        * (initial_flow / wet_width[active_rows])[:, None]
    )
    scenario.initial_discharge = initial_unit_flow

    upstream_wet = wet[0]
    upstream_width = float(np.sum(domain.dy_m[upstream_wet]))
    if upstream_width <= 0.0 and initial_flow > 0.0:
        raise ValueError(
            "Positive 2-D inflow requires initially wet upstream cells"
        )

    def distributed_inflow(time_min):
        values = np.zeros(len(domain.y_m))
        if upstream_width > 0.0:
            values[upstream_wet] = _forcing_value(inflow, time_min) / upstream_width
        return values

    distributed_inflow.breakpoints_min = getattr(inflow, "breakpoints_min", ())
    scenario.left_inflow = distributed_inflow


def summarize_member_fields(
    member_depth,
    dx_m,
    dy_m,
    quantiles,
    wet_depth_threshold_m,
):
    """Aggregate member depth fields into probabilistic flood outcomes."""
    depth = np.asarray(member_depth, dtype=float)
    if depth.ndim != 4 or np.any(~np.isfinite(depth)) or np.any(depth < 0.0):
        raise ValueError("member_depth must have shape (member, time, x, y)")
    cell_area = np.asarray(dx_m)[:, None] * np.asarray(dy_m)[None, :]
    peak_depth = np.max(depth, axis=1)
    ever_wet = peak_depth > wet_depth_threshold_m
    wet_area_by_time = np.sum(
        (depth > wet_depth_threshold_m) * cell_area[None, None, :, :],
        axis=(2, 3),
    )
    maximum_wet_area = np.max(wet_area_by_time, axis=1)
    return {
        "depth_quantiles_m": np.quantile(depth, quantiles, axis=0),
        "peak_depth_quantiles_m": np.quantile(
            peak_depth, quantiles, axis=0
        ),
        "wet_probability": np.mean(ever_wet, axis=0),
        "member_maximum_wet_area_m2": maximum_wet_area,
        "maximum_wet_area_quantiles_m2": np.quantile(
            maximum_wet_area, quantiles
        ),
    }


def run_ensemble(
    profile,
    channel_width_m,
    bankfull_depth_m,
    config,
    *,
    domain_width_m,
    cross_cells,
    t_final_min,
    record_interval_min=1.0,
    left_inflow=0.0,
    rainfall_rate_m_per_min=0.0,
    temporal_rainfall=None,
    downstream_stage_m=None,
    floodplain_slope=0.02,
    cfl=0.45,
    spatial_order=1,
):
    """Run every sampled 2-D member and return fields plus diagnostics."""
    scales = sample_parameter_scales(config)
    member_depth = []
    member_bed = []
    member_mass_error = []
    member_times = None
    reference_domain = None

    base_scenario = scenario_from_profile(
        profile,
        t_final_min=t_final_min,
        record_interval_min=record_interval_min,
        left_inflow=left_inflow,
        rainfall_rate_m_per_min=rainfall_rate_m_per_min,
        cfl=cfl,
    )
    if temporal_rainfall is not None:
        profile_rainfall = base_scenario.rainfall

        def combined_rainfall(x, time_min):
            base = (
                np.zeros_like(x, dtype=float)
                if profile_rainfall is None
                else profile_rainfall(x, time_min)
            )
            return base + temporal_rainfall(time_min)

        combined_rainfall.breakpoints_min = temporal_rainfall.breakpoints_min
        base_scenario.rainfall = combined_rainfall

    for row in scales:
        parameters = dict(zip(PARAMETER_NAMES, row))
        member_profile = replace(
            profile,
            manning_n=profile.manning_n * parameters["manning_scale"],
            slope=(
                profile.slope
                * parameters["longitudinal_slope_scale"]
            ),
        )
        scaled_channel_width = (
            np.asarray(channel_width_m)
            * parameters["channel_width_scale"]
        )
        if np.any(scaled_channel_width >= domain_width_m):
            raise ValueError(
                "Sampled channel width reaches the 2-D domain boundary; "
                "increase domain_width_m or narrow channel_width_scale bounds"
            )
        domain = domain2d_from_profile(
            member_profile,
            domain_width_m,
            cross_cells,
            channel_width_m=scaled_channel_width,
            bankfull_depth_m=(
                np.asarray(bankfull_depth_m)
                * parameters["bankfull_depth_scale"]
            ),
            floodplain_slope=(
                floodplain_slope
                * parameters["floodplain_slope_scale"]
            ),
        )
        scenario = replace(
            base_scenario,
            rainfall=_scaled_rainfall(
                base_scenario.rainfall, parameters["rainfall_scale"]
            ),
            boundary_x=(
                "inflow_stage"
                if downstream_stage_m is not None
                else "inflow_outflow"
            ),
            downstream_stage_m=_offset_stage(
                downstream_stage_m,
                parameters["downstream_stage_offset_m"],
            ),
            spatial_order=spatial_order,
        )
        member_inflow = _scaled_forcing(
            left_inflow, parameters["inflow_scale"]
        )
        _initialize_2d_scenario(scenario, domain, member_inflow)
        result = dispatch("saint_venant_2d", domain, scenario)
        if member_times is None:
            member_times = result.times
            reference_domain = domain
        elif not np.array_equal(member_times, result.times):
            raise RuntimeError("Ensemble members produced inconsistent record times")
        member_depth.append(result.depth_history)
        member_bed.append(domain.bed_elevation_m)
        cell_area = domain.dx_m[:, None] * domain.dy_m[None, :]
        storage_change = float(
            np.sum(
                (result.depth_final - result.depth_initial) * cell_area
            )
        )
        member_mass_error.append(
            result.mass_inflow
            + result.mass_source
            + result.mass_correction
            - result.mass_outflow
            - storage_change
        )

    member_depth = np.asarray(member_depth)
    member_bed = np.asarray(member_bed)
    aggregate = summarize_member_fields(
        member_depth,
        reference_domain.dx_m,
        reference_domain.dy_m,
        config["quantiles"],
        config["wet_depth_threshold_m"],
    )
    return {
        "parameter_names": PARAMETER_NAMES,
        "parameter_scales": scales,
        "quantiles": config["quantiles"],
        "times_min": member_times,
        "x_m": reference_domain.x_m,
        "y_m": reference_domain.y_m,
        "dx_m": reference_domain.dx_m,
        "dy_m": reference_domain.dy_m,
        "member_depth_m": (
            member_depth if config["retain_member_depth"] else None
        ),
        "bed_elevation_quantiles_m": np.quantile(
            member_bed, config["quantiles"], axis=0
        ),
        "water_surface_elevation_quantiles_m": np.quantile(
            member_depth + member_bed[:, None, :, :],
            config["quantiles"],
            axis=0,
        ),
        "peak_water_surface_elevation_quantiles_m": np.quantile(
            np.max(
                member_depth + member_bed[:, None, :, :],
                axis=1,
            ),
            config["quantiles"],
            axis=0,
        ),
        "member_mass_balance_error_m3": np.asarray(member_mass_error),
        **aggregate,
    }


def save_ensemble(result, config, output_prefix, context):
    """Save compressed numeric evidence and a readable provenance summary."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fields_path = prefix.with_name(prefix.name + "_ensemble.npz")
    arrays = {
        key: value
        for key, value in result.items()
        if isinstance(value, np.ndarray)
    }
    arrays["parameter_names"] = np.asarray(result["parameter_names"])
    np.savez_compressed(fields_path, **arrays)

    area_quantiles = result["maximum_wet_area_quantiles_m2"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "method": config["method"],
        "sample_count": config["sample_count"],
        "seed": config["seed"],
        "quantiles": config["quantiles"].tolist(),
        "wet_depth_threshold_m": config["wet_depth_threshold_m"],
        "parameter_scales": {
            name: list(config["bounds"][name])
            for name in PARAMETER_NAMES
        },
        "maximum_wet_area_quantiles_m2": {
            str(probability): float(value)
            for probability, value in zip(config["quantiles"], area_quantiles)
        },
        "maximum_absolute_mass_balance_error_m3": float(
            np.max(np.abs(result["member_mass_balance_error_m3"]))
        ),
        "fields_path": str(fields_path),
        "context": context,
        "interpretation": (
            "These probabilities are conditional on the configured input "
            "ranges and model structure; they are not calibrated forecast probabilities."
        ),
    }
    summary_path = prefix.with_name(prefix.name + "_ensemble_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return fields_path, summary_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a reproducible 2-D hydraulic uncertainty ensemble."
    )
    parser.add_argument("profile", type=Path)
    parser.add_argument("--hydraulic-geometry", type=Path, required=True)
    parser.add_argument("--ensemble-config", type=Path, required=True)
    parser.add_argument("--width", type=float, required=True)
    parser.add_argument("--cross-cells", type=int, default=20)
    parser.add_argument("--longitudinal-cells", type=int)
    parser.add_argument("--t-final", type=float, required=True)
    parser.add_argument("--record-interval", type=float, default=1.0)
    parser.add_argument("--left-inflow", type=float, default=0.0)
    parser.add_argument("--inflow-series", type=Path)
    parser.add_argument("--rainfall-rate", type=float, default=0.0)
    parser.add_argument("--rainfall-series", type=Path)
    parser.add_argument("--downstream-stage", type=float)
    parser.add_argument("--downstream-stage-series", type=Path)
    parser.add_argument("--floodplain-slope", type=float, default=0.02)
    parser.add_argument("--cfl", type=float, default=0.45)
    parser.add_argument("--spatial-order", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("data/ensembles"))
    parser.add_argument("--run-name", default="uncertainty")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.inflow_series is not None and args.left_inflow != 0.0:
        raise SystemExit(
            "error: use either --left-inflow or --inflow-series, not both"
        )
    if (
        args.downstream_stage is not None
        and args.downstream_stage_series is not None
    ):
        raise SystemExit(
            "error: use either --downstream-stage or "
            "--downstream-stage-series, not both"
        )
    for path in (
        args.profile,
        args.hydraulic_geometry,
        args.ensemble_config,
        args.inflow_series,
        args.rainfall_series,
        args.downstream_stage_series,
    ):
        if path is not None and not path.is_file():
            raise SystemExit(f"error: input does not exist: {path}")
    try:
        config = load_ensemble_config(args.ensemble_config)
        profile = load_profile(args.profile)
        if args.longitudinal_cells is not None:
            profile = resample_profile(profile, args.longitudinal_cells)
        channel_width, bankfull_depth = load_channel_geometry(
            args.hydraulic_geometry, profile.station_m
        )
        inflow = (
            args.left_inflow
            if args.inflow_series is None
            else _load_temporal_series(args.inflow_series, "left_inflow")
        )
        temporal_rainfall = (
            None
            if args.rainfall_series is None
            else _load_temporal_series(
                args.rainfall_series, "rainfall_rate_m_per_min"
            )
        )
        downstream_stage = (
            args.downstream_stage
            if args.downstream_stage_series is None
            else _load_temporal_series(
                args.downstream_stage_series,
                "downstream_stage_m",
                allow_negative=True,
            )
        )
        result = run_ensemble(
            profile,
            channel_width,
            bankfull_depth,
            config,
            domain_width_m=args.width,
            cross_cells=args.cross_cells,
            t_final_min=args.t_final,
            record_interval_min=args.record_interval,
            left_inflow=inflow,
            rainfall_rate_m_per_min=args.rainfall_rate,
            temporal_rainfall=temporal_rainfall,
            downstream_stage_m=downstream_stage,
            floodplain_slope=args.floodplain_slope,
            cfl=args.cfl,
            spatial_order=args.spatial_order,
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    context = {
        "profile": _portable_path(args.profile),
        "hydraulic_geometry": _portable_path(args.hydraulic_geometry),
        "ensemble_config": _portable_path(args.ensemble_config),
        "domain_width_m": args.width,
        "cross_cells": args.cross_cells,
        "longitudinal_cells": args.longitudinal_cells,
        "t_final_min": args.t_final,
        "record_interval_min": args.record_interval,
        "left_inflow": args.left_inflow if args.inflow_series is None else None,
        "inflow_series": (
            None
            if args.inflow_series is None
            else _portable_path(args.inflow_series)
        ),
        "rainfall_rate_m_per_min": args.rainfall_rate,
        "rainfall_series": (
            None
            if args.rainfall_series is None
            else _portable_path(args.rainfall_series)
        ),
        "downstream_stage_m": (
            args.downstream_stage
            if args.downstream_stage_series is None
            else None
        ),
        "downstream_stage_series": (
            None
            if args.downstream_stage_series is None
            else _portable_path(args.downstream_stage_series)
        ),
        "floodplain_slope": args.floodplain_slope,
        "cfl": args.cfl,
        "spatial_order": args.spatial_order,
    }
    fields_path, summary_path = save_ensemble(
        result,
        config,
        args.output_dir / args.run_name,
        context,
    )
    print(f"Done. Ensemble: {fields_path}  Summary: {summary_path}")


if __name__ == "__main__":
    main()
