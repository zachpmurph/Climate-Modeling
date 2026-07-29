import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from general.solvers.contract import Domain2D
from general.solvers.profile import (
    domain2d_from_profile,
    domain_from_profile,
    load_channel_geometry,
    load_profile,
)
from rivers.simulations.ingest_to_simulate import scenario_from_profile
from rivers.simulations.registry import SOLVERS, dispatch


DEFAULT_OUTPUT_DIR = Path("data") / "real_world_rivers" / "runs"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _portable_path(path):
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _load_temporal_series(path, value_column):
    """Load a linearly interpolated forcing series with constant end values."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError(f"{path} must contain at least two forcing rows")
    try:
        times = np.asarray([float(row["t_min"]) for row in rows])
        values = np.asarray([float(row[value_column]) for row in rows])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{path} must contain numeric t_min and {value_column} columns"
        ) from exc
    if (
        np.any(~np.isfinite(times))
        or np.any(~np.isfinite(values))
        or np.any(np.diff(times) <= 0)
        or times[0] != 0.0
        or np.any(values < 0)
    ):
        raise ValueError(
            f"{path} needs t_min starting at 0 and strictly increasing, "
            f"with finite non-negative {value_column}"
        )

    def forcing(time_min):
        return float(np.interp(time_min, times, values))

    forcing.breakpoints_min = times.copy()
    return forcing


def _forcing_value(forcing, time_min):
    value = forcing(time_min) if callable(forcing) else forcing
    return float(value)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run a 1-D or 2-D river solver on a profile.")
    p.add_argument("profile", help="CSV or JSON river profile path")
    p.add_argument(
        "--solver",
        choices=sorted(SOLVERS),
        default="saint_venant",
        help="Which solver to use",
    )
    p.add_argument("--t-final", type=float, required=True, help="Simulation duration in minutes")
    p.add_argument("--record-interval", type=float, default=1.0)
    p.add_argument(
        "--left-inflow",
        type=float,
        default=0.0,
        help=(
            "Constant upstream flow: m^3/min with --hydraulic-geometry, "
            "legacy unit-width m^2/min otherwise"
        ),
    )
    p.add_argument(
        "--inflow-series",
        type=Path,
        help="CSV with t_min,left_inflow for a time-varying upstream hydrograph",
    )
    p.add_argument("--rainfall-rate", type=float, default=0.0, help="Uniform rainfall rate, m/min")
    p.add_argument(
        "--rainfall-series",
        type=Path,
        help="CSV with t_min,rainfall_rate_m_per_min for a uniform time-varying storm",
    )
    p.add_argument("--cfl", type=float, default=0.5)
    p.add_argument(
        "--width",
        type=float,
        help="Channel width in metres (required by saint_venant_2d)",
    )
    p.add_argument(
        "--cross-cells",
        type=int,
        default=10,
        help="Number of cells across the channel for saint_venant_2d (default: 10)",
    )
    p.add_argument(
        "--hydraulic-geometry",
        type=Path,
        help=(
            "Reviewed station/width/bankfull CSV; enables physical 1-D "
            "cross-sections and is required for a terrain-backed 2-D run"
        ),
    )
    p.add_argument(
        "--floodplain-slope",
        type=float,
        default=0.02,
        help="Lateral rise/run outside the reviewed channel width (default: 0.02)",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--run-name", default="simulation")
    p.add_argument(
        "--map-markers",
        type=Path,
        help="Ordered centerline/marker CSV to record for geographic visualization",
    )
    p.add_argument(
        "--map-geometry",
        type=Path,
        help="Channel geometry CSV to record for geographic visualization",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if (args.map_markers is None) != (args.map_geometry is None):
        raise SystemExit("error: --map-markers and --map-geometry must be supplied together")
    for map_path in (args.map_markers, args.map_geometry):
        if map_path is not None and not map_path.is_file():
            raise SystemExit(f"error: map input does not exist: {map_path}")
    if args.inflow_series is not None and args.left_inflow != 0.0:
        raise SystemExit("error: use either --left-inflow or --inflow-series, not both")
    for forcing_path in (args.inflow_series, args.rainfall_series):
        if forcing_path is not None and not forcing_path.is_file():
            raise SystemExit(f"error: forcing input does not exist: {forcing_path}")

    try:
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
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    profile = load_profile(args.profile)
    if args.solver == "saint_venant_2d":
        if args.width is None:
            raise SystemExit("error: --width is required for saint_venant_2d")
        if args.hydraulic_geometry is None:
            raise SystemExit(
                "error: --hydraulic-geometry is required for saint_venant_2d"
            )
        if not args.hydraulic_geometry.is_file():
            raise SystemExit(
                f"error: hydraulic geometry does not exist: {args.hydraulic_geometry}"
            )
        channel_width, bankfull_depth = load_channel_geometry(
            args.hydraulic_geometry, profile.station_m
        )
        domain = domain2d_from_profile(
            profile,
            args.width,
            args.cross_cells,
            channel_width_m=channel_width,
            bankfull_depth_m=bankfull_depth,
            floodplain_slope=args.floodplain_slope,
        )
    else:
        if args.width is not None:
            raise SystemExit(
                "error: --width is only valid with saint_venant_2d"
            )
        if args.hydraulic_geometry is None:
            domain = domain_from_profile(profile)
        else:
            if not args.hydraulic_geometry.is_file():
                raise SystemExit(
                    f"error: hydraulic geometry does not exist: {args.hydraulic_geometry}"
                )
            channel_width, bankfull_depth = load_channel_geometry(
                args.hydraulic_geometry, profile.station_m
            )
            domain = domain_from_profile(
                profile,
                channel_width_m=channel_width,
                bankfull_depth_m=bankfull_depth,
            )

    scenario = scenario_from_profile(
        profile,
        t_final_min=args.t_final,
        record_interval_min=args.record_interval,
        left_inflow=inflow,
        rainfall_rate_m_per_min=args.rainfall_rate,
        cfl=args.cfl,
    )
    if temporal_rainfall is not None:
        base_rainfall = scenario.rainfall

        def combined_rainfall(x, time):
            base = (
                np.zeros_like(x, dtype=float)
                if base_rainfall is None
                else base_rainfall(x, time)
            )
            return base + temporal_rainfall(time)

        combined_rainfall.breakpoints_min = temporal_rainfall.breakpoints_min
        scenario.rainfall = combined_rainfall

    if isinstance(domain, Domain2D):
        channel_depth = (
            np.zeros(len(profile.station_m))
            if profile.initial_depth_m is None
            else np.asarray(profile.initial_depth_m, dtype=float)
        )
        channel_bed = np.min(domain.bed_elevation_m, axis=1)
        water_surface = channel_bed + channel_depth
        scenario.initial_depth_m = np.maximum(
            water_surface[:, None] - domain.bed_elevation_m,
            0.0,
        )
        wet = scenario.initial_depth_m > 0.0
        wet_width = np.sum(wet * domain.dy_m[None, :], axis=1)
        inflow_at_zero = _forcing_value(inflow, 0.0)
        initial_unit_flow = np.zeros_like(scenario.initial_depth_m)
        active_rows = wet_width > 0.0
        initial_unit_flow[active_rows] = (
            wet[active_rows]
            * (inflow_at_zero / wet_width[active_rows])[:, None]
        )
        scenario.initial_discharge = initial_unit_flow

        upstream_wet = wet[0]
        upstream_width = float(np.sum(domain.dy_m[upstream_wet]))
        if upstream_width <= 0.0 and inflow_at_zero > 0.0:
            raise SystemExit(
                "error: positive 2-D inflow needs at least one initially wet upstream cell"
            )

        def distributed_inflow(time):
            values = np.zeros(len(domain.y_m))
            if upstream_width > 0.0:
                values[upstream_wet] = _forcing_value(inflow, time) / upstream_width
            return values

        if hasattr(inflow, "breakpoints_min"):
            distributed_inflow.breakpoints_min = inflow.breakpoints_min
        scenario.left_inflow = distributed_inflow
    elif args.solver == "saint_venant":
        scenario.initial_discharge = np.full(
            len(domain.x_m), _forcing_value(inflow, 0.0)
        )

    result = dispatch(args.solver, domain, scenario)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    is_2d = isinstance(result.domain, Domain2D)

    # Keep a longitudinal CSV for existing animation/reporting. For a 2-D run
    # this is an area-weighted cross-channel mean, while the NPZ below retains
    # the complete field.
    csv_path = out / f"{args.run_name}_timeseries.csv"
    csv_depth = result.depth_history
    if is_2d:
        csv_depth = np.average(csv_depth, axis=2, weights=result.domain.dy_m)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_min"] + [f"{x:.6f}" for x in result.domain.x_m])
        for t, row in zip(result.times, csv_depth):
            writer.writerow([f"{t:.6f}"] + [f"{d:.10g}" for d in row])

    fields_path = None
    if is_2d:
        fields_path = out / f"{args.run_name}_fields.npz"
        np.savez_compressed(
            fields_path,
            x_m=result.domain.x_m,
            y_m=result.domain.y_m,
            dx_m=result.domain.dx_m,
            dy_m=result.domain.dy_m,
            bed_elevation_m=result.extra["bed_elevation_m"],
            times_min=result.times,
            depth_m=result.depth_history,
            depth_initial_m=result.depth_initial,
            depth_final_m=result.depth_final,
            discharge_x_m2_per_min=result.extra["discharge_x_history"],
            discharge_y_m2_per_min=result.extra["discharge_y_history"],
            discharge_x_final=result.extra["discharge_x_final"],
            discharge_y_final=result.extra["discharge_y_final"],
        )

    # Mass balance error
    cell_measure = (
        result.domain.dx_m[:, None] * result.domain.dy_m[None, :]
        if is_2d
        else (
            result.domain.dx_m
            if result.domain.channel_width_m is None
            else result.domain.dx_m * result.domain.channel_width_m
        )
    )
    physical_volume = is_2d or (
        not is_2d and result.domain.channel_width_m is not None
    )
    mass_balance_error = (
        result.mass_inflow + result.mass_source + result.mass_correction - result.mass_outflow
        - float(np.sum((result.depth_final - result.depth_initial) * cell_measure))
    )

    summary = {
        "solver": args.solver,
        "dimension": 2 if is_2d else 1,
        "profile": str(args.profile),
        "t_final_min": args.t_final,
        "timeseries_path": str(csv_path),
        "fields_path": None if fields_path is None else str(fields_path),
        "mass_inflow": result.mass_inflow,
        "mass_source": result.mass_source,
        "mass_outflow": result.mass_outflow,
        "mass_correction": result.mass_correction,
        "mass_balance_error": mass_balance_error,
        "mass_unit": "m3" if physical_volume else "m2",
        "forcing_inputs": {
            "inflow_series": (
                None
                if args.inflow_series is None
                else _portable_path(args.inflow_series)
            ),
            "rainfall_series": (
                None
                if args.rainfall_series is None
                else _portable_path(args.rainfall_series)
            ),
        },
    }
    if is_2d:
        summary["grid"] = {
            "nx": len(result.domain.x_m),
            "ny": len(result.domain.y_m),
            "width_m": float(np.sum(result.domain.dy_m)),
            "hydraulic_geometry": _portable_path(args.hydraulic_geometry),
            "floodplain_slope": args.floodplain_slope,
            "bankfull_depth_m": bankfull_depth.tolist(),
        }
    elif result.domain.channel_width_m is not None:
        summary["cross_section"] = {
            "hydraulic_geometry": _portable_path(args.hydraulic_geometry),
            "channel_width_m": result.domain.channel_width_m.tolist(),
            "bankfull_depth_m": result.domain.bankfull_depth_m.tolist(),
            "shape": "rectangular",
        }
    if args.map_markers is not None:
        summary["map_inputs"] = {
            "markers": _portable_path(args.map_markers),
            "geometry": _portable_path(args.map_geometry),
        }
    json_path = out / f"{args.run_name}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    artifact_text = f"  Fields: {fields_path}" if fields_path else ""
    print(f"Done. CSV: {csv_path}{artifact_text}  Summary: {json_path}")
    print(
        f"Mass balance error: {mass_balance_error:.4e} "
        f"{'m^3' if physical_volume else 'm^2'}"
    )


if __name__ == "__main__":
    main()
