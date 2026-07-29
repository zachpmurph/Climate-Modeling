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
    load_compound_cross_sections,
    load_profile,
    load_reviewed_terrain,
    load_stage_dependent_manning,
    load_surveyed_cross_sections,
    resample_profile,
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


def _load_temporal_series(path, value_column, *, allow_negative=False):
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
        or (not allow_negative and np.any(values < 0))
    ):
        raise ValueError(
            f"{path} needs t_min starting at 0 and strictly increasing, "
            f"with finite "
            f"{'values' if allow_negative else 'non-negative values'} "
            f"for {value_column}"
        )

    def forcing(time_min):
        return float(np.interp(time_min, times, values))

    forcing.breakpoints_min = times.copy()
    return forcing


def _load_point_lateral_inflows(path, x_m, dx_m):
    """Map measured point inflows to conservative nearest-cell source rates."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} must contain point-inflow rows")
    grouped = {}
    try:
        for row in rows:
            station = float(row["station_m"])
            time = float(row["t_min"])
            flow = float(row["discharge_m3_per_min"])
            grouped.setdefault(station, []).append((time, flow))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{path} must contain numeric station_m, t_min, and "
            "discharge_m3_per_min columns"
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
            or not np.isfinite(station)
            or station < stations[0]
            or station > stations[-1]
            or np.any(~np.isfinite(times))
            or np.any(~np.isfinite(flows))
            or times[0] != 0.0
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError(
                f"Each point inflow in {path} needs a station inside the "
                "domain and at least two finite signed values with "
                "t_min starting at 0 and strictly increasing"
            )
        cell = int(np.argmin(np.abs(stations - station)))
        series.append((cell, times, flows))
        all_breakpoints.update(float(value) for value in times)

    def lateral_inflow(x, time_min):
        rates = np.zeros_like(x, dtype=float)
        for cell, times, flows in series:
            rates[cell] += float(np.interp(time_min, times, flows)) / cell_lengths[cell]
        return rates

    lateral_inflow.breakpoints_min = tuple(sorted(all_breakpoints))
    lateral_inflow.point_count = len(series)
    return lateral_inflow


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
            "Constant upstream flow: m^3/min with reviewed cross-section "
            "geometry, legacy unit-width m^2/min otherwise"
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
    p.add_argument(
        "--lateral-inflow-rate",
        type=float,
        default=0.0,
        help="Uniform lateral discharge in m^3/min per metre of reach",
    )
    p.add_argument(
        "--lateral-inflow-series",
        type=Path,
        help=(
            "CSV with t_min,lateral_inflow_m3_per_min_per_m for uniform "
            "time-varying distributed reach inflow"
        ),
    )
    p.add_argument(
        "--lateral-inflow-points",
        type=Path,
        help=(
            "CSV with station_m,t_min,discharge_m3_per_min for measured "
            "tributary/return inflows (positive) or withdrawals (negative)"
        ),
    )
    p.add_argument(
        "--downstream-boundary",
        choices=("outflow", "wall", "stage"),
        default="outflow",
        help="1-D Saint-Venant downstream condition (default: outflow)",
    )
    p.add_argument(
        "--downstream-stage",
        type=float,
        help="Fixed water-surface elevation in metres for --downstream-boundary stage",
    )
    p.add_argument(
        "--downstream-stage-series",
        type=Path,
        help="CSV with t_min,downstream_stage_m for a measured stage boundary",
    )
    p.add_argument(
        "--spatial-order",
        type=int,
        choices=(1, 2),
        default=1,
        help="Saint-Venant reconstruction order (default: 1)",
    )
    p.add_argument("--cfl", type=float, default=0.5)
    p.add_argument(
        "--longitudinal-cells",
        type=int,
        help=(
            "Linearly interpolate reviewed profile fields onto this many "
            "derived solver cells"
        ),
    )
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
        "--cross-section-shape",
        choices=("rectangular", "trapezoidal", "compound", "surveyed"),
        default="rectangular",
        help="1-D channel shape (default: rectangular)",
    )
    p.add_argument(
        "--compound-cross-sections",
        type=Path,
        help=(
            "Reviewed station/depth/top-width CSV for stage-dependent "
            "compound 1-D sections"
        ),
    )
    p.add_argument(
        "--surveyed-cross-sections",
        type=Path,
        help=(
            "Reviewed station/offset/elevation CSV for asymmetric surveyed "
            "1-D sections"
        ),
    )
    p.add_argument(
        "--stage-manning",
        type=Path,
        help=(
            "Reviewed station/depth/Manning-n CSV for depth-dependent "
            "1-D conveyance"
        ),
    )
    p.add_argument(
        "--bottom-width-fraction",
        type=float,
        default=0.5,
        help=(
            "Trapezoid bottom width divided by reviewed bankfull width "
            "(default: 0.5)"
        ),
    )
    p.add_argument(
        "--floodplain-slope",
        type=float,
        default=0.02,
        help="Lateral rise/run outside the reviewed channel width (default: 0.02)",
    )
    p.add_argument(
        "--terrain-grid",
        type=Path,
        help=(
            "Reviewed Cartesian x/y/dx/dy/bed CSV for saint_venant_2d; "
            "replaces synthetic channel terrain"
        ),
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
    if (
        sum(
            (
                args.lateral_inflow_series is not None,
                args.lateral_inflow_points is not None,
                args.lateral_inflow_rate != 0.0,
            )
        )
        > 1
    ):
        raise SystemExit(
            "error: use only one lateral inflow rate, uniform series, "
            "or point-input file"
        )
    if args.lateral_inflow_rate < 0.0:
        raise SystemExit("error: --lateral-inflow-rate must be non-negative")
    if args.solver != "saint_venant" and (
        args.lateral_inflow_series is not None
        or args.lateral_inflow_points is not None
        or args.lateral_inflow_rate != 0.0
    ):
        raise SystemExit(
            "error: lateral inflow currently requires --solver saint_venant"
        )
    if args.solver not in {"saint_venant", "saint_venant_2d"} and (
        args.downstream_boundary != "outflow"
        or args.downstream_stage is not None
        or args.downstream_stage_series is not None
    ):
        raise SystemExit(
            "error: downstream boundary options require a Saint-Venant solver"
        )
    if (
        args.solver == "saint_venant_2d"
        and args.downstream_boundary == "wall"
    ):
        raise SystemExit(
            "error: the 2-D runner supports outflow or stage downstream boundaries"
        )
    if args.cross_section_shape == "trapezoidal" and (
        args.solver != "saint_venant" or args.hydraulic_geometry is None
    ):
        raise SystemExit(
            "error: trapezoidal cross-sections require --solver saint_venant "
            "and --hydraulic-geometry"
        )
    if args.cross_section_shape == "compound" and (
        args.solver != "saint_venant"
        or args.compound_cross_sections is None
    ):
        raise SystemExit(
            "error: compound cross-sections require --solver saint_venant "
            "and --compound-cross-sections"
        )
    if (
        args.compound_cross_sections is not None
        and args.cross_section_shape != "compound"
    ):
        raise SystemExit(
            "error: --compound-cross-sections requires "
            "--cross-section-shape compound"
        )
    if (
        args.compound_cross_sections is not None
        and args.hydraulic_geometry is not None
    ):
        raise SystemExit(
            "error: use compound cross-sections or parameterized hydraulic "
            "geometry, not both"
        )
    if args.cross_section_shape == "surveyed" and (
        args.solver != "saint_venant"
        or args.surveyed_cross_sections is None
    ):
        raise SystemExit(
            "error: surveyed cross-sections require --solver saint_venant "
            "and --surveyed-cross-sections"
        )
    if (
        args.surveyed_cross_sections is not None
        and args.cross_section_shape != "surveyed"
    ):
        raise SystemExit(
            "error: --surveyed-cross-sections requires "
            "--cross-section-shape surveyed"
        )
    if args.surveyed_cross_sections is not None and (
        args.hydraulic_geometry is not None
        or args.compound_cross_sections is not None
    ):
        raise SystemExit(
            "error: use surveyed cross-sections, compound stage-width "
            "curves, or parameterized geometry, not more than one"
        )
    if args.terrain_grid is not None and args.solver != "saint_venant_2d":
        raise SystemExit(
            "error: --terrain-grid requires --solver saint_venant_2d"
        )
    if args.stage_manning is not None and args.solver != "saint_venant":
        raise SystemExit(
            "error: --stage-manning requires --solver saint_venant"
        )
    if args.terrain_grid is not None and (
        args.width is not None or args.hydraulic_geometry is not None
    ):
        raise SystemExit(
            "error: reviewed terrain replaces --width and "
            "--hydraulic-geometry"
        )
    if not np.isfinite(args.bottom_width_fraction) or not (
        0.0 < args.bottom_width_fraction <= 1.0
    ):
        raise SystemExit(
            "error: --bottom-width-fraction must be in the interval (0, 1]"
        )
    if (
        args.solver not in {"saint_venant", "saint_venant_2d"}
        and args.spatial_order != 1
    ):
        raise SystemExit(
            "error: --spatial-order requires a Saint-Venant solver"
        )
    if (
        args.downstream_stage is not None
        and args.downstream_stage_series is not None
    ):
        raise SystemExit(
            "error: use either --downstream-stage or "
            "--downstream-stage-series, not both"
        )
    has_downstream_stage = (
        args.downstream_stage is not None
        or args.downstream_stage_series is not None
    )
    if (args.downstream_boundary == "stage") != has_downstream_stage:
        raise SystemExit(
            "error: one downstream stage input is required only with "
            "--downstream-boundary stage"
        )
    for forcing_path in (
        args.inflow_series,
        args.rainfall_series,
        args.lateral_inflow_series,
        args.lateral_inflow_points,
        args.downstream_stage_series,
        args.compound_cross_sections,
        args.surveyed_cross_sections,
        args.stage_manning,
        args.terrain_grid,
    ):
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
        temporal_lateral_inflow = (
            None
            if args.lateral_inflow_series is None
            else _load_temporal_series(
                args.lateral_inflow_series,
                "lateral_inflow_m3_per_min_per_m",
            )
        )
        temporal_downstream_stage = (
            None
            if args.downstream_stage_series is None
            else _load_temporal_series(
                args.downstream_stage_series,
                "downstream_stage_m",
                allow_negative=True,
            )
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    source_profile = load_profile(args.profile)
    source_cells = len(source_profile.station_m)
    try:
        profile = (
            source_profile
            if args.longitudinal_cells is None
            else resample_profile(source_profile, args.longitudinal_cells)
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    stage_manning_arguments = {}
    if args.stage_manning is not None:
        try:
            manning_depth, manning_table = load_stage_dependent_manning(
                args.stage_manning, profile.station_m
            )
        except ValueError as exc:
            raise SystemExit(f"error: {exc}") from exc
        stage_manning_arguments = {
            "manning_depth_m": manning_depth,
            "manning_n_table": manning_table,
        }
    if args.solver == "saint_venant_2d":
        if args.terrain_grid is not None:
            try:
                domain, terrain_roughness_source = load_reviewed_terrain(
                    args.terrain_grid, profile
                )
            except ValueError as exc:
                raise SystemExit(f"error: {exc}") from exc
        else:
            terrain_roughness_source = "river_profile_repeated_across_y"
            if args.width is None:
                raise SystemExit(
                    "error: --width is required for synthetic "
                    "saint_venant_2d terrain"
                )
            if args.hydraulic_geometry is None:
                raise SystemExit(
                    "error: --hydraulic-geometry is required for synthetic "
                    "saint_venant_2d terrain"
                )
            if not args.hydraulic_geometry.is_file():
                raise SystemExit(
                    "error: hydraulic geometry does not exist: "
                    f"{args.hydraulic_geometry}"
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
        if args.surveyed_cross_sections is not None:
            try:
                (
                    section_depth,
                    section_width,
                    section_perimeter,
                ) = load_surveyed_cross_sections(
                    args.surveyed_cross_sections, profile.station_m
                )
            except ValueError as exc:
                raise SystemExit(f"error: {exc}") from exc
            domain = domain_from_profile(
                profile,
                cross_section_depth_m=section_depth,
                cross_section_top_width_m=section_width,
                cross_section_wetted_perimeter_m=section_perimeter,
                **stage_manning_arguments,
            )
        elif args.compound_cross_sections is not None:
            try:
                section_depth, section_width = (
                    load_compound_cross_sections(
                        args.compound_cross_sections, profile.station_m
                    )
                )
            except ValueError as exc:
                raise SystemExit(f"error: {exc}") from exc
            domain = domain_from_profile(
                profile,
                cross_section_depth_m=section_depth,
                cross_section_top_width_m=section_width,
                **stage_manning_arguments,
            )
        elif args.hydraulic_geometry is None:
            domain = domain_from_profile(
                profile, **stage_manning_arguments
            )
        else:
            if not args.hydraulic_geometry.is_file():
                raise SystemExit(
                    f"error: hydraulic geometry does not exist: {args.hydraulic_geometry}"
                )
            channel_width, bankfull_depth = load_channel_geometry(
                args.hydraulic_geometry, profile.station_m
            )
            bottom_width = None
            side_slope = None
            if args.cross_section_shape == "trapezoidal":
                bottom_width = channel_width * args.bottom_width_fraction
                side_slope = (
                    channel_width - bottom_width
                ) / (2.0 * bankfull_depth)
            domain = domain_from_profile(
                profile,
                channel_width_m=channel_width,
                bankfull_depth_m=bankfull_depth,
                channel_bottom_width_m=bottom_width,
                side_slope_h_to_v=side_slope,
                **stage_manning_arguments,
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

    if args.lateral_inflow_points is not None:
        try:
            scenario.lateral_inflow = _load_point_lateral_inflows(
                args.lateral_inflow_points, domain.x_m, domain.dx_m
            )
        except ValueError as exc:
            raise SystemExit(f"error: {exc}") from exc
    elif args.lateral_inflow_rate != 0.0 or temporal_lateral_inflow is not None:
        lateral_forcing = (
            args.lateral_inflow_rate
            if temporal_lateral_inflow is None
            else temporal_lateral_inflow
        )

        def uniform_lateral_inflow(x, time):
            return np.full_like(
                x, _forcing_value(lateral_forcing, time), dtype=float
            )

        if hasattr(lateral_forcing, "breakpoints_min"):
            uniform_lateral_inflow.breakpoints_min = (
                lateral_forcing.breakpoints_min
            )
        scenario.lateral_inflow = uniform_lateral_inflow

    if isinstance(domain, Domain2D):
        channel_depth = (
            np.zeros(len(domain.x_m))
            if profile.initial_depth_m is None
            else np.interp(
                domain.x_m,
                profile.station_m,
                np.asarray(profile.initial_depth_m, dtype=float),
            )
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
        scenario.boundary_x = (
            "inflow_stage"
            if args.downstream_boundary == "stage"
            else "inflow_outflow"
        )
        scenario.downstream_stage_m = (
            args.downstream_stage
            if temporal_downstream_stage is None
            else temporal_downstream_stage
        )
        scenario.spatial_order = args.spatial_order
    elif args.solver == "saint_venant":
        scenario.initial_discharge = np.full(
            len(domain.x_m), _forcing_value(inflow, 0.0)
        )
        scenario.downstream_boundary = args.downstream_boundary
        scenario.downstream_stage_m = (
            args.downstream_stage
            if temporal_downstream_stage is None
            else temporal_downstream_stage
        )
        scenario.spatial_order = args.spatial_order

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

    discharge_path = None
    if not is_2d and "discharge_history" in result.extra:
        discharge_path = out / f"{args.run_name}_discharge.csv"
        with discharge_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["t_min"] + [f"{x:.6f}" for x in result.domain.x_m]
            )
            for t, row in zip(
                result.times, result.extra["discharge_history"]
            ):
                writer.writerow(
                    [f"{t:.6f}"] + [f"{q:.10g}" for q in row]
                )

    manning_history_path = None
    if (
        not is_2d
        and result.extra.get("manning_n_table") is not None
    ):
        manning_history_path = out / f"{args.run_name}_manning_n.csv"
        with manning_history_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["t_min"] + [f"{x:.6f}" for x in result.domain.x_m]
            )
            for t, row in zip(
                result.times, result.extra["manning_n_history"]
            ):
                writer.writerow(
                    [f"{t:.6f}"] + [f"{value:.10g}" for value in row]
                )

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
            slope_x=result.domain.slope_x,
            slope_y=result.domain.slope_y,
            manning_n=result.domain.manning_n,
            times_min=result.times,
            depth_m=result.depth_history,
            depth_initial_m=result.depth_initial,
            depth_final_m=result.depth_final,
            discharge_x_m2_per_min=result.extra["discharge_x_history"],
            discharge_y_m2_per_min=result.extra["discharge_y_history"],
            discharge_x_final=result.extra["discharge_x_final"],
            discharge_y_final=result.extra["discharge_y_final"],
            cumulative_infiltration_m=result.extra[
                "cumulative_infiltration_history"
            ],
        )

    # Mass balance error
    has_physical_section = (
        not is_2d
        and (
            result.domain.channel_width_m is not None
            or result.domain.cross_section_top_width_m is not None
        )
    )
    physical_volume = is_2d or has_physical_section
    if is_2d:
        storage_change = float(
            np.sum(
                (result.depth_final - result.depth_initial)
                * result.domain.dx_m[:, None]
                * result.domain.dy_m[None, :]
            )
        )
    elif "cross_section_area_history" in result.extra:
        area_history = result.extra["cross_section_area_history"]
        storage_change = float(
            np.sum((area_history[-1] - area_history[0]) * result.domain.dx_m)
        )
    else:
        cell_measure = (
            result.domain.dx_m
            if result.domain.channel_width_m is None
            else result.domain.dx_m * result.domain.channel_width_m
        )
        storage_change = float(
            np.sum((result.depth_final - result.depth_initial) * cell_measure)
        )
    mass_balance_error = (
        result.mass_inflow + result.mass_source + result.mass_correction - result.mass_outflow
        - storage_change
    )

    summary = {
        "solver": args.solver,
        "dimension": 2 if is_2d else 1,
        "profile": str(args.profile),
        "t_final_min": args.t_final,
        "timeseries_path": str(csv_path),
        "discharge_path": (
            None if discharge_path is None else str(discharge_path)
        ),
        "discharge_unit": (
            None
            if discharge_path is None
            else (
                "m3_per_min"
                if has_physical_section
                else "m2_per_min"
            )
        ),
        "manning_n_history_path": (
            None
            if manning_history_path is None
            else str(manning_history_path)
        ),
        "fields_path": None if fields_path is None else str(fields_path),
        "mass_inflow": result.mass_inflow,
        "mass_source": result.mass_source,
        "mass_rainfall": result.extra.get("mass_rainfall"),
        "mass_lateral_inflow": result.extra.get("mass_lateral_inflow"),
        "mass_infiltration": result.extra.get("mass_infiltration"),
        "mass_outflow": result.mass_outflow,
        "mass_correction": result.mass_correction,
        "mass_balance_error": mass_balance_error,
        "mass_unit": "m3" if physical_volume else "m2",
        "profile_resolution": {
            "source_observation_stations": source_cells,
            "solver_cells": len(result.domain.x_m),
            "method": (
                "reviewed_terrain_longitudinal_grid"
                if args.terrain_grid is not None
                else (
                    "source_grid"
                    if args.longitudinal_cells is None
                    else "linear_interpolation_derived_grid"
                )
            ),
            "creates_observations": False,
        },
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
            "lateral_inflow_series": (
                None
                if args.lateral_inflow_series is None
                else _portable_path(args.lateral_inflow_series)
            ),
            "lateral_inflow_rate_m3_per_min_per_m": args.lateral_inflow_rate,
        },
        "downstream_boundary": {
            "type": args.downstream_boundary,
            "stage_m": args.downstream_stage,
        },
        "spatial_order": args.spatial_order,
    }
    if result.extra.get("soil_ksat_m_per_min") is not None:
        summary["soil_infiltration"] = {
            "method": "green_ampt",
            "soil_ksat_m_per_min": result.extra[
                "soil_ksat_m_per_min"
            ].tolist(),
            "soil_suction_head_m": result.extra[
                "soil_suction_head_m"
            ].tolist(),
            "soil_moisture_deficit": result.extra[
                "soil_moisture_deficit"
            ].tolist(),
            "cumulative_infiltration_final_m": result.extra[
                "cumulative_infiltration_final"
            ].tolist(),
            "recovery_between_storms": False,
        }
    if args.stage_manning is not None:
        summary["stage_dependent_manning"] = {
            "source": _portable_path(args.stage_manning),
            "depth_levels_m": result.domain.manning_depth_m.tolist(),
            "manning_n": result.domain.manning_n_table.tolist(),
            "interpolation": "linear_with_endpoint_clamping",
        }
    if args.lateral_inflow_points is not None:
        summary["forcing_inputs"]["lateral_inflow_points"] = _portable_path(
            args.lateral_inflow_points
        )
    if args.downstream_stage_series is not None:
        stage_series_path = _portable_path(args.downstream_stage_series)
        summary["forcing_inputs"]["downstream_stage_series"] = stage_series_path
        summary["downstream_boundary"]["stage_series"] = stage_series_path
    if is_2d:
        summary["grid"] = {
            "nx": len(result.domain.x_m),
            "ny": len(result.domain.y_m),
            "width_m": float(np.sum(result.domain.dy_m)),
            "terrain_source": (
                "reviewed_grid"
                if args.terrain_grid is not None
                else "parameterized_channel_and_floodplain"
            ),
            "terrain_grid": (
                None
                if args.terrain_grid is None
                else _portable_path(args.terrain_grid)
            ),
            "roughness_source": terrain_roughness_source,
        }
        if args.terrain_grid is None:
            summary["grid"].update(
                {
                    "hydraulic_geometry": _portable_path(
                        args.hydraulic_geometry
                    ),
                    "floodplain_slope": args.floodplain_slope,
                    "bankfull_depth_m": bankfull_depth.tolist(),
                }
            )
    elif (
        result.domain.channel_width_m is not None
        or result.domain.cross_section_top_width_m is not None
    ):
        summary["cross_section"] = {
            "shape": result.extra.get("cross_section_shape", "rectangular"),
        }
        if result.domain.channel_width_m is not None:
            summary["cross_section"].update(
                {
                    "hydraulic_geometry": _portable_path(
                        args.hydraulic_geometry
                    ),
                    "channel_width_m": (
                        result.domain.channel_width_m.tolist()
                    ),
                    "bankfull_depth_m": (
                        result.domain.bankfull_depth_m.tolist()
                    ),
                }
            )
        if result.domain.cross_section_top_width_m is not None:
            section_metadata = {
                "depth_levels_m": (
                    result.domain.cross_section_depth_m.tolist()
                ),
                "top_width_m": (
                    result.domain.cross_section_top_width_m.tolist()
                ),
                "above_reviewed_depth": "vertical_wall_extrapolation",
                "bank_symmetry_assumption": (
                    result.domain.cross_section_wetted_perimeter_m is None
                ),
            }
            if result.domain.cross_section_wetted_perimeter_m is None:
                section_metadata["compound_cross_sections"] = (
                    _portable_path(args.compound_cross_sections)
                )
            else:
                section_metadata.update(
                    {
                        "surveyed_cross_sections": _portable_path(
                            args.surveyed_cross_sections
                        ),
                        "source_format": (
                            "station_offset_elevation_polyline"
                        ),
                        "wetted_perimeter_m": (
                            result.domain
                            .cross_section_wetted_perimeter_m.tolist()
                        ),
                    }
                )
            summary["cross_section"].update(section_metadata)
        if result.domain.channel_bottom_width_m is not None:
            summary["cross_section"]["channel_bottom_width_m"] = (
                result.domain.channel_bottom_width_m.tolist()
            )
            summary["cross_section"]["side_slope_h_to_v"] = (
                result.domain.side_slope_h_to_v.tolist()
            )
    if args.map_markers is not None:
        summary["map_inputs"] = {
            "markers": _portable_path(args.map_markers),
            "geometry": _portable_path(args.map_geometry),
        }
    json_path = out / f"{args.run_name}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    artifact_text = f"  Fields: {fields_path}" if fields_path else ""
    discharge_text = (
        f"  Discharge: {discharge_path}" if discharge_path else ""
    )
    print(
        f"Done. CSV: {csv_path}{discharge_text}{artifact_text}  "
        f"Summary: {json_path}"
    )
    print(
        f"Mass balance error: {mass_balance_error:.4e} "
        f"{'m^3' if physical_volume else 'm^2'}"
    )


if __name__ == "__main__":
    main()
