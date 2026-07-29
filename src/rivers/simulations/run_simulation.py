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
    p.add_argument("--rainfall-rate", type=float, default=0.0, help="Uniform rainfall rate, m/min")
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
        left_inflow=args.left_inflow,
        rainfall_rate_m_per_min=args.rainfall_rate,
        cfl=args.cfl,
    )
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
