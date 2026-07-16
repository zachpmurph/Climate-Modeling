import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from floods import river_kinematic_wave as rkw


DEFAULT_OUTPUT_DIR = Path("data") / "real_world_rivers" / "runs"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the 1D kinematic wave model on a river profile.")
    parser.add_argument("profile", help="CSV or JSON profile with station_m, slope, and manning_n fields")
    parser.add_argument("--left-inflow-flux", type=float, required=True, help="Constant upstream inflow flux, m^2/min")
    parser.add_argument("--t-final", type=float, required=True, help="Simulation duration in minutes")
    parser.add_argument("--record-interval", type=float, default=1.0, help="Snapshot interval in minutes")
    parser.add_argument("--base-depth", type=float, default=0.01, help="Initial depth if profile has no initial_depth_m column")
    parser.add_argument("--wave-amplitude", type=float, default=0.0, help="Gaussian initial wave amplitude in meters")
    parser.add_argument("--wave-center", type=float, default=None, help="Gaussian initial wave center station in meters")
    parser.add_argument("--wave-width", type=float, default=None, help="Gaussian initial wave standard deviation in meters")
    parser.add_argument("--rainfall-rate", type=float, default=0.0, help="Uniform rainfall/source rate in m/min")
    parser.add_argument("--rainfall-start", type=float, default=0.0, help="Rainfall start time in minutes")
    parser.add_argument("--rainfall-end", type=float, default=None, help="Rainfall end time in minutes; omit to continue through t-final")
    parser.add_argument("--cfl", type=float, default=0.5, help="CFL target in (0, 1]")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output CSV and summary JSON")
    parser.add_argument("--run-name", default="river_kinematic_wave", help="Output filename prefix")
    return parser.parse_args()


def main():
    args = parse_args()
    profile = rkw.load_profile(args.profile)
    result = rkw.run_model(
        profile,
        t_final_min=args.t_final,
        left_inflow_flux=args.left_inflow_flux,
        record_interval_min=args.record_interval,
        base_depth_m=args.base_depth,
        wave_center_m=args.wave_center,
        wave_amplitude_m=args.wave_amplitude,
        wave_width_m=args.wave_width,
        rainfall_rate_m_per_min=args.rainfall_rate,
        rainfall_start_min=args.rainfall_start,
        rainfall_end_min=args.rainfall_end,
        cfl=args.cfl,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.run_name}_depth_timeseries.csv"
    summary_path = args.output_dir / f"{args.run_name}_summary.json"
    rkw.save_time_series_csv(result, csv_path)
    summary = rkw.save_summary_json(result, summary_path)

    print(f"Wrote depth time series: {csv_path}")
    print(f"Wrote run summary: {summary_path}")
    print(f"Final max depth: {summary['max_depth_final_m']:.6g} m")
    print(f"Source mass from rainfall: {summary['mass_source_m2']:.6g} m^2")
    print(f"Mass balance error: {summary['mass_balance_error_m2']:.6g} m^2")


if __name__ == "__main__":
    main()
