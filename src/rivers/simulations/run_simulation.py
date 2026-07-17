import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from general.solvers.contract import Scenario
from general.solvers.profile import domain_from_profile, load_profile
from rivers.simulations.registry import SOLVERS, dispatch


DEFAULT_OUTPUT_DIR = Path("data") / "real_world_rivers" / "runs"


def parse_args():
    p = argparse.ArgumentParser(description="Run a 1-D river solver on a profile.")
    p.add_argument("profile", help="CSV or JSON river profile path")
    p.add_argument(
        "--solver",
        choices=sorted(SOLVERS),
        default="saint_venant_1d",
        help="Which solver to use",
    )
    p.add_argument("--t-final", type=float, required=True, help="Simulation duration in minutes")
    p.add_argument("--record-interval", type=float, default=1.0)
    p.add_argument("--left-inflow", type=float, default=0.0, help="Constant upstream inflow flux, m^2/min")
    p.add_argument("--rainfall-rate", type=float, default=0.0, help="Uniform rainfall rate, m/min")
    p.add_argument("--cfl", type=float, default=0.5)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--run-name", default="simulation")
    return p.parse_args()


def main():
    args = parse_args()

    profile = load_profile(args.profile)
    domain = domain_from_profile(profile)

    rainfall_fn = None
    if args.rainfall_rate > 0:
        rate = args.rainfall_rate
        rainfall_fn = lambda x, t: np.full_like(x, rate)

    scenario = Scenario(
        t_final_min=args.t_final,
        record_interval_min=args.record_interval,
        left_inflow=args.left_inflow,
        rainfall=rainfall_fn,
        cfl=args.cfl,
    )

    result = dispatch(args.solver, domain, scenario)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Write depth CSV (animate_depth-compatible)
    csv_path = out / f"{args.run_name}_timeseries.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_min"] + [f"{x:.6f}" for x in result.domain.x_m])
        for t, row in zip(result.times, result.depth_history):
            writer.writerow([f"{t:.6f}"] + [f"{d:.10g}" for d in row])

    # Mass balance error
    mass_balance_error = (
        result.mass_inflow + result.mass_source - result.mass_outflow
        - float(np.sum((result.depth_final - result.depth_initial) * result.domain.dx_m))
    )

    summary = {
        "solver": args.solver,
        "profile": str(args.profile),
        "t_final_min": args.t_final,
        "mass_inflow": result.mass_inflow,
        "mass_source": result.mass_source,
        "mass_outflow": result.mass_outflow,
        "mass_balance_error": mass_balance_error,
    }
    json_path = out / f"{args.run_name}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"Done. CSV: {csv_path}  Summary: {json_path}")
    print(f"Mass balance error: {mass_balance_error:.4e} m^2")


if __name__ == "__main__":
    main()
