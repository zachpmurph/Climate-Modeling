"""Run a reproducible two-gauge river validation case.

The upstream observed discharge drives the model boundary.  The downstream
observations are held out and scored against the simulated downstream
hydrograph.  Configuration and observations are plain JSON/CSV so a committed
case runs offline and retains its provenance.
"""

from __future__ import annotations

import argparse
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
    """Return relative minutes and discharge arrays for upstream/downstream gauges."""
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

    origin = min(timestamp for _, timestamp, _ in parsed)
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


def discharge_boundary(times_min, discharge_m3_per_min, width_m):
    """Create a unit-width upstream hydrograph callable in m²/min."""
    if not math.isfinite(width_m) or width_m <= 0:
        raise ValueError("Boundary width must be finite and positive")
    times = np.asarray(times_min, dtype=float)
    unit_discharge = np.asarray(discharge_m3_per_min, dtype=float) / width_m

    def boundary(time_min):
        return float(np.interp(time_min, times, unit_discharge))

    return boundary


def run_validation_case(config_path, *, output_path=None):
    """Run one configured Saint-Venant case and return its validation evidence."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
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
    boundary = discharge_boundary(upstream_times, upstream_flow, upstream_width)
    initial_q = boundary(0.0)
    normal_depth = (initial_q * manning_n / math.sqrt(slope)) ** (3.0 / 5.0)

    result = saint_venant_1d.run_model(
        length_m,
        duration,
        record_interval=float(config.get("record_interval_min", 5.0)),
        h_init=np.full(cells, normal_depth),
        q_init=np.full(cells, initial_q),
        left_inflow=boundary,
        rainfall=lambda x, time: np.zeros_like(x),
        x_m=x_m,
        dx_m=dx_m,
        slope=bed_slope,
        manning_n=roughness,
        cfl=float(config.get("cfl", 0.4)),
    )

    target = (downstream_times >= 0.0) & (downstream_times <= duration)
    target_times = downstream_times[target]
    target_flow = downstream_flow[target]
    predicted_flow = result["q_history"][:, -1] * downstream_width
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
        "status": "uncalibrated_baseline",
        "observations": {
            "path": str(observation_path),
            "upstream_count": int(len(upstream_times)),
            "downstream_count": int(len(target_times)),
            "duration_min": duration,
        },
        "assumptions": {
            **reach,
            "initial_condition": "uniform Manning normal depth and upstream unit discharge",
            "lateral_inflow": "zero",
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
            "inflow_m2": float(result["mass_inflow"]),
            "source_m2": float(result["mass_source"]),
            "outflow_m2": float(result["mass_outflow"]),
            "floor_correction_m2": float(result["mass_floor_correction"]),
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
