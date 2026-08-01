"""Compare fixed longer-window event results with their retained baselines."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def analyze_extended_windows(manifest_path, *, output_path=None):
    """Build reproducible short-versus-extended routing evidence."""
    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    extension = float(manifest["extension_hours"])
    material_lag = float(manifest["material_routing_error_min"])
    minimum_recession = float(manifest["minimum_recession_hours"])
    mass_tolerance = float(manifest["mass_residual_tolerance"])
    cases = []
    for relative in manifest["base_cases"]:
        base_path = root / relative
        base_config = json.loads(base_path.read_text(encoding="utf-8"))
        base = json.loads(
            base_path.with_suffix(".results.json").read_text(encoding="utf-8")
        )
        extended_name = f"{base_path.stem}_extended_{extension:g}h.json"
        extended = json.loads(
            (root / extended_name).with_suffix(".results.json").read_text(
                encoding="utf-8"
            )
        )
        routing_comparable = base_config.get("point_flow_series") is None
        short_duration = float(base["observations"]["duration_min"])
        long_duration = float(extended["observations"]["duration_min"])
        short_lag = base["reach_diagnosis"]["routing_lag_error_min"]
        long_lag = extended["reach_diagnosis"]["routing_lag_error_min"]
        recession_hours = (
            long_duration - extended["error_diagnosis"]["observed_peak_time_min"]
        ) / 60.0
        mass_residual = abs(float(extended["mass"]["relative_balance_residual"]))
        cases.append(
            {
                "base_config": relative,
                "extended_config": extended_name,
                "river": base_config["case"]["river"],
                "has_internal_sources": not routing_comparable,
                "duration_days": {
                    "baseline": short_duration / 1440.0,
                    "extended": long_duration / 1440.0,
                },
                "observed_recession_hours_after_peak": recession_hours,
                "minimum_recession_covered": recession_hours >= minimum_recession,
                "scores": {
                    metric: {
                        "baseline": base["scores"].get(metric),
                        "extended": extended["scores"].get(metric),
                        "change": (
                            None
                            if base["scores"].get(metric) is None
                            or extended["scores"].get(metric) is None
                            else extended["scores"][metric]
                            - base["scores"][metric]
                        ),
                    }
                    for metric in (
                        "nse",
                        "pearson_r",
                        "percent_bias",
                        "kge",
                        "volumetric_efficiency",
                    )
                },
                "routing": {
                    "comparable": routing_comparable,
                    "baseline_lag_error_min": short_lag,
                    "extended_lag_error_min": long_lag,
                    "materially_early_extended": (
                        routing_comparable
                        and long_lag is not None
                        and long_lag <= -material_lag
                    ),
                    "short_window_artifact_reduction_min": (
                        None
                        if not routing_comparable
                        or short_lag is None
                        or long_lag is None
                        else abs(float(short_lag)) - abs(float(long_lag))
                    ),
                },
                "numerics": {
                    "relative_mass_balance_residual": mass_residual,
                    "within_mass_tolerance": mass_residual <= mass_tolerance,
                    "floor_correction_m3": extended["mass"]["floor_correction_m3"],
                    "all_reported_scores_finite": all(
                        value is not None and math.isfinite(float(value))
                        for key, value in extended["scores"].items()
                        if key != "n"
                    ),
                },
            }
        )

    comparable = [case for case in cases if case["routing"]["comparable"]]
    extended_lags = [
        float(case["routing"]["extended_lag_error_min"])
        for case in comparable
        if case["routing"]["extended_lag_error_min"] is not None
    ]
    nse_improved = sum(
        case["scores"]["nse"]["change"] > 0.0 for case in cases
    )
    abs_bias_improved = sum(
        abs(case["scores"]["percent_bias"]["extended"])
        < abs(case["scores"]["percent_bias"]["baseline"])
        for case in cases
    )
    short_nse = [case["scores"]["nse"]["baseline"] for case in cases]
    long_nse = [case["scores"]["nse"]["extended"] for case in cases]
    short_abs_bias = [
        abs(case["scores"]["percent_bias"]["baseline"]) for case in cases
    ]
    long_abs_bias = [
        abs(case["scores"]["percent_bias"]["extended"]) for case in cases
    ]
    evidence = {
        "schema_version": 1,
        "title": manifest["title"],
        "solver_policy": "saint_venant_2d_only",
        "calibration": "none",
        "extension_hours": extension,
        "case_count": len(cases),
        "cases": cases,
        "summary": {
            "routing_comparable_case_count": len(comparable),
            "extended_routing_early_case_count": sum(
                lag < 0.0 for lag in extended_lags
            ),
            "extended_routing_materially_early_case_count": sum(
                lag <= -material_lag for lag in extended_lags
            ),
            "extended_routing_zero_error_case_count": sum(
                lag == 0.0 for lag in extended_lags
            ),
            "extended_routing_late_case_count": sum(
                lag > 0.0 for lag in extended_lags
            ),
            "early_routing_prevalent": (
                sum(lag < 0.0 for lag in extended_lags) > len(extended_lags) / 2
            ),
            "minimum_recession_covered_case_count": sum(
                case["minimum_recession_covered"] for case in cases
            ),
            "nse_improved_case_count": nse_improved,
            "absolute_percent_bias_improved_case_count": abs_bias_improved,
            "median_nse": {
                "baseline": float(np.median(short_nse)),
                "extended": float(np.median(long_nse)),
            },
            "median_absolute_percent_bias": {
                "baseline": float(np.median(short_abs_bias)),
                "extended": float(np.median(long_abs_bias)),
            },
            "maximum_relative_mass_balance_residual": max(
                case["numerics"]["relative_mass_balance_residual"]
                for case in cases
            ),
            "all_cases_numerically_stable_and_conservative": all(
                case["numerics"]["within_mass_tolerance"]
                and case["numerics"]["all_reported_scores_finite"]
                for case in cases
            ),
        },
        "interpretation": [
            "Fixed longer windows remove endpoint-peak artifacts without choosing an endpoint from model skill.",
            "Numerical stability and conservation do not imply correct routing speed, forcing, or geometry.",
            "Routing lag is not comparable for internal-source cases because downstream timing mixes source paths.",
            "Rio Grande is retained only because this study covers all committed baseline events; interpret it separately from cross-river attribution.",
        ],
    }
    destination = (
        Path(output_path)
        if output_path is not None
        else manifest_path.with_suffix(".results.json")
    )
    destination.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    evidence = analyze_extended_windows(args.manifest, output_path=args.output)
    print(json.dumps(evidence["summary"], indent=2))


if __name__ == "__main__":
    main()
