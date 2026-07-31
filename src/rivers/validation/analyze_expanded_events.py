"""Summarize error evidence from predeclared, uncalibrated river additions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _finite_correlation(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    if np.sum(valid) < 3:
        return None
    first = first[valid]
    second = second[valid]
    if np.std(first) == 0.0 or np.std(second) == 0.0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def analyze_expanded_events(manifest_path, *, output_path=None):
    """Read fixed case evidence and write a descriptive, non-calibrating analysis."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = []
    for relative_path in manifest["cases"]:
        config_path = manifest_path.parent / relative_path
        config = json.loads(config_path.read_text(encoding="utf-8"))
        result_path = config_path.with_suffix(".results.json")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["solver"] != "saint_venant_2d":
            raise ValueError(f"{relative_path} is not a 2-D validation result")
        context = config["reach_context"]
        diagnosis = result["error_diagnosis"]
        reach = result["reach_diagnosis"]
        cases.append(
            {
                "config": relative_path,
                "river": config["case"]["river"],
                "intervening_drainage_area_growth_fraction": float(
                    context["intervening_drainage_area_growth_fraction"]
                ),
                "nse": result["scores"]["nse"],
                "pearson_r": result["scores"]["pearson_r"],
                "percent_bias": result["scores"]["percent_bias"],
                "predicted_to_observed_volume_ratio": diagnosis["volume_ratio"],
                "observed_downstream_to_upstream_volume_ratio": (
                    1.0 + reach["observed_net_change_fraction"]
                ),
                "peak_lag_error_min": diagnosis["peak_lag_min"],
                "routing_lag_error_min": reach["routing_lag_error_min"],
                "unexplained_downstream_volume_m3": (
                    reach["unexplained_downstream_volume_m3"]
                ),
            }
        )

    growth = [case["intervening_drainage_area_growth_fraction"] for case in cases]
    missing_fraction = [
        1.0 - case["predicted_to_observed_volume_ratio"] for case in cases
    ]
    observed_gain = [
        case["observed_downstream_to_upstream_volume_ratio"] - 1.0
        for case in cases
    ]
    nse = [case["nse"] for case in cases]
    shape = [case["pearson_r"] for case in cases]
    routing_errors = [
        case["routing_lag_error_min"]
        for case in cases
        if case["routing_lag_error_min"] is not None
        and math.isfinite(float(case["routing_lag_error_min"]))
    ]
    early_routing_count = sum(value < 0.0 for value in routing_errors)
    diagnostic_variants = []
    for configured in manifest.get("diagnostic_variants", []):
        baseline = json.loads(
            (manifest_path.parent / configured["baseline"]).read_text(
                encoding="utf-8"
            )
        )
        variant = json.loads(
            (manifest_path.parent / configured["variant"]).read_text(
                encoding="utf-8"
            )
        )
        if baseline["solver"] != "saint_venant_2d" or variant["solver"] != (
            "saint_venant_2d"
        ):
            raise ValueError("Diagnostic variants must both use Saint-Venant 2-D")
        diagnostic_variants.append(
            {
                **configured,
                "baseline_terrain": baseline["terrain_representation"]["name"],
                "variant_terrain": variant["terrain_representation"]["name"],
                "baseline_nse": baseline["scores"]["nse"],
                "variant_nse": variant["scores"]["nse"],
                "baseline_pearson_r": baseline["scores"]["pearson_r"],
                "variant_pearson_r": variant["scores"]["pearson_r"],
                "baseline_volume_ratio": baseline["error_diagnosis"]["volume_ratio"],
                "variant_volume_ratio": variant["error_diagnosis"]["volume_ratio"],
                "baseline_routing_lag_error_min": baseline["reach_diagnosis"][
                    "routing_lag_error_min"
                ],
                "variant_routing_lag_error_min": variant["reach_diagnosis"][
                    "routing_lag_error_min"
                ],
                "interpretation": (
                    "A timing/correlation improvement without a volume recovery "
                    "supports storage/connectivity as a timing error source and "
                    "missing forcing as a separate volume error source."
                ),
            }
        )

    evidence = {
        "schema_version": 1,
        "title": manifest["title"],
        "solver_policy": "saint_venant_2d_only",
        "calibration": "none",
        "case_count": len(cases),
        "cases": cases,
        "descriptive_associations": {
            "sample_size": len(cases),
            "warning": (
                "Five purposively selected events are too few for causal inference; "
                "correlations are hypothesis-ranking evidence only and are strongly "
                "influenced by the Russian River case."
            ),
            "drainage_growth_vs_observed_net_volume_gain_pearson_r": (
                _finite_correlation(growth, observed_gain)
            ),
            "drainage_growth_vs_modeled_volume_deficit_pearson_r": (
                _finite_correlation(growth, missing_fraction)
            ),
            "drainage_growth_vs_nse_pearson_r": _finite_correlation(growth, nse),
            "drainage_growth_vs_hydrograph_correlation_pearson_r": (
                _finite_correlation(growth, shape)
            ),
            "modeled_routing_too_fast_case_count": early_routing_count,
            "routing_lag_comparable_case_count": len(routing_errors),
        },
        "structural_sensitivity": diagnostic_variants,
        "ranked_hypotheses": [
            {
                "rank": 1,
                "source": "missing intervening runoff and tributary inflow",
                "evidence": (
                    "Volume deficit and skill degradation rise sharply with "
                    "intervening drainage-area growth; the model currently supplies "
                    "only the upstream hydrograph."
                ),
                "next_discriminating_test": (
                    "Supply independently observed tributary hydrographs and "
                    "rainfall-runoff lateral sources without fitting them to the "
                    "downstream gauge."
                ),
            },
            {
                "rank": 2,
                "source": "underrepresented reach and floodplain storage",
                "evidence": (
                    f"The modeled routing lag is too short in "
                    f"{early_routing_count} of {len(routing_errors)} comparable cases; "
                    "the one-cell ribbon has no lateral floodplain storage. The "
                    "idealized Russian River shelf removes most routing-lag error "
                    "and improves correlation while leaving its volume deficit."
                ),
                "next_discriminating_test": (
                    "Replace the idealized ribbon with datum-reviewed terrain and "
                    "topobathymetry, then compare routing lag before changing "
                    "roughness."
                ),
            },
            {
                "rank": 3,
                "source": "reach-average geometry and slope proxies",
                "evidence": (
                    "Widths are assumed constant and slopes use gage-datum "
                    "differences, which are not channel-bed profiles; these errors "
                    "directly alter wave speed and storage."
                ),
                "next_discriminating_test": (
                    "Use surveyed sections or conditioned 3DEP/topobathymetry while "
                    "holding forcing and solver settings fixed."
                ),
            },
            {
                "rank": 4,
                "source": "unrepresented controls and event-window storage",
                "evidence": (
                    "Hydropower, reservoir operations, diversions, and incomplete "
                    "recession windows can change downstream volume and timing "
                    "without representing a solver conservation failure."
                ),
                "next_discriminating_test": (
                    "Add observed control operations where available and repeat "
                    "volume accounting over windows that include the full recession."
                ),
            },
            {
                "rank": 5,
                "source": "numerical diffusion or 2-D equation implementation",
                "evidence": (
                    "The same solver and grid achieve high skill on Connecticut and "
                    "Potomac, so numerics alone do not explain the cross-river "
                    "pattern; they remain a secondary contributor."
                ),
                "next_discriminating_test": (
                    "Repeat fixed-forcing cases at multiple longitudinal and lateral "
                    "resolutions and retain every result as a sensitivity test."
                ),
            },
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
    parser = argparse.ArgumentParser(
        description="Analyze fixed evidence from the expanded river validation set."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    evidence = analyze_expanded_events(args.manifest, output_path=args.output)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
