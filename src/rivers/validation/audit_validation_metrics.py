"""Audit headline validation scores against complementary hydrologic metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rivers.validation.case_inputs import load_two_gauge_observations
from rivers.validation.skill import benchmark_skill, skill_scores


def _finite(value):
    return None if value is None or not np.isfinite(float(value)) else float(value)


def audit_validation_metrics(manifest_path, *, output_path=None):
    """Re-score committed series and flag misleading headline metrics."""
    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    excluded = set(manifest.get("excluded_rivers", []))
    rules = manifest["screening_rules"]
    cases = []
    for relative in manifest["cases"]:
        config_path = root / relative
        config = json.loads(config_path.read_text(encoding="utf-8"))
        river = config["case"]["river"]
        if river in excluded:
            raise ValueError(f"Excluded river present in metric audit: {river}")
        result = json.loads(
            config_path.with_suffix(".results.json").read_text(encoding="utf-8")
        )
        if result["solver"] != "saint_venant_2d":
            raise ValueError(f"{relative} is not a 2-D result")
        times = np.asarray(result["series"]["times_min"], dtype=float)
        observed = np.asarray(
            result["series"]["observed_downstream_m3_per_min"], dtype=float
        )
        predicted = np.asarray(
            result["series"]["predicted_downstream_m3_per_min"], dtype=float
        )
        observations = load_two_gauge_observations(root / config["observations"])
        upstream_times, upstream_flow = observations["upstream"]
        upstream = np.interp(times, upstream_times, upstream_flow)
        scores = skill_scores(observed, predicted)
        passthrough = skill_scores(observed, upstream)
        value_added = benchmark_skill(observed, predicted, upstream)
        reach = result["reach_diagnosis"]
        flags = []
        if scores["pearson_r"] >= rules["headline_correlation"] and (
            abs(scores["percent_bias"]) >= rules["material_percent_bias"]
            or scores["kge"] < rules["weak_kge"]
            or scores["volumetric_efficiency"]
            < rules["weak_volumetric_efficiency"]
        ):
            flags.append("high_correlation_hides_magnitude_or_volume_failure")
        lag_comparable = config.get("point_flow_series") is None
        if (
            lag_comparable
            and scores["nse"] >= rules["headline_nse"]
            and abs(float(reach["routing_lag_error_min"]))
            >= rules["material_routing_lag_min"]
        ):
            flags.append("high_nse_hides_routing_lag_failure")
        if scores["nse"] >= rules["headline_nse"] and value_added <= 0.0:
            flags.append("high_nse_adds_no_value_over_upstream_passthrough")
        if scores["nse"] >= rules["headline_nse"] and scores["kge"] < rules[
            "weak_kge"
        ]:
            flags.append("high_nse_hides_kge_component_failure")
        if (
            scores["pearson_r"] >= rules["minimum_shape_correlation"]
            and value_added < rules["minimum_routing_value_added"]
        ):
            flags.append("shape_score_adds_little_value_over_upstream_passthrough")
        cases.append(
            {
                "config": relative,
                "river": river,
                "has_internal_sources": config.get("point_flow_series") is not None,
                "scores": {
                    key: int(value) if key == "n" else _finite(value)
                    for key, value in scores.items()
                },
                "upstream_passthrough_scores": {
                    key: int(value) if key == "n" else _finite(value)
                    for key, value in passthrough.items()
                },
                "squared_error_skill_over_upstream_passthrough": _finite(value_added),
                "routing_lag_error_min": reach["routing_lag_error_min"],
                "routing_lag_comparable": lag_comparable,
                "volume_ratio": result["error_diagnosis"]["volume_ratio"],
                "flags": flags,
            }
        )

    flag_counts = {}
    for case in cases:
        for flag in case["flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    evidence = {
        "schema_version": 1,
        "title": manifest["title"],
        "solver_policy": "saint_venant_2d_only",
        "calibration": "none",
        "excluded_rivers": sorted(excluded),
        "screening_rules": rules,
        "case_count": len(cases),
        "flagged_case_count": sum(bool(case["flags"]) for case in cases),
        "flag_counts": flag_counts,
        "cases": cases,
        "interpretation": {
            "nse": "Sensitive to squared peak errors but can reward a correlated boundary hydrograph despite wrong travel time.",
            "pearson_r": "Measures linear shape association and is invariant to additive and multiplicative bias.",
            "kge": "Combines correlation, variability ratio, and mean-flow ratio; inspect its components rather than using it alone.",
            "volumetric_efficiency": "Penalizes absolute flow-volume error and can be negative.",
            "upstream_passthrough": "Tests whether hydraulic routing beats using the simultaneous upstream gauge unchanged.",
            "guard": "Flags are screening diagnostics, not universal pass/fail thresholds or calibration objectives.",
        },
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
    evidence = audit_validation_metrics(args.manifest, output_path=args.output)
    print(
        json.dumps(
            {
                "case_count": evidence["case_count"],
                "flagged_case_count": evidence["flagged_case_count"],
                "flag_counts": evidence["flag_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
