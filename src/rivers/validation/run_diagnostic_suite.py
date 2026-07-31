"""Run fixed, uncalibrated cases and attribute error symptoms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rivers.validation.datasets import hierarchy_evidence
from rivers.validation.run_case import run_validation_case


def run_diagnostic_suite(manifest_path, *, output_path=None):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("calibration") != "none":
        raise ValueError("Diagnostic suite calibration must be 'none'")

    cases = []
    cases_by_config = {}
    dataset_ids = set()
    rivers = set()
    for relative in manifest["cases"]:
        result = run_validation_case(manifest_path.parent / relative)
        if result["solver"] != "saint_venant_2d":
            raise RuntimeError("Diagnostic suite accepts 2-D results only")
        dataset = result["hydraulic_dataset"]
        dataset_ids.add(dataset["dataset_id"])
        rivers.add(result["case"]["river"])
        case_evidence = {
                "config": relative,
                "name": result["case"]["name"],
                "river": result["case"]["river"],
                "hydraulic_dataset": dataset,
                "scores": result["scores"],
                "error_diagnosis": result["error_diagnosis"],
            }
        cases.append(case_evidence)
        cases_by_config[relative] = case_evidence

    paired_comparisons = []
    for comparison in manifest.get("paired_comparisons", []):
        control = cases_by_config[comparison["control"]]
        treatment = cases_by_config[comparison["treatment"]]
        nse_delta = treatment["scores"]["nse"] - control["scores"]["nse"]
        correlation_delta = (
            treatment["scores"]["pearson_r"] - control["scores"]["pearson_r"]
        )
        paired_comparisons.append(
            {
                **comparison,
                "control_dataset": control["hydraulic_dataset"]["dataset_id"],
                "treatment_dataset": treatment["hydraulic_dataset"]["dataset_id"],
                "score_deltas_treatment_minus_control": {
                    "nse": nse_delta,
                    "rmse": (
                        treatment["scores"]["rmse"] - control["scores"]["rmse"]
                    ),
                    "percent_bias": (
                        treatment["scores"]["percent_bias"]
                        - control["scores"]["percent_bias"]
                    ),
                    "pearson_r": correlation_delta,
                },
                "finding": (
                    "The hydraulic-data substitution materially improved the event."
                    if nse_delta > 0.1 and correlation_delta >= 0.0
                    else "The hydraulic-data substitution materially degraded the event."
                    if nse_delta < -0.1 and correlation_delta <= 0.0
                    else "The hydraulic-data substitution did not resolve the event error; "
                    "investigate forcing, controls, reach processes, and model structure."
                ),
            }
        )

    evidence = {
        "schema_version": 1,
        "suite": manifest["suite"],
        "calibration": "none",
        "solver_policy": "saint_venant_2d_only",
        "dataset_hierarchy": hierarchy_evidence(),
        "regions_tested": sorted(rivers),
        "datasets_tested": sorted(dataset_ids),
        "coverage_gaps": [
            dataset_id
            for level in hierarchy_evidence()
            for item in level
            for dataset_id in [item["dataset_id"]]
            if dataset_id not in dataset_ids
        ],
        "interpretation_guard": (
            "Differences between unrelated rivers confound dataset and river physics. "
            "Only paired cases on the same event can isolate a dataset substitution."
        ),
        "paired_comparisons": paired_comparisons,
        "cases": cases,
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
        description="Run uncalibrated cross-region validation diagnostics."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    evidence = run_diagnostic_suite(args.manifest, output_path=args.output)
    print(
        json.dumps(
            {
                "suite": evidence["suite"],
                "regions_tested": evidence["regions_tested"],
                "datasets_tested": evidence["datasets_tested"],
                "coverage_gaps": evidence["coverage_gaps"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
