"""Run every case in a committed validation-suite manifest."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rivers.validation.run_case import run_validation_case


def summarize_results(results):
    """Return transparent across-event score ranges and medians."""
    metrics = ("nse", "rmse", "bias", "percent_bias", "pearson_r")
    summary = {}
    for metric in metrics:
        values = np.asarray(
            [
                result["scores"][metric]
                for result in results
                if result["scores"][metric] is not None
                and math.isfinite(float(result["scores"][metric]))
            ],
            dtype=float,
        )
        summary[metric] = {
            "minimum": float(np.min(values)),
            "median": float(np.median(values)),
            "maximum": float(np.max(values)),
        }
    return summary


def run_validation_suite(manifest_path, *, output_path=None):
    """Run all manifest cases, update case evidence, and write suite evidence."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []
    cases = []
    for relative_path in manifest["cases"]:
        config_path = manifest_path.parent / relative_path
        result = run_validation_case(config_path)
        if result["solver"] != "saint_venant_2d":
            raise RuntimeError("Observed validation suite accepts 2-D results only")
        results.append(result)
        cases.append(
            {
                "config": relative_path,
                "name": result["case"]["name"],
                "status": result["status"],
                "solver": result["solver"],
                "terrain_representation": result["terrain_representation"],
                "scores": result["scores"],
                "reach_context": result.get("reach_context"),
                "error_diagnosis": result["error_diagnosis"],
                "reach_diagnosis": result["reach_diagnosis"],
                "observation_count": {
                    "upstream": result["observations"]["upstream_count"],
                    "downstream": result["observations"]["downstream_count"],
                },
            }
        )

    evidence = {
        "schema_version": 1,
        "suite": manifest["suite"],
        "solver_policy": "saint_venant_2d_only",
        "case_count": len(cases),
        "parameter_policy": manifest["parameter_policy"],
        "cases": cases,
        "score_summary": summarize_results(results),
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
        description="Run every case in an offline validation suite."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    evidence = run_validation_suite(args.manifest, output_path=args.output)
    print(
        json.dumps(
            {
                "suite": evidence["suite"],
                "case_count": evidence["case_count"],
                "score_summary": evidence["score_summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
