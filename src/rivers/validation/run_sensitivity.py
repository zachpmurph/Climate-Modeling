"""Run explicit 2-D structural sensitivities for an observed case."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rivers.validation.run_case_2d import run_validation_case_2d


def build_variants(config):
    """Return fixed 2-D terrain/resolution experiments, never fitted values."""
    baseline = config["validation_2d"]
    x_cells = int(baseline["x_cells"])
    return [
        (
            "baseline_2d_ribbon",
            {
                "representation": "ribbon",
                "x_cells": x_cells,
                "y_cells": int(baseline["y_cells"]),
            },
            {"purpose": "canonical constant-width 2-D screening corridor"},
        ),
        (
            "fine_longitudinal_2d_ribbon",
            {
                "representation": "ribbon",
                "x_cells": 2 * x_cells - 1,
                "y_cells": int(baseline["y_cells"]),
            },
            {"purpose": "test longitudinal numerical diffusion"},
        ),
        (
            "connected_lateral_shelves_2d",
            {
                "representation": "shelf",
                "x_cells": x_cells,
                "y_cells": 9,
                "floodplain_width_factor": 3.0,
                "bank_height_factor": 1.25,
            },
            {"purpose": "test lateral storage and bank connectivity"},
        ),
    ]


def run_sensitivity(config_path, *, output_path=None):
    """Evaluate 2-D structural sensitivity without choosing a best variant."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runs = []
    with tempfile.TemporaryDirectory(prefix="climate-model-2d-sensitivity-") as temp:
        for name, settings, changes in build_variants(config):
            result = run_validation_case_2d(
                config_path,
                output_path=Path(temp) / f"{name}.json",
                **settings,
            )
            runs.append(
                {
                    "name": name,
                    "settings": settings,
                    "changes": changes,
                    "scores": result["scores"],
                    "mass": result["mass"],
                    "reach_diagnosis": result["reach_diagnosis"],
                }
            )

    metrics = ("nse", "rmse", "bias", "percent_bias", "pearson_r")
    ranges = {}
    for metric in metrics:
        values = [
            float(run["scores"][metric])
            for run in runs
            if run["scores"][metric] is not None
            and math.isfinite(float(run["scores"][metric]))
        ]
        ranges[metric] = {"minimum": min(values), "maximum": max(values)}
    evidence = {
        "schema_version": 2,
        "case": config["case"],
        "solver_policy": "saint_venant_2d_only",
        "method": {
            "type": "one_at_a_time_structural_2d",
            "purpose": (
                "Diagnose numerical diffusion and lateral storage only. "
                "No best-scoring variant is selected or transferred."
            ),
        },
        "runs": runs,
        "score_ranges": ranges,
    }
    destination = (
        Path(output_path)
        if output_path is not None
        else config_path.with_suffix(".sensitivity.json")
    )
    destination.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run 2-D structural sensitivities for an observed case."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    evidence = run_sensitivity(args.config, output_path=args.output)
    print(
        json.dumps(
            {
                "case": evidence["case"],
                "solver_policy": evidence["solver_policy"],
                "runs": len(evidence["runs"]),
                "score_ranges": evidence["score_ranges"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
