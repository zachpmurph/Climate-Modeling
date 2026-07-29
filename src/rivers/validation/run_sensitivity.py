"""Run one-at-a-time structural sensitivities for a two-gauge validation case."""

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

from rivers.validation.run_case import run_validation_case


def build_variants(config):
    """Return named, auditable perturbations around the configured baseline."""
    reach = config["reach"]
    roughness = float(reach["manning_n"])
    upstream_width = float(reach["upstream_width_m"])
    downstream_width = float(reach["downstream_width_m"])
    cells = int(reach["cells"])
    coarse_cells = max(11, (cells + 1) // 2)
    fine_cells = 2 * cells - 1
    return [
        ("baseline", {}, {}),
        (
            "roughness_minus_20_percent",
            {"reach": {"manning_n": 0.8 * roughness}},
            {"manning_n_factor": 0.8},
        ),
        (
            "roughness_plus_20_percent",
            {"reach": {"manning_n": 1.2 * roughness}},
            {"manning_n_factor": 1.2},
        ),
        (
            "width_minus_20_percent",
            {
                "reach": {
                    "upstream_width_m": 0.8 * upstream_width,
                    "downstream_width_m": 0.8 * downstream_width,
                }
            },
            {"channel_width_factor": 0.8},
        ),
        (
            "width_plus_20_percent",
            {
                "reach": {
                    "upstream_width_m": 1.2 * upstream_width,
                    "downstream_width_m": 1.2 * downstream_width,
                }
            },
            {"channel_width_factor": 1.2},
        ),
        ("coarse_grid", {"reach": {"cells": coarse_cells}}, {"cells": coarse_cells}),
        ("fine_grid", {"reach": {"cells": fine_cells}}, {"cells": fine_cells}),
        (
            "second_order_reconstruction",
            {"spatial_order": 2},
            {"spatial_order": 2},
        ),
    ]


def run_sensitivity(config_path, *, output_path=None):
    """Evaluate structural sensitivity without calibrating to the held-out gauge."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runs = []
    with tempfile.TemporaryDirectory(prefix="climate-model-sensitivity-") as temp_dir:
        for name, overrides, changes in build_variants(config):
            result = run_validation_case(
                config_path,
                output_path=Path(temp_dir) / f"{name}.json",
                overrides=overrides,
            )
            runs.append(
                {
                    "name": name,
                    "changes": changes,
                    "scores": result["scores"],
                    "mass": result["mass"],
                }
            )

    metrics = ("nse", "rmse", "bias", "percent_bias", "pearson_r")
    score_ranges = {}
    for metric in metrics:
        values = [
            float(run["scores"][metric])
            for run in runs
            if run["scores"][metric] is not None
            and math.isfinite(float(run["scores"][metric]))
        ]
        score_ranges[metric] = {"minimum": min(values), "maximum": max(values)}

    evidence = {
        "schema_version": 1,
        "case": config["case"],
        "method": {
            "type": "one_at_a_time",
            "purpose": (
                "Structural sensitivity screening only; downstream observations "
                "remain held out and no parameter is calibrated."
            ),
            "parameters": [
                "Manning roughness +/-20%",
                "channel width +/-20%",
                "longitudinal cells approximately halved/doubled",
                "first- versus second-order spatial reconstruction",
            ],
        },
        "runs": runs,
        "score_ranges": score_ranges,
    }
    destination = (
        Path(output_path)
        if output_path is not None
        else config_path.with_suffix(".sensitivity.json")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run structural sensitivities for a real-river validation case."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    evidence = run_sensitivity(args.config, output_path=args.output)
    print(
        json.dumps(
            {
                "case": evidence["case"],
                "runs": len(evidence["runs"]),
                "score_ranges": evidence["score_ranges"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
