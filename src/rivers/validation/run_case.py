"""Canonical observed-event entry point: Saint-Venant 2-D only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Re-export stable input/control helpers used by fixture tests and fetch tools.
from rivers.validation.case_inputs import (  # noqa: F401
    _require_control_coverage,
    discharge_boundary,
    load_event_control_series,
    load_field_measurement_geometry,
    load_two_gauge_observations,
    shifted_boundary,
)
from rivers.validation.run_case_2d import run_validation_case_2d


def run_validation_case(config_path, *, output_path=None, overrides=None):
    """Run one observed case through Saint-Venant 2-D; never fall back to 1-D."""
    if overrides:
        raise ValueError(
            "Generic case overrides are disabled. Use explicit 2-D structural "
            "variants in run_sensitivity.py."
        )
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    settings = config.get("validation_2d")
    if not isinstance(settings, dict):
        raise ValueError(
            "Every observed case must declare validation_2d settings; "
            "1-D fallback is forbidden"
        )
    destination = (
        Path(output_path)
        if output_path is not None
        else config_path.with_suffix(".results.json")
    )
    result = run_validation_case_2d(
        config_path,
        representation=settings["representation"],
        x_cells=int(settings["x_cells"]),
        y_cells=int(settings["y_cells"]),
        floodplain_width_factor=float(
            settings.get("floodplain_width_factor", 3.0)
        ),
        bank_height_factor=float(settings.get("bank_height_factor", 1.25)),
        output_path=destination,
    )
    if result["solver"] != "saint_venant_2d":
        raise RuntimeError("Canonical validation produced a non-2-D result")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run an observed event through Saint-Venant 2-D only."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    evidence = run_validation_case(args.config, output_path=args.output)
    print(
        json.dumps(
            {
                "case": evidence["case"],
                "solver": evidence["solver"],
                "terrain": evidence["terrain_representation"],
                "scores": evidence["scores"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
