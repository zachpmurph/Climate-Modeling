"""Assemble the committed 2-D-only error-source screening evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rivers.validation.diagnose import diagnose_hydrograph


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _diagnosis(run):
    series = run["series"]
    if "times_min" in series:
        times = series["times_min"]
    else:
        times = series["observed_times_min"]
    return diagnose_hydrograph(
        times,
        series["observed_downstream_m3_per_min"],
        series["predicted_downstream_m3_per_min"],
    )


def _summary(run):
    diagnosis = _diagnosis(run)
    reach = run.get("reach_diagnosis")
    return {
        "solver": run["solver"],
        "terrain": run.get("terrain_representation"),
        "scores": run["scores"],
        "volume_ratio": diagnosis["volume_ratio"],
        "amplitude_ratio": diagnosis["amplitude_ratio"],
        "peak_lag_min": diagnosis["peak_lag_min"],
        "volume_only_counterfactual": diagnosis[
            "volume_only_counterfactual"
        ],
        "reach_diagnosis": reach,
    }


def _delta(treatment, control, field):
    return float(treatment[field]) - float(control[field])


def assess_error_sources(manifest_path, *, output_path=None):
    manifest_path = Path(manifest_path)
    manifest = _load(manifest_path)
    runs = {
        name: _load(manifest_path.parent / relative)
        for name, relative in manifest["runs"].items()
    }
    summaries = {name: _summary(run) for name, run in runs.items()}

    colorado_ribbon = summaries["colorado_ribbon"]
    colorado_shelf = summaries["colorado_shelf"]
    rio_ribbon = summaries["rio_ribbon"]
    rio_shelf = summaries["rio_shelf"]
    rio_measured = summaries["rio_measured_geometry"]
    colorado_fine = summaries["colorado_ribbon_fine"]
    rio_fine = summaries["rio_ribbon_fine"]
    truckee = summaries["truckee_ribbon"]

    evidence = {
        "schema_version": 1,
        "study": manifest["study"],
        "status": manifest["status"],
        "solver_policy": "saint_venant_2d_only",
        "controls": manifest["controls"],
        "runs": summaries,
        "error_sources": {
            "geometry_and_storage": {
                "colorado_ribbon_to_shelf": {
                    "nse_change": _delta(
                        colorado_shelf["scores"],
                        colorado_ribbon["scores"],
                        "nse",
                    ),
                    "modeled_net_change_fraction": [
                        colorado_ribbon["reach_diagnosis"][
                            "modeled_net_change_fraction"
                        ],
                        colorado_shelf["reach_diagnosis"][
                            "modeled_net_change_fraction"
                        ],
                    ],
                    "routing_lag_error_min": [
                        colorado_ribbon["reach_diagnosis"][
                            "routing_lag_error_min"
                        ],
                        colorado_shelf["reach_diagnosis"][
                            "routing_lag_error_min"
                        ],
                    ],
                    "finding": (
                        "Adding connected shelves nearly matches the observed "
                        "routing lag but stores far too much event water and "
                        "degrades NSE. Colorado is highly sensitive to unknown "
                        "bank/floodplain connectivity."
                    ),
                },
                "rio_ribbon_to_shelf": {
                    "nse_change": _delta(
                        rio_shelf["scores"], rio_ribbon["scores"], "nse"
                    ),
                    "unexplained_volume_change_m3": _delta(
                        rio_shelf["reach_diagnosis"],
                        rio_ribbon["reach_diagnosis"],
                        "unexplained_downstream_volume_m3",
                    ),
                    "routing_lag_error_change_min": _delta(
                        rio_shelf["reach_diagnosis"],
                        rio_ribbon["reach_diagnosis"],
                        "routing_lag_error_min",
                    ),
                    "finding": (
                        "The idealized Rio shelves do not change routing lag or "
                        "event volume materially, so this screened lateral-storage "
                        "mechanism does not explain the Rio failure."
                    ),
                },
                "rio_proxy_to_measured_sections": {
                    "nse_change": _delta(
                        rio_measured["scores"], rio_ribbon["scores"], "nse"
                    ),
                    "correlation_change": _delta(
                        rio_measured["scores"],
                        rio_ribbon["scores"],
                        "pearson_r",
                    ),
                    "routing_lag_error_change_min": _delta(
                        rio_measured["reach_diagnosis"],
                        rio_ribbon["reach_diagnosis"],
                        "routing_lag_error_min",
                    ),
                    "finding": (
                        "Using field-derived endpoint bed and roughness with a "
                        "constant-width 2-D ribbon does not resolve the Rio error. "
                        "The ribbon cannot test longitudinal width variation, so "
                        "continuous geometry remains an open error source."
                    ),
                },
            },
            "intervening_flows_and_storage": {
                "colorado": {
                    "observed_net_change_fraction": colorado_ribbon[
                        "reach_diagnosis"
                    ]["observed_net_change_fraction"],
                    "modeled_net_change_fraction": colorado_ribbon[
                        "reach_diagnosis"
                    ]["modeled_net_change_fraction"],
                    "unexplained_downstream_volume_m3": colorado_ribbon[
                        "reach_diagnosis"
                    ]["unexplained_downstream_volume_m3"],
                    "volume_only_nse": colorado_ribbon[
                        "volume_only_counterfactual"
                    ]["scores"]["nse"],
                    "finding": (
                        "The observed reach loses or stores about 4.2% of upstream "
                        "event volume, while the ribbon loses or stores about "
                        "11.7%. Excess modeled storage or missing positive reach "
                        "flow explains a material part, but not all, of the error."
                    ),
                },
                "rio": {
                    "observed_net_change_fraction": rio_ribbon[
                        "reach_diagnosis"
                    ]["observed_net_change_fraction"],
                    "modeled_net_change_fraction": rio_ribbon[
                        "reach_diagnosis"
                    ]["modeled_net_change_fraction"],
                    "unexplained_downstream_volume_m3": rio_ribbon[
                        "reach_diagnosis"
                    ]["unexplained_downstream_volume_m3"],
                    "volume_only_nse": rio_ribbon[
                        "volume_only_counterfactual"
                    ]["scores"]["nse"],
                    "finding": (
                        "The observed Rio reach loses or stores about 4.8% of "
                        "upstream volume, versus only 0.35% in the model. A missing "
                        "withdrawal/storage term explains the mean-volume bias, "
                        "but oracle volume matching still leaves NSE below zero "
                        "and low correlation."
                    ),
                },
            },
            "travel_time_attenuation_and_numerics": {
                "colorado_grid_refinement": {
                    "nse_change_31_to_61_cells": _delta(
                        colorado_fine["scores"],
                        colorado_ribbon["scores"],
                        "nse",
                    ),
                    "correlation_change_31_to_61_cells": _delta(
                        colorado_fine["scores"],
                        colorado_ribbon["scores"],
                        "pearson_r",
                    ),
                    "finding": (
                        "Doubling longitudinal resolution changes neither the "
                        "score nor routing lag materially; coarse-grid diffusion "
                        "is not the leading Colorado error."
                    ),
                },
                "rio_grid_refinement": {
                    "nse_change_31_to_61_cells": _delta(
                        rio_fine["scores"], rio_ribbon["scores"], "nse"
                    ),
                    "correlation_change_31_to_61_cells": _delta(
                        rio_fine["scores"],
                        rio_ribbon["scores"],
                        "pearson_r",
                    ),
                    "finding": (
                        "Rio remains too attenuated and badly mistimed after "
                        "grid refinement; numerical resolution is not the cause."
                    ),
                },
                "rio_shape": {
                    "amplitude_ratio": rio_ribbon["amplitude_ratio"],
                    "peak_lag_min": rio_ribbon["peak_lag_min"],
                    "routing_lag_error_min": rio_ribbon["reach_diagnosis"][
                        "routing_lag_error_min"
                    ],
                    "correlation": rio_ribbon["scores"]["pearson_r"],
                    "finding": (
                        "The Rio model has only about 72% of observed hydrograph "
                        "range, peaks much too early within the event window, and "
                        "has low shape correlation. This is a structural routing "
                        "failure, not only a volume bias."
                    ),
                },
                "cross_region_control": {
                    "truckee_nse": truckee["scores"]["nse"],
                    "truckee_correlation": truckee["scores"]["pearson_r"],
                    "finding": (
                        "The same 2-D ribbon pathway reproduces Truckee well, so "
                        "the core solver is not uniformly inaccurate; the failure "
                        "depends on reach/event representation."
                    ),
                },
            },
        },
        "overall_finding": (
            "Colorado error is dominated by uncertain reach storage/connectivity "
            "and event-volume retention. Rio error includes a missing net loss/"
            "storage term, but its dominant remaining problem is hydrograph "
            "timing and attenuation that survives 2-D routing, shelf storage, "
            "field-derived endpoint bed/roughness, and grid refinement. Continuous "
            "topobathymetry plus measured diversions/returns are required before "
            "distinguishing unresolved geometry from mobile-bed or other "
            "structural river dynamics."
        ),
        "limitations": [
            "No run uses surveyed continuous topobathymetry.",
            "Shelf elevations and connectivity are controlled screening assumptions.",
            "The field-geometry run interpolates between only two endpoint sections.",
            "Volume-only counterfactuals use held-out observations and are diagnostics, not calibration.",
            "Observed upstream-downstream volume differences also include gauge uncertainty and end-of-window storage.",
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
        description="Assemble committed 2-D-only error-source evidence."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    evidence = assess_error_sources(args.manifest, output_path=args.output)
    print(json.dumps(evidence["error_sources"], indent=2))
    print(json.dumps({"overall_finding": evidence["overall_finding"]}, indent=2))


if __name__ == "__main__":
    main()
