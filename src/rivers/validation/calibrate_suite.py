"""Constrained fixed-parameter calibration across multiple observed events."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rivers.validation.run_case import run_validation_case


PARAMETER_DEFAULTS = {
    "manning_scale": 1.0,
    "width_scale": 1.0,
    "slope_scale": 1.0,
    "lateral_inflow_fraction": 0.0,
}


def composite_objective(
    event_scores,
    *,
    nse_weight=0.7,
    correlation_weight=0.3,
    robustness_penalty=0.1,
    correlation_robustness_penalty=0.0,
    worst_event_weight=0.0,
):
    """Reward event skill while penalizing inconsistent or weak events."""
    return objective_components(
        event_scores,
        nse_weight=nse_weight,
        correlation_weight=correlation_weight,
        robustness_penalty=robustness_penalty,
        correlation_robustness_penalty=(
            correlation_robustness_penalty
        ),
        worst_event_weight=worst_event_weight,
    )["objective"]


def objective_components(
    event_scores,
    *,
    nse_weight=0.7,
    correlation_weight=0.3,
    robustness_penalty=0.1,
    correlation_robustness_penalty=0.0,
    worst_event_weight=0.0,
):
    """Return an auditable decomposition of the multi-event objective."""
    if not event_scores:
        raise ValueError("At least one training event is required")
    if min(
        nse_weight,
        correlation_weight,
        robustness_penalty,
        correlation_robustness_penalty,
        worst_event_weight,
    ) < 0:
        raise ValueError("Objective weights must be non-negative")
    if not math.isclose(nse_weight + correlation_weight, 1.0):
        raise ValueError("NSE and correlation weights must sum to 1")
    nse = np.asarray([score["nse"] for score in event_scores], dtype=float)
    correlation = np.asarray(
        [score["pearson_r"] for score in event_scores], dtype=float
    )
    if not np.all(np.isfinite(nse)) or not np.all(np.isfinite(correlation)):
        return {
            "objective": -math.inf,
            "finite": False,
        }
    mean_nse = float(np.mean(nse))
    mean_correlation = float(np.mean(correlation))
    nse_spread = float(np.std(nse))
    correlation_spread = float(np.std(correlation))
    minimum_nse = float(np.min(nse))
    worst_event_gap = mean_nse - minimum_nse
    mean_skill = (
        nse_weight * mean_nse
        + correlation_weight * mean_correlation
    )
    nse_spread_penalty = robustness_penalty * nse_spread
    correlation_spread_penalty = (
        correlation_robustness_penalty * correlation_spread
    )
    worst_event_penalty = worst_event_weight * worst_event_gap
    return {
        "objective": float(
            mean_skill
            - nse_spread_penalty
            - correlation_spread_penalty
            - worst_event_penalty
        ),
        "finite": True,
        "mean_nse": mean_nse,
        "minimum_nse": minimum_nse,
        "mean_correlation": mean_correlation,
        "nse_standard_deviation": nse_spread,
        "correlation_standard_deviation": correlation_spread,
        "worst_event_nse_gap": worst_event_gap,
        "weighted_mean_skill": float(mean_skill),
        "nse_spread_penalty": float(nse_spread_penalty),
        "correlation_spread_penalty": float(
            correlation_spread_penalty
        ),
        "worst_event_penalty": float(worst_event_penalty),
    }


def parameter_overrides(config, parameters):
    """Map global dimensionless parameters to one event's physical inputs."""
    reach = config["reach"]
    width_scale = float(parameters["width_scale"])
    lateral_fraction = float(parameters["lateral_inflow_fraction"])
    cross_section_shape = reach.get("cross_section_shape", "rectangular")
    if cross_section_shape != "rectangular" and not math.isclose(
        width_scale, 1.0
    ):
        raise ValueError(
            "width_scale cannot modify reviewed stage-dependent geometry; "
            "use width_scale=[1.0] for trapezoidal, compound, or surveyed cases"
        )
    if config.get("point_flow_series") is not None and not math.isclose(
        lateral_fraction, 0.0
    ):
        raise ValueError(
            "lateral_inflow_fraction cannot be fitted when measured "
            "point_flow_series are present; use "
            "lateral_inflow_fraction=[0.0]"
        )
    overrides = {
        "reach": {
            "manning_n": float(reach["manning_n"])
            * float(parameters["manning_scale"]),
            "slope": float(reach["slope"]) * float(parameters["slope_scale"]),
        },
        "lateral_inflow_fraction": lateral_fraction,
    }
    if cross_section_shape == "rectangular":
        overrides["reach"].update(
            {
                "upstream_width_m": (
                    float(reach["upstream_width_m"]) * width_scale
                ),
                "downstream_width_m": (
                    float(reach["downstream_width_m"]) * width_scale
                ),
            }
        )
    return overrides


def coordinate_search(
    evaluate,
    parameter_grid,
    *,
    initial=None,
    passes=2,
):
    """Deterministic coordinate search with an auditable evaluation trace."""
    if passes < 1:
        raise ValueError("passes must be at least 1")
    parameters = dict(PARAMETER_DEFAULTS if initial is None else initial)
    unknown = set(parameter_grid) - set(parameters)
    if unknown:
        raise ValueError(f"Unknown calibration parameters: {sorted(unknown)}")
    trace = []
    current = evaluate(parameters)
    trace.append({"step": "initial", **current})

    for pass_index in range(passes):
        changed = False
        for name, values in parameter_grid.items():
            candidates = []
            for value in values:
                candidate_parameters = dict(parameters)
                candidate_parameters[name] = float(value)
                evaluation = evaluate(candidate_parameters)
                candidates.append(evaluation)
            current_value = float(parameters[name])
            selected = max(
                candidates,
                key=lambda item: (
                    item["objective"],
                    -abs(float(item["parameters"][name]) - current_value),
                ),
            )
            if selected["parameters"] != parameters:
                changed = True
            parameters = dict(selected["parameters"])
            current = selected
            trace.append(
                {
                    "step": "coordinate",
                    "pass": pass_index + 1,
                    "parameter": name,
                    **selected,
                }
            )
        if not changed:
            break
    return current, trace


def _score_summary(cases):
    metrics = ("nse", "rmse", "bias", "percent_bias", "pearson_r")
    summary = {}
    for metric in metrics:
        values = np.asarray([case["scores"][metric] for case in cases], dtype=float)
        summary[metric] = {
            "minimum": float(np.min(values)),
            "mean": float(np.mean(values)),
            "maximum": float(np.max(values)),
        }
    return summary


def parameter_diagnostics(parameter_grid, selected):
    """Flag optima at search bounds, where calibration is not well identified."""
    boundary_hits = {}
    for name, values in parameter_grid.items():
        numeric = [float(value) for value in values]
        value = float(selected[name])
        boundaries = []
        if math.isclose(value, min(numeric)):
            boundaries.append("minimum")
        if math.isclose(value, max(numeric)):
            boundaries.append("maximum")
        if boundaries and len(set(numeric)) > 1:
            boundary_hits[name] = boundaries
    return {
        "boundary_hits": boundary_hits,
        "identifiability_warning": bool(boundary_hits),
        "interpretation": (
            "A boundary hit means the tested range did not bracket an optimum. "
            "Geometry, slope, roughness, and lateral gain may compensate for one "
            "another; selected values are effective model parameters until supported "
            "by independent measurements."
        ),
    }


def _baseline_splits(manifest_path, split):
    baseline = {}
    for split_name in ("training", "validation", "test"):
        events = []
        for relative_path in split.get(split_name, []):
            tracked_path = (
                manifest_path.parent / relative_path
            ).with_suffix(".results.json")
            tracked = json.loads(tracked_path.read_text(encoding="utf-8"))
            events.append(
                {
                    "config": relative_path,
                    "scores": tracked["scores"],
                }
            )
        baseline[split_name] = {
            "events": events,
            "summary": _score_summary(events) if events else None,
        }
    return baseline


def _improvement(baseline, calibrated):
    comparison = {}
    for split_name in ("training", "validation", "test"):
        before = baseline[split_name]["summary"]
        after = calibrated[split_name]["summary"]
        if before is None or after is None:
            comparison[split_name] = None
            continue
        comparison[split_name] = {
            "mean_nse_delta": after["nse"]["mean"] - before["nse"]["mean"],
            "mean_pearson_r_delta": (
                after["pearson_r"]["mean"] - before["pearson_r"]["mean"]
            ),
            "mean_absolute_percent_bias_before": float(
                np.mean(
                    [
                        abs(event["scores"]["percent_bias"])
                        for event in baseline[split_name]["events"]
                    ]
                )
            ),
            "mean_absolute_percent_bias_after": float(
                np.mean(
                    [
                        abs(event["scores"]["percent_bias"])
                        for event in calibrated[split_name]["events"]
                    ]
                )
            ),
        }
    return comparison


def _leave_one_training_event_out(
    *,
    manifest,
    manifest_path,
    training_paths,
    configs,
    objective_config,
    temp_root,
):
    """Refit on all-but-one training event and score the omitted event."""
    settings = manifest.get("cross_event_validation", {})
    enabled = bool(settings.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "interpretation": (
                "Leave-one-training-event-out refits were not requested."
            ),
            "folds": [],
        }
    if len(training_paths) < 2:
        raise ValueError(
            "Leave-one-event-out validation needs at least two training events"
        )
    passes = int(settings.get("passes", 1))
    if passes < 1:
        raise ValueError(
            "cross_event_validation passes must be at least 1"
        )

    folds = []
    for fold_index, held_out_path in enumerate(training_paths):
        fold_training = [
            path for path in training_paths if path != held_out_path
        ]
        cache = {}

        def evaluate_fold(parameters):
            key = tuple(sorted(parameters.items()))
            if key in cache:
                return cache[key]
            events = []
            for event_index, relative_path in enumerate(fold_training):
                result = run_validation_case(
                    manifest_path.parent / relative_path,
                    output_path=(
                        temp_root
                        / f"loeo-{fold_index}-train-{len(cache)}-"
                        f"{event_index}.json"
                    ),
                    overrides=parameter_overrides(
                        configs[relative_path], parameters
                    ),
                )
                events.append(
                    {
                        "config": relative_path,
                        "scores": result["scores"],
                    }
                )
            components = objective_components(
                [event["scores"] for event in events],
                **objective_config,
            )
            evaluation = {
                "parameters": dict(parameters),
                "objective": components["objective"],
                "objective_components": components,
                "training_events": events,
            }
            cache[key] = evaluation
            return evaluation

        selected, trace = coordinate_search(
            evaluate_fold,
            manifest["parameter_grid"],
            passes=passes,
        )
        held_out_result = run_validation_case(
            manifest_path.parent / held_out_path,
            output_path=temp_root / f"loeo-{fold_index}-held-out.json",
            overrides=parameter_overrides(
                configs[held_out_path], selected["parameters"]
            ),
        )
        folds.append(
            {
                "held_out_event": held_out_path,
                "fit_events": fold_training,
                "selected_parameters": selected["parameters"],
                "fit_objective": selected["objective"],
                "fit_objective_components": selected[
                    "objective_components"
                ],
                "held_out_scores": held_out_result["scores"],
                "parameter_diagnostics": parameter_diagnostics(
                    manifest["parameter_grid"],
                    selected["parameters"],
                ),
                "unique_parameter_sets_evaluated": len(cache),
                "trace": trace,
            }
        )
    held_out_cases = [
        {
            "config": fold["held_out_event"],
            "scores": fold["held_out_scores"],
        }
        for fold in folds
    ]
    return {
        "enabled": True,
        "passes": passes,
        "fold_count": len(folds),
        "held_out_summary": _score_summary(held_out_cases),
        "interpretation": (
            "Each fold selected global parameters without the reported event, "
            "then scored that event once. These events are historical and have "
            "been inspected, so this diagnoses transfer but is not prospective "
            "validation."
        ),
        "folds": folds,
    }


def calibrate_suite(manifest_path, *, output_path=None, passes=2):
    """Optimize global parameters on training events and evaluate untouched splits."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = manifest["split"]
    training_paths = list(split["training"])
    if not training_paths:
        raise ValueError("Calibration manifest needs training cases")
    nontraining_paths = list(split.get("validation", [])) + list(
        split.get("test", [])
    )
    if set(training_paths) & set(nontraining_paths):
        raise ValueError("Training and held-out cases must not overlap")

    configs = {}
    for relative_path in training_paths + nontraining_paths:
        path = manifest_path.parent / relative_path
        configs[relative_path] = json.loads(path.read_text(encoding="utf-8"))

    objective_config = manifest["objective"]
    cache = {}
    with tempfile.TemporaryDirectory(prefix="climate-model-calibration-") as temp_dir:
        temp_root = Path(temp_dir)

        def evaluate_training(parameters):
            key = tuple(sorted(parameters.items()))
            if key in cache:
                return cache[key]
            events = []
            for index, relative_path in enumerate(training_paths):
                config_path = manifest_path.parent / relative_path
                result = run_validation_case(
                    config_path,
                    output_path=temp_root / f"train-{len(cache)}-{index}.json",
                    overrides=parameter_overrides(
                        configs[relative_path], parameters
                    ),
                )
                events.append(
                    {
                        "config": relative_path,
                        "scores": result["scores"],
                    }
                )
            components = objective_components(
                [event["scores"] for event in events],
                **objective_config,
            )
            evaluation = {
                "parameters": dict(parameters),
                "objective": components["objective"],
                "objective_components": components,
                "training_events": events,
            }
            cache[key] = evaluation
            return evaluation

        selected, trace = coordinate_search(
            evaluate_training,
            manifest["parameter_grid"],
            passes=passes,
        )

        evaluated_splits = {}
        for split_name in ("training", "validation", "test"):
            events = []
            for index, relative_path in enumerate(split.get(split_name, [])):
                config_path = manifest_path.parent / relative_path
                result = run_validation_case(
                    config_path,
                    output_path=temp_root / f"{split_name}-{index}.json",
                    overrides=parameter_overrides(
                        configs[relative_path], selected["parameters"]
                    ),
                )
                events.append(
                    {
                        "config": relative_path,
                        "scores": result["scores"],
                    }
                )
            evaluated_splits[split_name] = {
                "events": events,
                "summary": _score_summary(events) if events else None,
            }
        leave_one_out = _leave_one_training_event_out(
            manifest=manifest,
            manifest_path=manifest_path,
            training_paths=training_paths,
            configs=configs,
            objective_config=objective_config,
            temp_root=temp_root,
        )

    baseline_splits = _baseline_splits(manifest_path, split)
    evidence = {
        "schema_version": 2,
        "suite": manifest["suite"],
        "method": {
            "optimizer": "deterministic coordinate search",
            "passes_requested": int(passes),
            "global_parameters_only": True,
            "event_specific_fitting": False,
            "objective": objective_config,
            "parameter_grid": manifest["parameter_grid"],
            "parameter_interpretation": manifest.get(
                "parameter_interpretation", {}
            ),
            "split_policy": manifest["split_policy"],
        },
        "selected_parameters": selected["parameters"],
        "parameter_diagnostics": parameter_diagnostics(
            manifest["parameter_grid"], selected["parameters"]
        ),
        "training_objective": selected["objective"],
        "training_objective_components": selected[
            "objective_components"
        ],
        "leave_one_training_event_out": leave_one_out,
        "baseline_splits": baseline_splits,
        "splits": evaluated_splits,
        "improvement": _improvement(baseline_splits, evaluated_splits),
        "trace": trace,
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
        description="Calibrate global hydraulic parameters across observed events."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--passes", type=int, default=2)
    args = parser.parse_args(argv)
    evidence = calibrate_suite(
        args.manifest,
        output_path=args.output,
        passes=args.passes,
    )
    print(
        json.dumps(
            {
                "selected_parameters": evidence["selected_parameters"],
                "training_objective": evidence["training_objective"],
                "splits": {
                    name: value["summary"]
                    for name, value in evidence["splits"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
