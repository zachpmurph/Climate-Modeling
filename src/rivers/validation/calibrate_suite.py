"""Constrained fixed-parameter calibration across multiple observed events."""

from __future__ import annotations

import argparse
import copy
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


def structural_variants(manifest):
    configured = manifest.get("structural_variants")
    if configured is None:
        return [{"name": "configured_baseline", "overrides": {}}]
    if not isinstance(configured, list) or not configured:
        raise ValueError("structural_variants must be a non-empty list")
    variants = []
    names = set()
    for item in configured:
        if not isinstance(item, dict):
            raise ValueError("Each structural variant must be an object")
        name = str(item.get("name", "")).strip()
        overrides = item.get("overrides", {})
        if not name or name in names:
            raise ValueError(
                "Structural variant names must be non-empty and unique"
            )
        if not isinstance(overrides, dict):
            raise ValueError(
                f"Structural variant {name!r} overrides must be an object"
            )
        names.add(name)
        variants.append({"name": name, "overrides": copy.deepcopy(overrides)})
    return variants


def variant_parameter_overrides(config, parameters, variant):
    """Combine one named structural model with calibrated physical scales."""
    variant_config = copy.deepcopy(config)
    structural = copy.deepcopy(variant["overrides"])
    variant_config.setdefault("reach", {}).update(
        structural.get("reach", {})
    )
    for key, value in structural.items():
        if key != "reach":
            variant_config[key] = value
    calibrated = parameter_overrides(variant_config, parameters)
    combined = structural
    combined.setdefault("reach", {}).update(calibrated["reach"])
    for key, value in calibrated.items():
        if key != "reach":
            combined[key] = value
    return combined


def composite_objective(
    event_scores,
    *,
    group_labels=None,
    nse_weight=0.7,
    correlation_weight=0.3,
    robustness_penalty=0.1,
    correlation_robustness_penalty=0.0,
    worst_event_weight=0.0,
):
    """Reward event skill while penalizing inconsistent or weak events."""
    return objective_components(
        event_scores,
        group_labels=group_labels,
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
    group_labels=None,
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
    event_nse = np.asarray(
        [score["nse"] for score in event_scores], dtype=float
    )
    event_correlation = np.asarray(
        [score["pearson_r"] for score in event_scores], dtype=float
    )
    if not np.all(np.isfinite(event_nse)) or not np.all(
        np.isfinite(event_correlation)
    ):
        return {
            "objective": -math.inf,
            "finite": False,
        }
    group_summary = None
    if group_labels is None:
        nse = event_nse
        correlation = event_correlation
    else:
        if len(group_labels) != len(event_scores):
            raise ValueError(
                "group_labels must contain one label per event score"
            )
        groups = []
        for label in group_labels:
            label = str(label).strip()
            if label not in groups:
                groups.append(label)
        if not groups or any(not label for label in groups):
            raise ValueError("group_labels must be non-empty")
        nse = np.asarray(
            [
                np.mean(
                    event_nse[
                        np.asarray(
                            [
                                str(value).strip() == group
                                for value in group_labels
                            ]
                        )
                    ]
                )
                for group in groups
            ],
            dtype=float,
        )
        correlation = np.asarray(
            [
                np.mean(
                    event_correlation[
                        np.asarray(
                            [
                                str(value).strip() == group
                                for value in group_labels
                            ]
                        )
                    ]
                )
                for group in groups
            ],
            dtype=float,
        )
        group_summary = [
            {
                "group": group,
                "event_count": sum(
                    str(value).strip() == group for value in group_labels
                ),
                "mean_nse": float(group_nse),
                "mean_correlation": float(group_correlation),
            }
            for group, group_nse, group_correlation in zip(
                groups, nse, correlation
            )
        ]
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
    components = {
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
    if group_summary is not None:
        components.update(
            {
                "aggregation": "equal_weight_per_group",
                "group_count": len(group_summary),
                "group_summary": group_summary,
            }
        )
    return components


def parameter_overrides(config, parameters):
    """Map global dimensionless parameters to one event's physical inputs."""
    reach = config["reach"]
    width_scale = float(parameters["width_scale"])
    lateral_fraction = float(parameters["lateral_inflow_fraction"])
    cross_section_shape = reach.get("cross_section_shape", "rectangular")
    has_reviewed_width = (
        cross_section_shape != "rectangular"
        or reach.get("field_measurement_geometry") is not None
    )
    has_field_measurement_geometry = (
        reach.get("field_measurement_geometry") is not None
    )
    has_reviewed_roughness = (
        has_field_measurement_geometry
        or reach.get("stage_dependent_manning") is not None
    )
    if has_reviewed_width and not math.isclose(width_scale, 1.0):
        raise ValueError(
            "width_scale cannot modify reviewed field or stage-dependent "
            "geometry; use width_scale=[1.0] for measured, trapezoidal, "
            "compound, or surveyed cases"
        )
    if config.get("point_flow_series") is not None and not math.isclose(
        lateral_fraction, 0.0
    ):
        raise ValueError(
            "lateral_inflow_fraction cannot be fitted when measured "
            "point_flow_series are present; use "
            "lateral_inflow_fraction=[0.0]"
        )
    if has_reviewed_roughness and not math.isclose(
        float(parameters["manning_scale"]), 1.0
    ):
        raise ValueError(
            "manning_scale cannot modify field-inferred roughness or "
            "stage-dependent reviewed roughness; use manning_scale=[1.0]"
        )
    if has_field_measurement_geometry and not math.isclose(
        float(parameters["slope_scale"]), 1.0
    ):
        raise ValueError(
            "slope_scale cannot modify field-inferred bed elevation; use "
            "slope_scale=[1.0] for measured geometry cases"
        )
    overrides = {
        "reach": {
            "manning_n": float(reach["manning_n"])
            * float(parameters["manning_scale"]),
            "slope": float(reach["slope"]) * float(parameters["slope_scale"]),
        },
        "lateral_inflow_fraction": lateral_fraction,
    }
    if cross_section_shape == "rectangular" and not has_reviewed_width:
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


def search_structural_variants(
    evaluate,
    parameter_grid,
    variants,
    *,
    passes,
):
    """Fit parameters within each training-only structural variant."""
    selections = []
    traces = []
    for variant_index, variant in enumerate(variants):
        selected, trace = coordinate_search(
            lambda parameters, current=variant: evaluate(
                parameters, current
            ),
            parameter_grid,
            passes=passes,
        )
        selected = {
            **selected,
            "structural_variant": variant["name"],
            "structural_overrides": copy.deepcopy(variant["overrides"]),
        }
        selections.append(selected)
        traces.append(
            {
                "structural_variant": variant["name"],
                "structural_overrides": copy.deepcopy(
                    variant["overrides"]
                ),
                "trace": trace,
            }
        )
    selected_index, selected = max(
        enumerate(selections),
        key=lambda item: (item[1]["objective"], -item[0]),
    )
    return selected, selections, traces


def training_pareto_front(evaluations, *, selected_parameters=None):
    """Retain candidates not dominated on training NSE and correlation."""
    candidates = []
    seen = set()
    for evaluation in evaluations:
        parameters = dict(evaluation["parameters"])
        variant = evaluation.get(
            "structural_variant", "configured_baseline"
        )
        key = (variant, tuple(sorted(parameters.items())))
        if key in seen:
            continue
        seen.add(key)
        components = evaluation["objective_components"]
        if not components.get("finite", False):
            continue
        candidates.append(
            {
                "parameters": parameters,
                "structural_variant": variant,
                "mean_nse": float(components["mean_nse"]),
                "mean_correlation": float(
                    components["mean_correlation"]
                ),
                "minimum_nse": float(components["minimum_nse"]),
                "nse_standard_deviation": float(
                    components["nse_standard_deviation"]
                ),
                "correlation_standard_deviation": float(
                    components["correlation_standard_deviation"]
                ),
                "weighted_objective": float(evaluation["objective"]),
            }
        )
    tolerance = 1e-12
    front = []
    for candidate in candidates:
        dominated = any(
            (
                other["mean_nse"]
                >= candidate["mean_nse"] - tolerance
                and other["mean_correlation"]
                >= candidate["mean_correlation"] - tolerance
                and (
                    other["mean_nse"]
                    > candidate["mean_nse"] + tolerance
                    or other["mean_correlation"]
                    > candidate["mean_correlation"] + tolerance
                )
            )
            for other in candidates
            if other is not candidate
        )
        if not dominated:
            item = dict(candidate)
            item["selected_by_weighted_objective"] = (
                selected_parameters is not None
                and item["parameters"] == selected_parameters.get(
                    "parameters", selected_parameters
                )
                and item["structural_variant"]
                == selected_parameters.get(
                    "structural_variant", "configured_baseline"
                )
            )
            front.append(item)
    front.sort(
        key=lambda item: (
            -item["mean_nse"],
            -item["mean_correlation"],
            item["structural_variant"],
            tuple(sorted(item["parameters"].items())),
        )
    )
    return {
        "objectives": ["mean_nse", "mean_correlation"],
        "scope": (
            "Unique training-only parameter sets visited by deterministic "
            "coordinate search; validation and test scores are not used."
        ),
        "evaluated_candidate_count": len(candidates),
        "front_count": len(front),
        "candidates": front,
    }


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


def _group_labels(configs, paths, balance_by):
    if balance_by is None:
        return None
    labels = []
    for path in paths:
        case = configs[path].get("case", {})
        label = case.get(balance_by)
        if label is None or not str(label).strip():
            raise ValueError(
                f"{path} is missing case.{balance_by} required by the "
                "balanced objective"
            )
        labels.append(str(label))
    return labels


def _leave_one_training_event_out(
    *,
    manifest,
    manifest_path,
    training_paths,
    configs,
    objective_config,
    balance_by,
    variants,
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

        def evaluate_fold(parameters, variant):
            key = (
                variant["name"],
                tuple(sorted(parameters.items())),
            )
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
                    overrides=variant_parameter_overrides(
                        configs[relative_path], parameters, variant
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
                group_labels=_group_labels(
                    configs, fold_training, balance_by
                ),
                **objective_config,
            )
            evaluation = {
                "parameters": dict(parameters),
                "structural_variant": variant["name"],
                "objective": components["objective"],
                "objective_components": components,
                "training_events": events,
            }
            cache[key] = evaluation
            return evaluation

        selected, _, variant_traces = search_structural_variants(
            evaluate_fold,
            manifest["parameter_grid"],
            variants,
            passes=passes,
        )
        selected_variant = next(
            variant
            for variant in variants
            if variant["name"] == selected["structural_variant"]
        )
        held_out_result = run_validation_case(
            manifest_path.parent / held_out_path,
            output_path=temp_root / f"loeo-{fold_index}-held-out.json",
            overrides=variant_parameter_overrides(
                configs[held_out_path],
                selected["parameters"],
                selected_variant,
            ),
        )
        folds.append(
            {
                "held_out_event": held_out_path,
                "fit_events": fold_training,
                "selected_parameters": selected["parameters"],
                "selected_structural_variant": selected[
                    "structural_variant"
                ],
                "selected_structural_overrides": selected[
                    "structural_overrides"
                ],
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
                "trace": (
                    variant_traces[0]["trace"]
                    if len(variant_traces) == 1
                    else variant_traces
                ),
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


def _leave_one_group_out(
    *,
    manifest,
    manifest_path,
    training_paths,
    configs,
    objective_config,
    balance_by,
    variants,
    temp_root,
):
    """Refit without one river/group, then score that group once."""
    settings = manifest.get("cross_group_validation", {})
    enabled = bool(settings.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "interpretation": "Leave-one-group-out refits were not requested.",
            "folds": [],
        }
    if balance_by is None:
        raise ValueError(
            "cross_group_validation requires objective.balance_by"
        )
    labels = _group_labels(configs, training_paths, balance_by)
    groups = list(dict.fromkeys(labels))
    if len(groups) < 2:
        raise ValueError(
            "Leave-one-group-out validation needs at least two training groups"
        )
    passes = int(settings.get("passes", 1))
    if passes < 1:
        raise ValueError("cross_group_validation passes must be at least 1")

    folds = []
    for fold_index, held_out_group in enumerate(groups):
        fit_paths = [
            path
            for path, label in zip(training_paths, labels)
            if label != held_out_group
        ]
        held_out_paths = [
            path
            for path, label in zip(training_paths, labels)
            if label == held_out_group
        ]
        cache = {}

        def evaluate_fold(parameters, variant):
            key = (
                variant["name"],
                tuple(sorted(parameters.items())),
            )
            if key in cache:
                return cache[key]
            events = []
            for event_index, relative_path in enumerate(fit_paths):
                result = run_validation_case(
                    manifest_path.parent / relative_path,
                    output_path=(
                        temp_root
                        / f"logo-{fold_index}-train-{len(cache)}-"
                        f"{event_index}.json"
                    ),
                    overrides=variant_parameter_overrides(
                        configs[relative_path], parameters, variant
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
                group_labels=_group_labels(
                    configs, fit_paths, balance_by
                ),
                **objective_config,
            )
            evaluation = {
                "parameters": dict(parameters),
                "structural_variant": variant["name"],
                "objective": components["objective"],
                "objective_components": components,
                "training_events": events,
            }
            cache[key] = evaluation
            return evaluation

        selected, _, variant_traces = search_structural_variants(
            evaluate_fold,
            manifest["parameter_grid"],
            variants,
            passes=passes,
        )
        selected_variant = next(
            variant
            for variant in variants
            if variant["name"] == selected["structural_variant"]
        )
        held_out_events = []
        for event_index, relative_path in enumerate(held_out_paths):
            result = run_validation_case(
                manifest_path.parent / relative_path,
                output_path=(
                    temp_root
                    / f"logo-{fold_index}-held-{event_index}.json"
                ),
                overrides=variant_parameter_overrides(
                    configs[relative_path],
                    selected["parameters"],
                    selected_variant,
                ),
            )
            held_out_events.append(
                {
                    "config": relative_path,
                    "scores": result["scores"],
                }
            )
        folds.append(
            {
                "held_out_group": held_out_group,
                "held_out_events": held_out_events,
                "fit_events": fit_paths,
                "selected_parameters": selected["parameters"],
                "selected_structural_variant": selected[
                    "structural_variant"
                ],
                "selected_structural_overrides": selected[
                    "structural_overrides"
                ],
                "fit_objective": selected["objective"],
                "fit_objective_components": selected[
                    "objective_components"
                ],
                "held_out_summary": _score_summary(held_out_events),
                "parameter_diagnostics": parameter_diagnostics(
                    manifest["parameter_grid"],
                    selected["parameters"],
                ),
                "unique_parameter_sets_evaluated": len(cache),
                "trace": (
                    variant_traces[0]["trace"]
                    if len(variant_traces) == 1
                    else variant_traces
                ),
            }
        )
    return {
        "enabled": True,
        "balance_by": balance_by,
        "passes": passes,
        "fold_count": len(folds),
        "interpretation": (
            "Each fold selected global parameters without any event from the "
            "reported group, then scored every held-out event once."
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

    objective_config = dict(manifest["objective"])
    balance_by = objective_config.pop("balance_by", None)
    variants = structural_variants(manifest)
    cache = {}
    with tempfile.TemporaryDirectory(prefix="climate-model-calibration-") as temp_dir:
        temp_root = Path(temp_dir)

        def evaluate_training(parameters, variant):
            key = (
                variant["name"],
                tuple(sorted(parameters.items())),
            )
            if key in cache:
                return cache[key]
            events = []
            for index, relative_path in enumerate(training_paths):
                config_path = manifest_path.parent / relative_path
                result = run_validation_case(
                    config_path,
                    output_path=temp_root / f"train-{len(cache)}-{index}.json",
                    overrides=variant_parameter_overrides(
                        configs[relative_path], parameters, variant
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
                group_labels=_group_labels(
                    configs, training_paths, balance_by
                ),
                **objective_config,
            )
            evaluation = {
                "parameters": dict(parameters),
                "structural_variant": variant["name"],
                "objective": components["objective"],
                "objective_components": components,
                "training_events": events,
            }
            cache[key] = evaluation
            return evaluation

        selected, variant_selections, variant_traces = (
            search_structural_variants(
            evaluate_training,
            manifest["parameter_grid"],
            variants,
            passes=passes,
            )
        )
        pareto_front = training_pareto_front(
            cache.values(),
            selected_parameters=selected,
        )
        selected_variant = next(
            variant
            for variant in variants
            if variant["name"] == selected["structural_variant"]
        )

        evaluated_splits = {}
        for split_name in ("training", "validation", "test"):
            events = []
            for index, relative_path in enumerate(split.get(split_name, [])):
                config_path = manifest_path.parent / relative_path
                result = run_validation_case(
                    config_path,
                    output_path=temp_root / f"{split_name}-{index}.json",
                    overrides=variant_parameter_overrides(
                        configs[relative_path],
                        selected["parameters"],
                        selected_variant,
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
            balance_by=balance_by,
            variants=variants,
            temp_root=temp_root,
        )
        leave_one_group_out = _leave_one_group_out(
            manifest=manifest,
            manifest_path=manifest_path,
            training_paths=training_paths,
            configs=configs,
            objective_config=objective_config,
            balance_by=balance_by,
            variants=variants,
            temp_root=temp_root,
        )

    baseline_splits = _baseline_splits(manifest_path, split)
    evidence = {
        "schema_version": 2,
        "suite": manifest["suite"],
        "experiment_protocol": manifest.get("experiment_protocol"),
        "method": {
            "optimizer": "deterministic coordinate search",
            "passes_requested": int(passes),
            "global_parameters_only": True,
            "event_specific_fitting": False,
            "objective": {
                **objective_config,
                **(
                    {}
                    if balance_by is None
                    else {"balance_by": balance_by}
                ),
            },
            "parameter_grid": manifest["parameter_grid"],
            "structural_variants": variants,
            "parameter_interpretation": manifest.get(
                "parameter_interpretation", {}
            ),
            "split_policy": manifest["split_policy"],
        },
        "selected_parameters": selected["parameters"],
        "selected_structural_variant": selected[
            "structural_variant"
        ],
        "selected_structural_overrides": selected[
            "structural_overrides"
        ],
        "structural_variant_training_results": [
            {
                "name": item["structural_variant"],
                "overrides": item["structural_overrides"],
                "selected_parameters": item["parameters"],
                "training_objective": item["objective"],
                "training_objective_components": item[
                    "objective_components"
                ],
            }
            for item in variant_selections
        ],
        "parameter_diagnostics": parameter_diagnostics(
            manifest["parameter_grid"], selected["parameters"]
        ),
        "training_objective": selected["objective"],
        "training_objective_components": selected[
            "objective_components"
        ],
        "training_pareto_front": pareto_front,
        "leave_one_training_event_out": leave_one_out,
        "baseline_splits": baseline_splits,
        "splits": evaluated_splits,
        "improvement": _improvement(baseline_splits, evaluated_splits),
        "trace": (
            variant_traces[0]["trace"]
            if len(variant_traces) == 1
            else variant_traces
        ),
    }
    if leave_one_group_out["enabled"]:
        evidence["leave_one_group_out"] = leave_one_group_out
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
                "selected_structural_variant": evidence[
                    "selected_structural_variant"
                ],
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
