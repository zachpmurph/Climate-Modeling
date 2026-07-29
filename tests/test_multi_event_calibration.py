import json
from pathlib import Path

import pytest

from rivers.validation import calibrate_suite as calibration


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKED_CALIBRATION = (
    REPO_ROOT
    / "real_world_rivers"
    / "validation"
    / "calibration_suite.results.json"
)


def test_composite_objective_rewards_skill_and_penalizes_event_spread():
    consistent = [
        {"nse": 0.5, "pearson_r": 0.8},
        {"nse": 0.5, "pearson_r": 0.8},
    ]
    variable = [
        {"nse": 0.2, "pearson_r": 0.8},
        {"nse": 0.8, "pearson_r": 0.8},
    ]

    assert calibration.composite_objective(
        consistent
    ) > calibration.composite_objective(variable)


def test_objective_reports_worst_event_and_correlation_penalties():
    scores = [
        {"nse": 0.2, "pearson_r": 0.4},
        {"nse": 0.8, "pearson_r": 0.9},
    ]
    components = calibration.objective_components(
        scores,
        robustness_penalty=0.1,
        correlation_robustness_penalty=0.2,
        worst_event_weight=0.3,
    )

    assert components["minimum_nse"] == pytest.approx(0.2)
    assert components["worst_event_nse_gap"] == pytest.approx(0.3)
    assert components["nse_spread_penalty"] == pytest.approx(0.03)
    assert components["correlation_spread_penalty"] == pytest.approx(
        0.05
    )
    assert components["worst_event_penalty"] == pytest.approx(0.09)
    assert calibration.composite_objective(
        scores,
        robustness_penalty=0.1,
        correlation_robustness_penalty=0.2,
        worst_event_weight=0.3,
    ) == pytest.approx(components["objective"])


def test_group_balanced_objective_gives_each_river_equal_weight():
    scores = [
        {"nse": 1.0, "pearson_r": 0.8},
        {"nse": 1.0, "pearson_r": 0.8},
        {"nse": -1.0, "pearson_r": 0.4},
    ]

    components = calibration.objective_components(
        scores,
        group_labels=["River A", "River A", "River B"],
        nse_weight=1.0,
        correlation_weight=0.0,
        robustness_penalty=0.0,
    )

    assert components["mean_nse"] == pytest.approx(0.0)
    assert components["aggregation"] == "equal_weight_per_group"
    assert components["group_count"] == 2
    assert components["group_summary"][0] == {
        "group": "River A",
        "event_count": 2,
        "mean_nse": 1.0,
        "mean_correlation": 0.8,
    }


def test_parameter_overrides_apply_one_global_physical_scaling():
    config = {
        "reach": {
            "manning_n": 0.001,
            "slope": 0.002,
            "upstream_width_m": 20.0,
            "downstream_width_m": 30.0,
        }
    }
    parameters = {
        "manning_scale": 0.8,
        "width_scale": 1.2,
        "slope_scale": 1.1,
        "lateral_inflow_fraction": 0.05,
    }

    overrides = calibration.parameter_overrides(config, parameters)

    assert overrides["reach"]["manning_n"] == pytest.approx(0.0008)
    assert overrides["reach"]["slope"] == pytest.approx(0.0022)
    assert overrides["reach"]["upstream_width_m"] == pytest.approx(24.0)
    assert overrides["reach"]["downstream_width_m"] == pytest.approx(36.0)
    assert overrides["lateral_inflow_fraction"] == pytest.approx(0.05)


def test_calibration_does_not_scale_reviewed_stage_geometry():
    config = {
        "reach": {
            "manning_n": 0.001,
            "slope": 0.002,
            "cross_section_shape": "surveyed",
            "surveyed_cross_sections": "sections.csv",
        }
    }
    parameters = {
        "manning_scale": 1.0,
        "width_scale": 1.2,
        "slope_scale": 1.0,
        "lateral_inflow_fraction": 0.0,
    }

    with pytest.raises(ValueError, match="reviewed stage-dependent geometry"):
        calibration.parameter_overrides(config, parameters)


def test_calibration_does_not_double_count_measured_point_flows():
    config = {
        "point_flow_series": "measured.csv",
        "reach": {
            "manning_n": 0.001,
            "slope": 0.002,
            "upstream_width_m": 20.0,
            "downstream_width_m": 20.0,
        },
    }
    parameters = {
        "manning_scale": 1.0,
        "width_scale": 1.0,
        "slope_scale": 1.0,
        "lateral_inflow_fraction": 0.1,
    }

    with pytest.raises(ValueError, match="measured point_flow_series"):
        calibration.parameter_overrides(config, parameters)


def test_coordinate_search_selects_joint_candidate_and_records_trace():
    def evaluate(parameters):
        objective = -(
            (parameters["width_scale"] - 1.2) ** 2
            + (parameters["lateral_inflow_fraction"] - 0.1) ** 2
        )
        return {
            "parameters": dict(parameters),
            "objective": objective,
            "training_events": [],
        }

    selected, trace = calibration.coordinate_search(
        evaluate,
        {
            "width_scale": [0.8, 1.0, 1.2],
            "lateral_inflow_fraction": [0.0, 0.1],
        },
        passes=2,
    )

    assert selected["parameters"]["width_scale"] == pytest.approx(1.2)
    assert selected["parameters"]["lateral_inflow_fraction"] == pytest.approx(0.1)
    assert trace[0]["step"] == "initial"
    assert {entry.get("parameter") for entry in trace[1:]} == {
        "width_scale",
        "lateral_inflow_fraction",
    }


def test_parameter_diagnostics_flag_unbracketed_optimum():
    diagnostics = calibration.parameter_diagnostics(
        {
            "manning_scale": [0.8, 1.0, 1.2],
            "width_scale": [0.8, 1.0, 1.2],
        },
        {"manning_scale": 0.8, "width_scale": 1.0},
    )

    assert diagnostics["identifiability_warning"] is True
    assert diagnostics["boundary_hits"] == {"manning_scale": ["minimum"]}


def test_coordinate_search_does_not_move_on_an_objective_tie():
    def evaluate(parameters):
        return {
            "parameters": dict(parameters),
            "objective": 1.0,
            "training_events": [],
        }

    selected, _ = calibration.coordinate_search(
        evaluate,
        {"width_scale": [0.8, 1.0, 1.2]},
        passes=1,
    )

    assert selected["parameters"]["width_scale"] == pytest.approx(1.0)


def test_calibration_never_uses_validation_or_test_for_selection(
    tmp_path, monkeypatch
):
    reach = {
        "manning_n": 0.001,
        "slope": 0.002,
        "upstream_width_m": 20.0,
        "downstream_width_m": 20.0,
    }
    for name in ("train-a.json", "train-b.json", "validation.json", "test.json"):
        config_path = tmp_path / name
        config_path.write_text(
            json.dumps({"reach": reach}), encoding="utf-8"
        )
        config_path.with_suffix(".results.json").write_text(
            json.dumps(
                {
                    "scores": {
                        "nse": 0.0,
                        "rmse": 1.0,
                        "bias": 0.0,
                        "percent_bias": 0.0,
                        "pearson_r": 0.5,
                        "n": 2,
                    }
                }
            ),
            encoding="utf-8",
        )
    manifest = {
        "suite": {"name": "test"},
        "split_policy": "test split",
        "split": {
            "training": ["train-a.json", "train-b.json"],
            "validation": ["validation.json"],
            "test": ["test.json"],
        },
        "objective": {
            "nse_weight": 0.7,
            "correlation_weight": 0.3,
            "robustness_penalty": 0.1,
        },
        "parameter_grid": {
            "manning_scale": [1.0],
            "width_scale": [1.0],
            "slope_scale": [1.0],
            "lateral_inflow_fraction": [0.0, 0.1],
        },
    }
    manifest_path = tmp_path / "calibration.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    calls = []

    def fake_run(config_path, *, output_path, overrides):
        calls.append(config_path.name)
        fraction = overrides["lateral_inflow_fraction"]
        return {
            "scores": {
                "nse": fraction,
                "rmse": 1.0,
                "bias": 0.0,
                "percent_bias": 0.0,
                "pearson_r": 0.5,
                "n": 2,
            }
        }

    monkeypatch.setattr(calibration, "run_validation_case", fake_run)
    evidence = calibration.calibrate_suite(
        manifest_path,
        output_path=tmp_path / "result.json",
        passes=1,
    )

    assert evidence["selected_parameters"]["lateral_inflow_fraction"] == 0.1
    assert calls.count("validation.json") == 1
    assert calls.count("test.json") == 1
    assert set(calls[:-2]) == {"train-a.json", "train-b.json"}


def test_leave_one_event_out_refits_without_held_event(
    tmp_path, monkeypatch
):
    reach = {
        "manning_n": 0.001,
        "slope": 0.002,
        "upstream_width_m": 20.0,
        "downstream_width_m": 20.0,
    }
    for name in ("train-a.json", "train-b.json"):
        path = tmp_path / name
        path.write_text(json.dumps({"reach": reach}), encoding="utf-8")
        path.with_suffix(".results.json").write_text(
            json.dumps(
                {
                    "scores": {
                        "nse": 0.0,
                        "rmse": 1.0,
                        "bias": 0.0,
                        "percent_bias": 0.0,
                        "pearson_r": 0.5,
                        "n": 2,
                    }
                }
            ),
            encoding="utf-8",
        )
    manifest = {
        "suite": {"name": "loeo test"},
        "split_policy": "training only",
        "split": {
            "training": ["train-a.json", "train-b.json"],
            "validation": [],
            "test": [],
        },
        "objective": {
            "nse_weight": 0.7,
            "correlation_weight": 0.3,
            "robustness_penalty": 0.1,
            "worst_event_weight": 0.2,
        },
        "cross_event_validation": {"enabled": True, "passes": 1},
        "parameter_grid": {
            "manning_scale": [1.0],
            "width_scale": [1.0],
            "slope_scale": [1.0],
            "lateral_inflow_fraction": [0.0, 0.1],
        },
    }
    manifest_path = tmp_path / "calibration.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def fake_run(config_path, *, output_path, overrides):
        fraction = overrides["lateral_inflow_fraction"]
        nse = (
            1.0 - fraction
            if config_path.name == "train-a.json"
            else 10.0 * fraction
        )
        return {
            "scores": {
                "nse": nse,
                "rmse": 1.0,
                "bias": 0.0,
                "percent_bias": 0.0,
                "pearson_r": 0.8,
                "n": 2,
            }
        }

    monkeypatch.setattr(calibration, "run_validation_case", fake_run)
    evidence = calibration.calibrate_suite(
        manifest_path,
        output_path=tmp_path / "result.json",
        passes=1,
    )

    diagnostics = evidence["leave_one_training_event_out"]
    assert diagnostics["enabled"] is True
    assert diagnostics["fold_count"] == 2
    folds = {
        fold["held_out_event"]: fold for fold in diagnostics["folds"]
    }
    assert folds["train-a.json"]["fit_events"] == ["train-b.json"]
    assert folds["train-a.json"]["selected_parameters"][
        "lateral_inflow_fraction"
    ] == pytest.approx(0.1)
    assert folds["train-b.json"]["fit_events"] == ["train-a.json"]
    assert folds["train-b.json"]["selected_parameters"][
        "lateral_inflow_fraction"
    ] == pytest.approx(0.0)


def test_leave_one_group_out_refits_without_any_held_river(
    tmp_path, monkeypatch
):
    reach = {
        "manning_n": 0.001,
        "slope": 0.002,
        "upstream_width_m": 20.0,
        "downstream_width_m": 20.0,
    }
    rivers = {
        "a-1.json": "River A",
        "a-2.json": "River A",
        "b-1.json": "River B",
        "b-2.json": "River B",
    }
    for name, river in rivers.items():
        path = tmp_path / name
        path.write_text(
            json.dumps({"case": {"river": river}, "reach": reach}),
            encoding="utf-8",
        )
        path.with_suffix(".results.json").write_text(
            json.dumps(
                {
                    "scores": {
                        "nse": 0.0,
                        "rmse": 1.0,
                        "bias": 0.0,
                        "percent_bias": 0.0,
                        "pearson_r": 0.5,
                        "n": 2,
                    }
                }
            ),
            encoding="utf-8",
        )
    manifest = {
        "suite": {"name": "river transfer"},
        "split_policy": "training only",
        "split": {
            "training": list(rivers),
            "validation": [],
            "test": [],
        },
        "objective": {
            "nse_weight": 1.0,
            "correlation_weight": 0.0,
            "robustness_penalty": 0.0,
            "balance_by": "river",
        },
        "cross_group_validation": {"enabled": True, "passes": 1},
        "parameter_grid": {
            "manning_scale": [1.0],
            "width_scale": [1.0],
            "slope_scale": [1.0],
            "lateral_inflow_fraction": [0.0, 0.1],
        },
    }
    manifest_path = tmp_path / "calibration.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def fake_run(config_path, *, output_path, overrides):
        fraction = overrides["lateral_inflow_fraction"]
        river = rivers[config_path.name]
        nse = 1.0 - fraction if river == "River A" else 10.0 * fraction
        return {
            "scores": {
                "nse": nse,
                "rmse": 1.0,
                "bias": 0.0,
                "percent_bias": 0.0,
                "pearson_r": 0.5,
                "n": 2,
            }
        }

    monkeypatch.setattr(calibration, "run_validation_case", fake_run)
    evidence = calibration.calibrate_suite(
        manifest_path,
        output_path=tmp_path / "result.json",
        passes=1,
    )

    diagnostics = evidence["leave_one_group_out"]
    assert diagnostics["balance_by"] == "river"
    assert diagnostics["fold_count"] == 2
    folds = {fold["held_out_group"]: fold for fold in diagnostics["folds"]}
    assert folds["River A"]["fit_events"] == ["b-1.json", "b-2.json"]
    assert folds["River A"]["selected_parameters"][
        "lateral_inflow_fraction"
    ] == pytest.approx(0.1)
    assert folds["River B"]["fit_events"] == ["a-1.json", "a-2.json"]
    assert folds["River B"]["selected_parameters"][
        "lateral_inflow_fraction"
    ] == pytest.approx(0.0)


def test_tracked_calibration_preserves_holdouts_and_identifiability_warning():
    evidence = json.loads(TRACKED_CALIBRATION.read_text(encoding="utf-8"))

    assert evidence["schema_version"] == 2
    assert evidence["method"]["global_parameters_only"] is True
    assert evidence["method"]["event_specific_fitting"] is False
    assert evidence["parameter_diagnostics"]["identifiability_warning"] is True
    assert set(evidence["parameter_diagnostics"]["boundary_hits"]) == {
        "manning_scale",
        "width_scale",
        "slope_scale",
    }
    assert evidence["splits"]["validation"]["summary"]["nse"]["mean"] > 0.7
    assert evidence["splits"]["test"]["summary"]["pearson_r"]["mean"] > 0.8
    transfer = evidence["leave_one_training_event_out"]
    assert transfer["enabled"] is True
    assert transfer["fold_count"] == 2
    assert transfer["held_out_summary"]["nse"]["minimum"] > 0.7
    assert {
        tuple(sorted(fold["selected_parameters"].items()))
        for fold in transfer["folds"]
    } == {tuple(sorted(evidence["selected_parameters"].items()))}
