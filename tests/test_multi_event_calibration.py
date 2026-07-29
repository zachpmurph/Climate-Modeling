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


def test_tracked_calibration_preserves_holdouts_and_identifiability_warning():
    evidence = json.loads(TRACKED_CALIBRATION.read_text(encoding="utf-8"))

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
