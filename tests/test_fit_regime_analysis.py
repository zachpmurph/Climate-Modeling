import csv
import json
from pathlib import Path

from rivers.validation.analyze_fit_regimes import analyze_fit_regimes
from rivers.validation.datasets import validate_case_policy


ROOT = Path(__file__).resolve().parents[1] / "real_world_rivers" / "validation"
HOLDOUT_MANIFEST = ROOT / "cross_river_hypothesis_suite.json"
ANALYSIS_MANIFEST = ROOT / "fit_regime_analysis.json"


def test_predeclared_holdouts_use_approved_2d_evidence_without_calibration():
    manifest = json.loads(HOLDOUT_MANIFEST.read_text(encoding="utf-8"))
    assert len(manifest["cases"]) == 3
    for relative in manifest["cases"]:
        config = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        dataset = validate_case_policy(config)
        result = json.loads(
            (ROOT / relative).with_suffix(".results.json").read_text(
                encoding="utf-8"
            )
        )
        with (ROOT / config["observations"]).open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        assert dataset["dataset_id"] == "gage_datum_reach_proxy"
        assert config["validation_policy"] == {"calibration": "none"}
        assert config["validation_2d"] == {
            "representation": "ribbon",
            "x_cells": 31,
            "y_cells": 1,
        }
        assert result["solver"] == "saint_venant_2d"
        assert {row["approval_status"] for row in rows} == {"Approved"}


def test_fit_regime_analysis_is_reproducible_and_excludes_rio_grande(tmp_path):
    actual = analyze_fit_regimes(
        ANALYSIS_MANIFEST, output_path=tmp_path / "fit_regimes.json"
    )
    tracked = json.loads(
        ANALYSIS_MANIFEST.with_suffix(".results.json").read_text(encoding="utf-8")
    )
    assert actual == tracked
    assert actual["excluded_rivers"] == ["Rio Grande"]
    assert all(case["river"] != "Rio Grande" for case in actual["cases"])
    assert actual["matched_holdout_test"]["confirmed"] is True
    associations = actual["natural_flow_associations"]
    assert associations[
        "drainage_growth_vs_observed_volume_gain_pearson_r"
    ] > 0.9
    assert associations[
        "drainage_growth_vs_modeled_volume_deficit_pearson_r"
    ] > 0.75
    assert associations["drainage_growth_vs_nse_pearson_r"] < -0.9


def test_colorado_diagnostics_rule_out_window_and_grid_as_primary_causes():
    analysis = json.loads(
        ANALYSIS_MANIFEST.with_suffix(".results.json").read_text(encoding="utf-8")
    )
    diagnostics = analysis["diagnostics"]
    assert abs(
        diagnostics["colorado_fine"]["volume_ratio"]
        - diagnostics["colorado_coarse"]["volume_ratio"]
    ) < 0.002
    assert abs(
        diagnostics["colorado_three_day"]["volume_ratio"]
        - diagnostics["colorado_one_day"]["volume_ratio"]
    ) < 0.02
    assert abs(diagnostics["colorado_shelf"]["routing_lag_error_min"]) < abs(
        diagnostics["colorado_coarse"]["routing_lag_error_min"]
    )
    assert diagnostics["colorado_shelf"]["volume_ratio"] < diagnostics[
        "colorado_coarse"
    ]["volume_ratio"]
