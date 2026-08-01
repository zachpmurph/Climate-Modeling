import csv
import json
from datetime import datetime
from pathlib import Path

import pytest

from rivers.validation.analyze_extended_windows import analyze_extended_windows
from rivers.validation.extend_event_windows import build_extended_config


ROOT = Path(__file__).resolve().parents[1] / "real_world_rivers" / "validation"
STUDY = ROOT / "extended_window_study.json"


def test_extended_config_preserves_hydraulics_and_adds_fixed_time():
    base = {
        "case": {
            "name": "test event",
            "observation_window": [
                "2020-01-01T00:00:00Z",
                "2020-01-02T00:00:00Z",
            ],
        },
        "validation_policy": {"calibration": "none"},
        "observations": "base.csv",
        "sources": [
            {"role": "upstream", "gauge": "USGS-1"},
            {"role": "downstream", "gauge": "USGS-2"},
        ],
        "warmup": {"duration_min": 720.0},
        "reach": {"length_m": 1000.0, "manning_n": 0.035 / 60.0},
    }
    extended = build_extended_config(
        base, extension_hours=48.0, output_stem="test_extended_48h"
    )
    assert extended["case"]["observation_window"] == [
        "2020-01-01T00:00:00Z",
        "2020-01-04T00:00:00Z",
    ]
    assert extended["reach"] == base["reach"]
    assert extended["validation_policy"] == {"calibration": "none"}
    assert extended["observations"] == "test_extended_48h.csv"
    assert extended["observation_endpoint_tolerance_min"] == 15.0


def test_every_extended_case_has_approved_evidence_and_conserves_mass():
    study = json.loads(STUDY.read_text(encoding="utf-8"))
    extension = float(study["extension_hours"])
    for relative in study["base_cases"]:
        base_path = ROOT / relative
        base = json.loads(base_path.read_text(encoding="utf-8"))
        extended_path = ROOT / f"{base_path.stem}_extended_{extension:g}h.json"
        extended = json.loads(extended_path.read_text(encoding="utf-8"))
        result = json.loads(
            extended_path.with_suffix(".results.json").read_text(encoding="utf-8")
        )
        with (ROOT / extended["observations"]).open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        start = datetime.fromisoformat(
            base["case"]["observation_window"][1].replace("Z", "+00:00")
        )
        end = datetime.fromisoformat(
            extended["case"]["observation_window"][1].replace("Z", "+00:00")
        )
        assert (end - start).total_seconds() == pytest.approx(extension * 3600.0)
        assert extended["validation_policy"] == {"calibration": "none"}
        assert extended["reach"] == base["reach"]
        assert result["solver"] == "saint_venant_2d"
        assert {row["approval_status"] for row in rows} == {"Approved"}
        assert abs(result["mass"]["relative_balance_residual"]) < 1e-9


def test_extended_window_analysis_is_reproducible_and_finds_prevalent_early_routing(
    tmp_path,
):
    actual = analyze_extended_windows(
        STUDY, output_path=tmp_path / "extended.results.json"
    )
    expected = json.loads(
        STUDY.with_suffix(".results.json").read_text(encoding="utf-8")
    )
    assert actual == expected
    summary = actual["summary"]
    assert actual["case_count"] == 16
    assert summary["early_routing_prevalent"] is True
    assert summary["extended_routing_early_case_count"] == 13
    assert summary["extended_routing_materially_early_case_count"] == 10
    assert summary["all_cases_numerically_stable_and_conservative"] is True
    assert summary["median_nse"]["extended"] > summary["median_nse"]["baseline"]

    cases = {case["base_config"]: case for case in actual["cases"]}
    january = cases["truckee_reno_sparks_2017-01-08.json"]
    february = cases["truckee_reno_sparks_2017-02-10.json"]
    assert january["routing"]["baseline_lag_error_min"] == pytest.approx(-375.0)
    assert january["routing"]["extended_lag_error_min"] == pytest.approx(-60.0)
    assert february["routing"]["extended_lag_error_min"] == pytest.approx(-45.0)
