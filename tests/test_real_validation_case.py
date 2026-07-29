import json
from pathlib import Path

import pytest

from rivers.validation.run_case import (
    load_two_gauge_observations,
    run_validation_case,
)
from rivers.validation.run_sensitivity import build_variants


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE = REPO_ROOT / "real_world_rivers" / "validation" / "glen_canyon_lees_ferry.json"
OBSERVATIONS = CASE.with_name("glen_canyon_lees_ferry_2004-07-01.csv")
EVIDENCE = CASE.with_suffix(".results.json")
SENSITIVITY = CASE.with_suffix(".sensitivity.json")


def test_committed_two_gauge_observations_are_complete_and_ordered():
    observations = load_two_gauge_observations(OBSERVATIONS)

    upstream_times, upstream_flow = observations["upstream"]
    downstream_times, downstream_flow = observations["downstream"]
    assert len(upstream_times) == 289
    assert len(downstream_times) == 97
    assert upstream_times[0] == downstream_times[0] == 0.0
    assert upstream_times[-1] == downstream_times[-1] == pytest.approx(1440.0)
    assert min(upstream_flow) > 0
    assert min(downstream_flow) > 0


def test_real_case_reproduces_tracked_uncalibrated_baseline(tmp_path):
    tracked = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    actual = run_validation_case(CASE, output_path=tmp_path / "results.json")

    assert actual["status"] == "uncalibrated_baseline"
    assert actual["observations"]["upstream_count"] == 289
    assert actual["observations"]["downstream_count"] == 97
    for metric in ("nse", "rmse", "bias", "percent_bias", "pearson_r"):
        assert actual["scores"][metric] == pytest.approx(
            tracked["scores"][metric], rel=1e-10, abs=1e-10
        )


def test_sensitivity_variants_are_one_at_a_time_and_symmetric():
    config = json.loads(CASE.read_text(encoding="utf-8"))
    variants = {
        name: (overrides, changes)
        for name, overrides, changes in build_variants(config)
    }

    assert len(variants) == 8
    base_n = config["reach"]["manning_n"]
    assert variants["roughness_minus_20_percent"][0]["reach"][
        "manning_n"
    ] == pytest.approx(0.8 * base_n)
    assert variants["roughness_plus_20_percent"][0]["reach"][
        "manning_n"
    ] == pytest.approx(1.2 * base_n)
    assert variants["second_order_reconstruction"][0] == {"spatial_order": 2}


def test_tracked_sensitivity_evidence_covers_every_variant():
    evidence = json.loads(SENSITIVITY.read_text(encoding="utf-8"))
    configured = {
        name
        for name, _, _ in build_variants(
            json.loads(CASE.read_text(encoding="utf-8"))
        )
    }

    assert evidence["method"]["type"] == "one_at_a_time"
    assert {run["name"] for run in evidence["runs"]} == configured
    assert evidence["score_ranges"]["nse"]["minimum"] < 0.0
    assert evidence["score_ranges"]["nse"]["maximum"] > 0.5
