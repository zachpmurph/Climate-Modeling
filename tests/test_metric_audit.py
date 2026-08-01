import csv
import json
from pathlib import Path

from rivers.validation.audit_validation_metrics import audit_validation_metrics


ROOT = Path(__file__).resolve().parents[1] / "real_world_rivers" / "validation"
AUDIT = ROOT / "metric_audit.json"
TRIBUTARY_CONFIG = (
    ROOT
    / "snoqualmie_snoqualmie_carnation_2009-01-07_observed_tributaries.json"
)


def _by_config(audit):
    return {case["config"]: case for case in audit["cases"]}


def test_metric_audit_is_reproducible_and_excludes_rio_grande(tmp_path):
    actual = audit_validation_metrics(
        AUDIT, output_path=tmp_path / "metric_audit.results.json"
    )
    expected = json.loads(
        AUDIT.with_suffix(".results.json").read_text(encoding="utf-8")
    )
    assert actual == expected
    assert actual["excluded_rivers"] == ["Rio Grande"]
    assert all(case["river"] != "Rio Grande" for case in actual["cases"])


def test_metric_audit_exposes_false_positive_headline_scores():
    audit = json.loads(
        AUDIT.with_suffix(".results.json").read_text(encoding="utf-8")
    )
    cases = _by_config(audit)
    assert "high_nse_hides_routing_lag_failure" in cases[
        "truckee_reno_sparks_2017-01-08.json"
    ]["flags"]
    for config in (
        "eel_fort_seward_scotia_2019-02-26.json",
        "willamette_albany_salem_2012-01-18.json",
    ):
        assert "high_correlation_hides_magnitude_or_volume_failure" in cases[
            config
        ]["flags"]
    assert "shape_score_adds_little_value_over_upstream_passthrough" in cases[
        "glen_canyon_lees_ferry_2002-07-02.json"
    ]["flags"]
    for config in (
        "connecticut_montague_holyoke_2011-08-28.json",
        "delaware_montague_belvidere_2006-06-27.json",
        "potomac_point_of_rocks_little_falls_2018-06-02.json",
    ):
        assert cases[config]["flags"] == []
        assert cases[config]["squared_error_skill_over_upstream_passthrough"] > 0.6


def test_observed_tributaries_recover_snoqualmie_volume_without_calibration():
    config = json.loads(TRIBUTARY_CONFIG.read_text(encoding="utf-8"))
    result = json.loads(
        TRIBUTARY_CONFIG.with_suffix(".results.json").read_text(encoding="utf-8")
    )
    baseline = json.loads(
        (ROOT / "snoqualmie_snoqualmie_carnation_2009-01-07.results.json").read_text(
            encoding="utf-8"
        )
    )
    with (ROOT / config["point_flow_series"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert config["validation_policy"] == {"calibration": "none"}
    assert len(config["internal_sources"]) == 2
    assert {row["approval_status"] for row in rows} == {"Approved"}
    assert {row["site_id"] for row in rows} == {
        "USGS-12145500",
        "USGS-12148500",
    }
    assert result["mass"]["lateral_inflow_m3"] > 69_000_000.0
    assert result["mass"]["lateral_inflow_m3"] == result["mass"][
        "lateral_requested_m3"
    ]
    assert abs(result["scores"]["percent_bias"]) < abs(
        baseline["scores"]["percent_bias"]
    )
    assert result["error_diagnosis"]["volume_ratio"] > baseline[
        "error_diagnosis"
    ]["volume_ratio"]
    assert result["reach_diagnosis"]["routing_lag_comparable"] is False
