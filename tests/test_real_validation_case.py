import csv
import json
from datetime import datetime
from pathlib import Path

import pytest

from rivers.validation.run_case import (
    discharge_boundary,
    load_two_gauge_observations,
    run_validation_case,
    shifted_boundary,
)
from rivers.validation.fetch_event import collect_event_rows
from rivers.validation.run_sensitivity import build_variants
from rivers.validation.run_suite import summarize_results


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE = REPO_ROOT / "real_world_rivers" / "validation" / "glen_canyon_lees_ferry.json"
OBSERVATIONS = CASE.with_name("glen_canyon_lees_ferry_2004-07-01.csv")
EVIDENCE = CASE.with_suffix(".results.json")
SENSITIVITY = CASE.with_suffix(".sensitivity.json")
SUITE = CASE.with_name("validation_suite.json")
SUITE_EVIDENCE = SUITE.with_suffix(".results.json")


def test_committed_two_gauge_observations_are_complete_and_ordered():
    observations = load_two_gauge_observations(OBSERVATIONS)

    upstream_times, upstream_flow = observations["upstream"]
    downstream_times, downstream_flow = observations["downstream"]
    assert len(upstream_times) == 433
    assert len(downstream_times) == 97
    assert upstream_times[0] == pytest.approx(-720.0)
    assert downstream_times[0] == 0.0
    assert upstream_times[-1] == downstream_times[-1] == pytest.approx(1440.0)
    assert min(upstream_flow) > 0
    assert min(downstream_flow) > 0


def test_real_case_reproduces_tracked_uncalibrated_baseline(tmp_path):
    tracked = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    actual = run_validation_case(CASE, output_path=tmp_path / "results.json")

    assert actual["status"] == "uncalibrated_baseline"
    assert actual["observations"]["upstream_count"] == 433
    assert actual["observations"]["downstream_count"] == 97
    assert actual["assumptions"]["warmup_upstream_forcing"] == "observed"
    assert actual["mass"]["lateral_inflow_m3"] == pytest.approx(0.0)
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


def test_multi_event_suite_observations_and_results_are_reproducible(tmp_path):
    manifest = json.loads(SUITE.read_text(encoding="utf-8"))
    tracked_suite = json.loads(SUITE_EVIDENCE.read_text(encoding="utf-8"))

    assert len(manifest["cases"]) == tracked_suite["case_count"] == 7
    for relative_path in manifest["cases"]:
        config_path = SUITE.parent / relative_path
        config = json.loads(config_path.read_text(encoding="utf-8"))
        observation_path = config_path.parent / config["observations"]
        observations = load_two_gauge_observations(observation_path)
        with observation_path.open(newline="", encoding="utf-8") as handle:
            raw_rows = list(csv.DictReader(handle))
        assert len(observations["upstream"][0]) >= 145
        assert len(observations["downstream"][0]) >= 97
        assert {row["approval_status"] for row in raw_rows} == {"Approved"}
        start, end = config["case"]["observation_window"]
        duration_min = (
            datetime.fromisoformat(end.replace("Z", "+00:00"))
            - datetime.fromisoformat(start.replace("Z", "+00:00"))
        ).total_seconds() / 60.0
        assert observations["upstream"][0][-1] == pytest.approx(duration_min)
        assert observations["downstream"][0][-1] == pytest.approx(duration_min)

        tracked = json.loads(
            config_path.with_suffix(".results.json").read_text(encoding="utf-8")
        )
        actual = run_validation_case(
            config_path,
            output_path=tmp_path / f"{config_path.stem}.results.json",
        )
        assert actual["status"] == tracked["status"]
        for metric in ("nse", "rmse", "bias", "percent_bias", "pearson_r"):
            assert actual["scores"][metric] == pytest.approx(
                tracked["scores"][metric], rel=1e-10, abs=1e-10
            )

    independent = [
        run
        for run in tracked_suite["cases"]
        if run["config"].startswith("truckee_")
    ]
    assert len(independent) == 2
    assert min(run["scores"]["nse"] for run in independent) > 0.75
    assert min(run["scores"]["pearson_r"] for run in independent) > 0.9

    rio_grande = [
        run
        for run in tracked_suite["cases"]
        if run["config"].startswith("rio_grande_")
    ]
    assert len(rio_grande) == 1
    assert rio_grande[0]["status"] == "predeclared_third_river_transfer_test"


def test_third_river_event_retains_predeclared_first_run_protocol():
    config_path = SUITE.parent / "rio_grande_alameda_albuquerque_2023-05-12.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = config["case"]["selection_protocol"]

    assert config["retrieved_at"] == "2026-07-29"
    assert protocol["declared_before_observation_fetch"] is True
    assert "first completed run" in protocol["first_run_policy"]
    assert "No Rio Grande observation" in protocol["calibration_policy"]


def test_validation_case_uses_surveyed_stage_dependent_geometry(tmp_path):
    observations = tmp_path / "observations.csv"
    observations.write_text(
        "role,observed_at,discharge_m3_per_min\n"
        "upstream,2020-01-01T00:00:00Z,100\n"
        "upstream,2020-01-01T00:15:00Z,100\n"
        "downstream,2020-01-01T00:00:00Z,100\n"
        "downstream,2020-01-01T00:15:00Z,100\n",
        encoding="utf-8",
    )
    surveys = tmp_path / "sections.csv"
    surveys.write_text(
        "station_m,offset_m,elevation_m\n"
        "0,0,2\n0,2,0\n0,8,0\n0,10,2\n"
        "100,0,3\n100,3,0\n100,9,0\n100,12,3\n",
        encoding="utf-8",
    )
    config = tmp_path / "case.json"
    config.write_text(
        json.dumps(
            {
                "case": {
                    "name": "surveyed validation",
                    "observation_window": [
                        "2020-01-01T00:00:00Z",
                        "2020-01-01T00:15:00Z",
                    ],
                },
                "observations": observations.name,
                "reach": {
                    "length_m": 100.0,
                    "cells": 3,
                    "slope": 0.001,
                    "manning_n": 0.035 / 60.0,
                    "cross_section_shape": "surveyed",
                    "surveyed_cross_sections": surveys.name,
                },
                "record_interval_min": 5.0,
            }
        ),
        encoding="utf-8",
    )

    evidence = run_validation_case(
        config, output_path=tmp_path / "results.json"
    )

    assert evidence["assumptions"]["cross_section_shape"] == "surveyed"
    assert evidence["assumptions"]["surveyed_cross_sections"] == surveys.name
    assert (
        evidence["assumptions"]["initial_condition"]
        == "per-cell cross-section Manning normal depth and discharge"
    )
    assert evidence["scores"]["n"] == 2


def test_observed_warmup_boundary_maps_negative_event_time_to_spinup_clock():
    boundary = discharge_boundary([-10.0, 0.0], [100.0, 200.0])
    warmup = shifted_boundary(boundary, -10.0)

    assert warmup(0.0) == pytest.approx(100.0)
    assert warmup(10.0) == pytest.approx(200.0)
    assert warmup.breakpoints_min == pytest.approx((0.0, 10.0))


def test_fetch_event_normalizes_approved_usgs_rows():
    config = {
        "case": {
            "observation_window": [
                "2020-01-01T00:00:00Z",
                "2020-01-01T00:15:00Z",
            ]
        },
        "sources": [
            {"role": "upstream", "gauge": "USGS-1"},
            {"role": "downstream", "gauge": "USGS-2"},
        ],
    }

    def requester(url, params=None):
        site = params["monitoring_location_id"]
        values = (100.0, 110.0) if site == "USGS-1" else (90.0, 105.0)
        features = [
            {
                "properties": {
                    "parameter_code": "00060",
                    "time": timestamp,
                    "value": value,
                    "unit_of_measure": "ft^3/s",
                    "approval_status": "Approved",
                }
            }
            for timestamp, value in zip(
                ("2020-01-01T00:00:00Z", "2020-01-01T00:15:00Z"),
                values,
            )
        ]
        return {"features": features}, f"https://example.test/{site}"

    rows, urls = collect_event_rows(config, requester=requester)

    assert len(rows) == 4
    assert [row["role"] for row in rows] == [
        "upstream",
        "upstream",
        "downstream",
        "downstream",
    ]
    assert rows[0]["observed_at"] == "2020-01-01T00:00:00+00:00"
    assert float(rows[0]["discharge_m3_per_min"]) == pytest.approx(
        100.0 * 0.028316846592 * 60.0
    )
    assert set(urls) == {"upstream", "downstream"}


def test_suite_summary_uses_event_ranges_and_medians():
    summary = summarize_results(
        [
            {"scores": {key: value for key in ("nse", "rmse", "bias", "percent_bias", "pearson_r")}}
            for value in (1.0, 3.0, 2.0)
        ]
    )

    assert summary["nse"] == {"minimum": 1.0, "median": 2.0, "maximum": 3.0}
