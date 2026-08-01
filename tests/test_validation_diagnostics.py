import json
from pathlib import Path

import numpy as np
import pytest

from rivers.validation.datasets import (
    dataset_record,
    hierarchy_evidence,
    validate_case_policy,
)
from rivers.validation.diagnose import diagnose_hydrograph, diagnose_reach_routing
from rivers.validation.assess_error_sources import assess_error_sources
from rivers.validation.run_case_2d import run_validation_case_2d
from rivers.validation.run_case import run_validation_case


def test_dataset_hierarchy_prefers_observed_and_high_resolution_sources():
    hierarchy = hierarchy_evidence()
    assert [level[0]["rank"] for level in hierarchy] == [1, 2, 3, 4]
    assert hierarchy[0][0]["dataset_id"] == "surveyed_topobathymetry"
    assert {item["dataset_id"] for item in hierarchy[1]} == {
        "usgs_3dep_1m",
        "noaa_topobathy",
    }
    assert dataset_record("gedtm30")["rank"] == 4
    assert dataset_record("gage_datum_reach_proxy")["rank"] == 5


def test_validation_policy_rejects_missing_policy_and_calibration_fields():
    with pytest.raises(ValueError, match="calibration='none'"):
        validate_case_policy({"hydraulic_dataset": "gedtm30"})
    with pytest.raises(ValueError, match="forbidden"):
        validate_case_policy(
            {
                "validation_policy": {"calibration": "none"},
                "hydraulic_dataset": "gedtm30",
                "manning_scale": 0.8,
            }
        )


def test_canonical_validation_refuses_1d_fallback(tmp_path):
    config = tmp_path / "case.json"
    config.write_text(
        json.dumps(
            {
                "validation_policy": {"calibration": "none"},
                "hydraulic_dataset": "assumed_reach_geometry",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="1-D fallback is forbidden"):
        run_validation_case(config)


def test_hydrograph_diagnosis_separates_volume_attenuation_and_timing():
    times = np.arange(0.0, 7.0)
    observed = np.array([0.0, 1.0, 4.0, 7.0, 4.0, 1.0, 0.0])
    predicted = np.array([0.0, 0.0, 0.5, 2.0, 3.0, 2.0, 0.5])
    diagnosis = diagnose_hydrograph(times, observed, predicted)
    components = {item["component"] for item in diagnosis["signals"]}
    assert diagnosis["volume_ratio"] < 0.75
    assert diagnosis["amplitude_ratio"] < 0.75
    assert diagnosis["peak_lag_min"] == pytest.approx(1.0)
    assert {"volume", "attenuation"} <= components


def test_reach_diagnosis_separates_net_volume_and_routing_lag():
    times = np.arange(0.0, 10.0)
    upstream = np.array([0, 0, 1, 3, 5, 3, 1, 0, 0, 0], dtype=float)
    observed = np.roll(upstream, 2) * 0.9
    predicted = np.roll(upstream, 1) * 0.7
    diagnosis = diagnose_reach_routing(
        times, upstream, observed, predicted
    )
    assert diagnosis["observed_net_change_fraction"] == pytest.approx(-0.1)
    assert diagnosis["observed_routing_lag_min"] == pytest.approx(2.0)
    assert diagnosis["modeled_routing_lag_min"] == pytest.approx(1.0)
    assert diagnosis["routing_lag_error_min"] == pytest.approx(-1.0)


def test_every_committed_diagnostic_case_declares_no_calibration():
    manifest_path = (
        "real_world_rivers/validation/diagnostic_suite.json"
    )
    manifest = json.loads(open(manifest_path, encoding="utf-8").read())
    assert manifest["calibration"] == "none"
    for relative in manifest["cases"]:
        config_path = "real_world_rivers/validation/" + relative
        config = json.loads(open(config_path, encoding="utf-8").read())
        dataset = validate_case_policy(config)
        assert dataset["dataset_id"] == config["hydraulic_dataset"]


def test_2d_validation_path_scores_boundary_flux_and_labels_idealized_terrain(
    tmp_path,
):
    observations = tmp_path / "event.csv"
    observations.write_text(
        "role,observed_at,discharge_m3_per_min\n"
        "upstream,2020-01-01T00:00:00Z,20\n"
        "upstream,2020-01-01T00:05:00Z,25\n"
        "upstream,2020-01-01T00:10:00Z,20\n"
        "downstream,2020-01-01T00:00:00Z,20\n"
        "downstream,2020-01-01T00:05:00Z,22\n"
        "downstream,2020-01-01T00:10:00Z,20\n",
        encoding="utf-8",
    )
    config = tmp_path / "case.json"
    config.write_text(
        json.dumps(
            {
                "case": {
                    "name": "short 2-D validation",
                    "river": "synthetic",
                    "observation_window": [
                        "2020-01-01T00:00:00Z",
                        "2020-01-01T00:10:00Z",
                    ],
                },
                "validation_policy": {"calibration": "none"},
                "hydraulic_dataset": "assumed_reach_geometry",
                "validation_2d": {
                    "representation": "ribbon",
                    "x_cells": 5,
                    "y_cells": 2,
                },
                "observations": observations.name,
                "reach": {
                    "length_m": 100.0,
                    "cells": 5,
                    "slope": 0.001,
                    "manning_n": 0.035 / 60.0,
                    "upstream_width_m": 10.0,
                    "downstream_width_m": 10.0,
                },
                "record_interval_min": 5.0,
            }
        ),
        encoding="utf-8",
    )
    evidence = run_validation_case_2d(
        config,
        representation="ribbon",
        x_cells=5,
        y_cells=2,
        output_path=tmp_path / "result.json",
    )
    assert evidence["solver"] == "saint_venant_2d"
    assert evidence["scores"]["n"] == 3
    assert "not mapped flood terrain" in evidence["terrain_representation"]["limitation"]
    assert np.isfinite(evidence["mass"]["outflow_m3"])
    assert abs(evidence["mass"]["relative_balance_residual"]) < 1e-12


def test_committed_2d_error_source_assessment_is_reproducible(tmp_path):
    manifest = "real_world_rivers/validation/error_source_experiments.json"
    actual = assess_error_sources(
        manifest, output_path=tmp_path / "assessment.json"
    )
    expected = json.loads(
        Path(
            "real_world_rivers/validation/error_source_experiments.results.json"
        ).read_text(encoding="utf-8")
    )
    assert actual == expected
    rio = actual["error_sources"]["geometry_and_storage"][
        "rio_proxy_to_measured_sections"
    ]
    assert abs(rio["nse_change"]) < 0.02
    assert actual["error_sources"]["travel_time_attenuation_and_numerics"][
        "cross_region_control"
    ]["truckee_nse"] > 0.95
