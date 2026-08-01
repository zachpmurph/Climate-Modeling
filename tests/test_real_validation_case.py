import csv
import csv
import json
from datetime import datetime
from pathlib import Path

import pytest

from rivers.validation.run_case import (
    _require_control_coverage,
    discharge_boundary,
    load_event_control_series,
    load_field_measurement_geometry,
    load_two_gauge_observations,
    run_validation_case,
    shifted_boundary,
)
from rivers.validation.fetch_event import collect_event_rows
from rivers.validation.fetch_point_flows import collect_point_flow_rows
from rivers.validation.fetch_channel_geometry import (
    collect_channel_geometry_rows,
)
from rivers.validation.fetch_stage_control import collect_stage_rows
from rivers.validation.run_sensitivity import build_variants
from rivers.validation.run_suite import summarize_results


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE = REPO_ROOT / "real_world_rivers" / "validation" / "glen_canyon_lees_ferry.json"
OBSERVATIONS = CASE.with_name("glen_canyon_lees_ferry_2004-07-01.csv")
EVIDENCE = CASE.with_suffix(".results.json")
SENSITIVITY = CASE.with_suffix(".sensitivity.json")
SUITE = CASE.with_name("validation_suite.json")
SUITE_EVIDENCE = SUITE.with_suffix(".results.json")
RIO_STAGE_CASE = CASE.with_name(
    "rio_grande_alameda_albuquerque_2023-05-12_stage.json"
)
RIO_STAGE_EVIDENCE = RIO_STAGE_CASE.with_suffix(".results.json")
RIO_STAGE_SERIES = RIO_STAGE_CASE.with_suffix(".csv")
RIO_GEOMETRY_CASE = CASE.with_name(
    "rio_grande_alameda_albuquerque_2023-05-12_stage_geometry.json"
)
RIO_GEOMETRY_EVIDENCE = RIO_GEOMETRY_CASE.with_suffix(".results.json")
RIO_GEOMETRY_SERIES = RIO_GEOMETRY_CASE.with_suffix(".csv")
RIO_CHANNEL_GEOMETRY = CASE.with_name(
    "rio_grande_alameda_albuquerque_2023-05-12_channel_geometry.csv"
)


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

    assert actual["status"] == "uncalibrated_2d_screening"
    assert actual["solver"] == "saint_venant_2d"
    assert actual["solver_policy"] == "saint_venant_2d_only"
    assert actual["observations"]["upstream_count"] == 433
    assert actual["observations"]["downstream_count"] == 97
    assert actual["terrain_representation"]["name"] == "ribbon"
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

    assert len(variants) == 3
    assert variants["baseline_2d_ribbon"][0]["representation"] == "ribbon"
    assert variants["fine_longitudinal_2d_ribbon"][0]["x_cells"] == 61
    assert variants["connected_lateral_shelves_2d"][0]["representation"] == "shelf"


def test_tracked_sensitivity_evidence_covers_every_variant():
    evidence = json.loads(SENSITIVITY.read_text(encoding="utf-8"))
    configured = {
        name
        for name, _, _ in build_variants(
            json.loads(CASE.read_text(encoding="utf-8"))
        )
    }

    assert evidence["method"]["type"] == "one_at_a_time_structural_2d"
    assert evidence["solver_policy"] == "saint_venant_2d_only"
    assert {run["name"] for run in evidence["runs"]} == configured
    assert evidence["score_ranges"]["nse"]["minimum"] < 0.0
    assert evidence["score_ranges"]["nse"]["maximum"] > 0.4


def test_multi_event_suite_observations_and_results_are_reproducible(tmp_path):
    manifest = json.loads(SUITE.read_text(encoding="utf-8"))
    tracked_suite = json.loads(SUITE_EVIDENCE.read_text(encoding="utf-8"))

    assert len(manifest["cases"]) == tracked_suite["case_count"] == 12
    for relative_path in manifest["cases"]:
        config_path = SUITE.parent / relative_path
        config = json.loads(config_path.read_text(encoding="utf-8"))
        observation_path = config_path.parent / config["observations"]
        observations = load_two_gauge_observations(observation_path)
        with observation_path.open(newline="", encoding="utf-8") as handle:
            raw_rows = list(csv.DictReader(handle))
        assert len(observations["upstream"][0]) >= 145
        assert len(observations["downstream"][0]) >= 90
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


def test_measured_stage_experiment_reproduces_without_replacing_baseline(
    tmp_path,
):
    tracked = json.loads(RIO_STAGE_EVIDENCE.read_text(encoding="utf-8"))
    baseline = json.loads(
        RIO_STAGE_CASE.with_name(
            "rio_grande_alameda_albuquerque_2023-05-12.results.json"
        ).read_text(encoding="utf-8")
    )
    actual = run_validation_case(
        RIO_STAGE_CASE,
        output_path=tmp_path / "rio-stage-results.json",
    )
    with RIO_STAGE_SERIES.open(newline="", encoding="utf-8") as handle:
        stage_rows = list(csv.DictReader(handle))

    assert len(stage_rows) == 241
    assert {row["approval_status"] for row in stage_rows} == {"Approved"}
    assert stage_rows[0]["observed_at"] == "2023-05-11T12:00:00+00:00"
    assert stage_rows[-1]["observed_at"] == "2023-05-14T00:00:00+00:00"
    assert actual["status"] == "post_baseline_measured_stage_experiment"
    assert actual["solver"] == "saint_venant_2d"
    assert actual["boundary"]["x"] == "inflow_stage"
    for metric in ("nse", "rmse", "bias", "percent_bias", "pearson_r"):
        assert actual["scores"][metric] == pytest.approx(
            tracked["scores"][metric], rel=1e-10, abs=1e-10
        )
    assert actual["boundary"]["downstream_score_observable"] == (
        "finite-volume 2-D downstream boundary discharge flux"
    )
    assert actual["scores"]["nse"] < baseline["scores"]["nse"]
    assert actual["scores"]["rmse"] > baseline["scores"]["rmse"]


def test_measured_field_geometry_experiment_is_reproducible(tmp_path):
    tracked = json.loads(
        RIO_GEOMETRY_EVIDENCE.read_text(encoding="utf-8")
    )
    actual = run_validation_case(
        RIO_GEOMETRY_CASE,
        output_path=tmp_path / "rio-stage-geometry-results.json",
    )
    with RIO_GEOMETRY_SERIES.open(newline="", encoding="utf-8") as handle:
        stage_rows = list(csv.DictReader(handle))
    with RIO_CHANNEL_GEOMETRY.open(
        newline="", encoding="utf-8"
    ) as handle:
        geometry_rows = list(csv.DictReader(handle))

    assert len(stage_rows) == 241
    assert len(geometry_rows) == 2
    assert [float(row["active_width_m"]) for row in geometry_rows] == (
        pytest.approx([102.4128, 68.2752])
    )
    assert [float(row["effective_bed_elevation_ft"]) for row in geometry_rows] == (
        pytest.approx([4995.030357142858, 4950.407142857143])
    )
    assert all(
        float(row["inferred_manning_n_model"]) > 0.0
        for row in geometry_rows
    )
    assert actual["status"] == (
        "post_baseline_measured_stage_geometry_experiment"
    )
    assert actual["boundary"]["x"] == "inflow_stage"
    provenance = actual["terrain_representation"]["provenance"]
    assert provenance["field_measurement_geometry"] == (
        RIO_CHANNEL_GEOMETRY.name
    )
    assert provenance["modeled_manning_n_range"] == pytest.approx(
        {
            "minimum": 0.000472853714174,
            "maximum": 0.000526252232655,
        }
    )
    assert "not used" in provenance[
        "configured_manning_n_role"
    ]
    for metric in ("nse", "rmse", "bias", "percent_bias", "pearson_r"):
        assert actual["scores"][metric] == pytest.approx(
            tracked["scores"][metric], rel=1e-10, abs=1e-10
        )

    manifest = json.loads(SUITE.read_text(encoding="utf-8"))
    assert RIO_STAGE_CASE.name not in manifest["cases"]
    assert RIO_GEOMETRY_CASE.name not in manifest["cases"]


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
    stage_manning = tmp_path / "stage_manning.csv"
    stage_manning.write_text(
        "station_m,depth_m,manning_n\n"
        "0,0,0.0005\n0,1,0.0006\n"
        "100,0,0.0007\n100,1,0.0008\n",
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
                    "validation_policy": {"calibration": "none"},
                        "hydraulic_dataset": "surveyed_topobathymetry",
                        "validation_2d": {
                            "representation": "ribbon",
                            "x_cells": 3,
                            "y_cells": 1,
                        },
                    "observations": observations.name,
                "reach": {
                    "length_m": 100.0,
                    "cells": 3,
                    "slope": 0.001,
                    "manning_n": 0.035 / 60.0,
                    "cross_section_shape": "surveyed",
                    "surveyed_cross_sections": surveys.name,
                    "stage_dependent_manning": stage_manning.name,
                },
                "record_interval_min": 5.0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="rectangular source sections only"):
        run_validation_case(config, output_path=tmp_path / "results.json")


def test_validation_case_uses_timestamped_stage_and_signed_point_flows(
    tmp_path,
):
    observations = tmp_path / "observations.csv"
    observations.write_text(
        "role,observed_at,discharge_m3_per_min\n"
        "upstream,2019-12-31T23:55:00Z,100\n"
        "upstream,2020-01-01T00:00:00Z,100\n"
        "upstream,2020-01-01T00:15:00Z,100\n"
        "downstream,2020-01-01T00:00:00Z,100\n"
        "downstream,2020-01-01T00:15:00Z,100\n",
        encoding="utf-8",
    )
    stage = tmp_path / "downstream_stage.csv"
    stage.write_text(
        "observed_at,downstream_stage_m\n"
        "2019-12-31T23:55:00Z,1.0\n"
        "2020-01-01T00:00:00Z,1.0\n"
        "2020-01-01T00:15:00Z,1.0\n",
        encoding="utf-8",
    )
    point_flows = tmp_path / "point_flows.csv"
    point_flows.write_text(
        "station_m,observed_at,discharge_m3_per_min\n"
        "50,2019-12-31T23:55:00Z,10\n"
        "50,2020-01-01T00:00:00Z,10\n"
        "50,2020-01-01T00:15:00Z,10\n"
        "100,2019-12-31T23:55:00Z,-5\n"
        "100,2020-01-01T00:00:00Z,-5\n"
        "100,2020-01-01T00:15:00Z,-5\n",
        encoding="utf-8",
    )
    config = tmp_path / "case.json"
    config.write_text(
        json.dumps(
            {
                    "case": {
                        "name": "measured controls",
                    "observation_window": [
                        "2020-01-01T00:00:00Z",
                        "2020-01-01T00:15:00Z",
                        ],
                    },
                    "validation_policy": {"calibration": "none"},
                        "hydraulic_dataset": "assumed_reach_geometry",
                        "validation_2d": {
                            "representation": "ribbon",
                            "x_cells": 3,
                            "y_cells": 1,
                        },
                    "observations": observations.name,
                "downstream_stage_series": stage.name,
                "point_flow_series": point_flows.name,
                "reach": {
                    "length_m": 100.0,
                    "cells": 3,
                    "slope": 0.001,
                    "manning_n": 0.035 / 60.0,
                    "upstream_width_m": 10.0,
                    "downstream_width_m": 10.0,
                },
                "warmup": {
                    "duration_min": 5.0,
                    "upstream_forcing": "observed",
                },
                "record_interval_min": 5.0,
            }
        ),
        encoding="utf-8",
    )

    result = run_validation_case(
        config, output_path=tmp_path / "results.json"
    )

    assert result["solver"] == "saint_venant_2d"
    assert result["boundary"]["point_flow_series"] == point_flows.name
    assert result["boundary"]["point_flow_count"] == 2
    assert result["mass"]["lateral_requested_m3"] == pytest.approx(75.0)
    assert result["mass"]["lateral_inflow_m3"] == pytest.approx(75.0)
    assert result["mass"]["unmet_withdrawal_m3"] == pytest.approx(0.0)


def test_measured_control_must_cover_observed_warmup(tmp_path):
    stage = tmp_path / "stage.csv"
    stage.write_text(
        "t_min,downstream_stage_m\n0,1.0\n15,1.1\n",
        encoding="utf-8",
    )
    control = load_event_control_series(
        stage,
        "downstream_stage_m",
        datetime.fromisoformat("2020-01-01T00:00:00+00:00"),
    )

    with pytest.raises(ValueError, match="full simulation"):
        _require_control_coverage(control, -5.0, 15.0, "stage")


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


def test_fetch_event_allows_only_explicit_small_endpoint_gap():
    config = {
        "case": {
            "observation_window": [
                "2020-01-01T00:00:00Z",
                "2020-01-01T00:15:00Z",
            ]
        },
        "observation_endpoint_tolerance_min": 5.0,
        "sources": [
            {"role": "upstream", "gauge": "USGS-1"},
            {"role": "downstream", "gauge": "USGS-2"},
        ],
    }

    def requester(url, params=None):
        return {
            "features": [
                {
                    "properties": {
                        "parameter_code": "00060",
                        "time": timestamp,
                        "value": 10.0,
                        "unit_of_measure": "ft^3/s",
                        "approval_status": "Approved",
                    }
                }
                for timestamp in (
                    "2020-01-01T00:00:00Z",
                    "2020-01-01T00:10:00Z",
                )
            ]
        }, f"https://example.test/{params['monitoring_location_id']}"

    rows, _ = collect_event_rows(config, requester=requester)
    assert len(rows) == 4
    config["observation_endpoint_tolerance_min"] = 4.0
    with pytest.raises(ValueError, match="configured end"):
        collect_event_rows(config, requester=requester)


def test_fetch_point_flows_preserves_station_and_requires_warmup_coverage():
    config = {
        "case": {
            "observation_window": [
                "2020-01-01T00:00:00Z",
                "2020-01-01T00:15:00Z",
            ]
        },
        "warmup": {"duration_min": 15.0},
        "internal_sources": [
            {"name": "tributary", "gauge": "USGS-3", "station_m": 1250.0}
        ],
    }

    def requester(url, params=None):
        assert params["datetime"] == (
            "2019-12-31T23:45:00Z/2020-01-01T00:15:00Z"
        )
        return {
            "features": [
                {
                    "properties": {
                        "parameter_code": "00060",
                        "time": timestamp,
                        "value": value,
                        "unit_of_measure": "ft^3/s",
                        "approval_status": "Approved",
                    }
                }
                for timestamp, value in (
                    ("2019-12-31T23:45:00Z", 10.0),
                    ("2020-01-01T00:00:00Z", 20.0),
                    ("2020-01-01T00:15:00Z", 15.0),
                )
            ]
        }, "https://example.test/USGS-3"

    rows, urls = collect_point_flow_rows(config, requester=requester)

    assert len(rows) == 3
    assert {row["source_name"] for row in rows} == {"tributary"}
    assert {float(row["station_m"]) for row in rows} == {1250.0}
    assert {row["approval_status"] for row in rows} == {"Approved"}
    assert set(urls) == {"tributary"}


def test_fetch_stage_control_converts_gage_height_to_model_datum():
    config = {
        "case": {
            "observation_window": [
                "2020-01-01T00:00:00Z",
                "2020-01-01T00:15:00Z",
            ]
        },
        "warmup": {"duration_min": 5.0},
        "downstream_stage_source": {
            "gauge": "USGS-1",
            "parameter": "00065",
            "gage_datum_ft": 90.0,
            "model_vertical_datum_ft": 100.0,
        },
    }

    def requester(url, params=None):
        assert params["datetime"] == (
            "2019-12-31T23:55:00Z/2020-01-01T00:15:00Z"
        )
        return {
            "features": [
                {
                    "properties": {
                        "parameter_code": "00065",
                        "time": timestamp,
                        "value": value,
                        "unit_of_measure": "ft",
                        "approval_status": "Approved",
                    }
                }
                for timestamp, value in (
                    ("2019-12-31T23:55:00Z", 12.0),
                    ("2020-01-01T00:15:00Z", 13.0),
                )
            ]
        }, "https://example.test/stage"

    rows, url = collect_stage_rows(config, requester=requester)

    assert url == "https://example.test/stage"
    assert len(rows) == 2
    assert float(rows[0]["downstream_stage_m"]) == pytest.approx(
        2.0 * 0.3048
    )
    assert float(rows[1]["downstream_stage_m"]) == pytest.approx(
        3.0 * 0.3048
    )
    assert {row["approval_status"] for row in rows} == {"Approved"}


def test_fetch_channel_geometry_aggregates_only_represented_channels():
    config = {
        "field_measurement_sources": [
            {
                "station_m": 0.0,
                "gauge": "USGS-1",
                "field_visit_id": "a",
                "gage_datum_ft": 90.0,
            },
            {
                "station_m": 100.0,
                "gauge": "USGS-2",
                "field_visit_id": "b",
                "gage_datum_ft": 80.0,
            },
        ]
    }

    def requester(url, params=None):
        visit = params["field_visit_id"]
        if "field-measurements" in url:
            return {
                "features": [
                    {
                        "properties": {
                            "field_visit_id": visit,
                            "parameter_code": "00065",
                            "reading_type": "MeanGageHeight",
                            "value": "12",
                            "unit_of_measure": "ft",
                            "approval_status": "Approved",
                        }
                    }
                ]
            }, f"https://example.test/{visit}/field"
        features = [
            {
                "properties": {
                    "field_visit_id": visit,
                    "time": "2020-01-01T00:00:00+00:00",
                    "channel_flow": "100",
                    "channel_flow_unit": "ft^3/s",
                    "channel_width": "20",
                    "channel_width_unit": "ft",
                    "channel_area": "40",
                    "channel_area_unit": "ft^2",
                }
            },
            {
                "properties": {
                    "field_visit_id": visit,
                    "time": "2020-01-01T00:00:00+00:00",
                    "channel_flow": "5",
                    "channel_flow_unit": "ft^3/s",
                    "channel_width": "",
                    "channel_width_unit": "",
                    "channel_area": "",
                    "channel_area_unit": "",
                }
            },
        ]
        return {"features": features}, f"https://example.test/{visit}"

    rows, urls = collect_channel_geometry_rows(
        config, requester=requester
    )

    assert len(rows) == 2
    assert float(rows[0]["active_width_m"]) == pytest.approx(20 * 0.3048)
    assert float(rows[0]["channel_area_m2"]) == pytest.approx(
        40 * 0.3048**2
    )
    assert float(rows[0]["hydraulic_mean_depth_m"]) == pytest.approx(
        2 * 0.3048
    )
    assert float(rows[0]["represented_flow_fraction"]) == pytest.approx(
        100 / 105
    )
    assert float(rows[0]["water_surface_elevation_ft"]) == pytest.approx(
        102.0
    )
    assert float(rows[0]["effective_bed_elevation_ft"]) == pytest.approx(
        100.0
    )
    assert rows[0]["represented_channel_count"] == 1
    assert rows[0]["published_channel_count"] == 2
    assert float(rows[0]["inferred_manning_n_model"]) > 0.0
    assert float(rows[0]["inferred_manning_n_si"]) == pytest.approx(
        60.0 * float(rows[0]["inferred_manning_n_model"])
    )
    assert set(urls) == {
        "USGS-1:a:channel",
        "USGS-1:a:field",
        "USGS-2:b:channel",
        "USGS-2:b:field",
    }


def test_fetch_channel_geometry_builds_multiple_dated_snapshots():
    config = {
        "field_measurement_sources": [
            {
                "station_m": station,
                "gauge": gauge,
                "field_visit_id": f"{snapshot}-{role}",
                "gage_datum_ft": datum,
                "geometry_snapshot": snapshot,
            }
            for snapshot in ("old", "new")
            for station, gauge, datum, role in (
                (0.0, "USGS-1", 90.0, "up"),
                (100.0, "USGS-2", 80.0, "down"),
            )
        ]
    }

    def requester(url, params=None):
        visit = params["field_visit_id"]
        snapshot, role = visit.split("-")
        if "field-measurements" in url:
            gage_height = 12.0 if snapshot == "old" else 14.0
            return {
                "features": [
                    {
                        "properties": {
                            "field_visit_id": visit,
                            "parameter_code": "00065",
                            "reading_type": "MeanGageHeight",
                            "value": str(gage_height),
                            "unit_of_measure": "ft",
                            "approval_status": "Approved",
                        }
                    }
                ]
            }, f"https://example.test/{visit}/field"
        width = {
            ("old", "up"): 20.0,
            ("old", "down"): 30.0,
            ("new", "up"): 40.0,
            ("new", "down"): 60.0,
        }[(snapshot, role)]
        depth = 2.0 if snapshot == "old" else 3.0
        return {
            "features": [
                {
                    "properties": {
                        "field_visit_id": visit,
                        "time": (
                            "2019-01-01T00:00:00+00:00"
                            if snapshot == "old"
                            else "2021-01-01T00:00:00+00:00"
                        ),
                        "channel_flow": "100",
                        "channel_flow_unit": "ft^3/s",
                        "channel_width": str(width),
                        "channel_width_unit": "ft",
                        "channel_area": str(width * depth),
                        "channel_area_unit": "ft^2",
                    }
                }
            ]
        }, f"https://example.test/{visit}/channel"

    rows, _ = collect_channel_geometry_rows(config, requester=requester)

    assert len(rows) == 4
    assert {row["geometry_snapshot"] for row in rows} == {"old", "new"}
    assert [float(row["station_m"]) for row in rows] == [
        0.0,
        0.0,
        100.0,
        100.0,
    ]
    assert all(float(row["inferred_manning_n_model"]) > 0.0 for row in rows)


def test_field_geometry_catalog_selects_latest_pre_event_snapshot(tmp_path):
    catalog = tmp_path / "geometry_catalog.csv"
    catalog.write_text(
        "geometry_snapshot,station_m,measured_at,active_width_m,"
        "effective_bed_elevation_ft,inferred_manning_n_model\n"
        "old,0,2019-01-01T00:00:00Z,20,100,0.0005\n"
        "new,0,2021-01-01T00:00:00Z,40,101,0.0007\n"
        "old,100,2019-02-01T00:00:00Z,30,90,0.0006\n"
        "new,100,2021-02-01T00:00:00Z,50,91,0.0008\n",
        encoding="utf-8",
    )

    width, bed, roughness, selection = load_field_measurement_geometry(
        catalog,
        [0.0, 50.0, 100.0],
        event_start=datetime.fromisoformat("2020-01-01T00:00:00+00:00"),
        time_policy="latest_not_after_event_start",
    )

    assert width == pytest.approx([20.0, 25.0, 30.0])
    assert bed == pytest.approx([0.0, -1.524, -3.048])
    assert roughness == pytest.approx([0.0005, 0.00055, 0.0006])
    assert selection["policy"] == "latest_not_after_event_start"
    assert selection["selection_mode"] == "latest_complete_snapshot"
    assert selection["selected_snapshot"] == "old"
    assert [
        row["measured_at"] for row in selection["selected_rows"]
    ] == [
        "2019-01-01T00:00:00+00:00",
        "2019-02-01T00:00:00+00:00",
    ]


def test_field_geometry_catalog_rejects_future_only_or_stale_state(tmp_path):
    catalog = tmp_path / "geometry_catalog.csv"
    catalog.write_text(
        "station_m,measured_at,active_width_m,"
        "effective_bed_elevation_ft,inferred_manning_n_model\n"
        "0,2019-01-01T00:00:00Z,20,100,0.0005\n"
        "100,2021-01-01T00:00:00Z,30,90,0.0006\n",
        encoding="utf-8",
    )
    event_start = datetime.fromisoformat("2020-01-01T00:00:00+00:00")

    with pytest.raises(ValueError, match="at or before event start"):
        load_field_measurement_geometry(
            catalog,
            [0.0, 100.0],
            event_start=event_start,
            time_policy="latest_not_after_event_start",
        )

    catalog.write_text(
        "station_m,measured_at,active_width_m,"
        "effective_bed_elevation_ft,inferred_manning_n_model\n"
        "0,2018-01-01T00:00:00Z,20,100,0.0005\n"
        "100,2018-01-01T00:00:00Z,30,90,0.0006\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exceeding"):
        load_field_measurement_geometry(
            catalog,
            [0.0, 100.0],
            event_start=event_start,
            time_policy="latest_not_after_event_start",
            max_age_days=30.0,
        )


def test_suite_summary_uses_event_ranges_and_medians():
    summary = summarize_results(
        [
            {"scores": {key: value for key in ("nse", "rmse", "bias", "percent_bias", "pearson_r")}}
            for value in (1.0, 3.0, 2.0)
        ]
    )

    assert summary["nse"] == {"minimum": 1.0, "median": 2.0, "maximum": 3.0}
