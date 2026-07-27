import json
import subprocess
import sys

import numpy as np
import pytest

from rivers.reporting.generate_flood_report import (
    FieldSeries2D,
    TimeSeries,
    calculate_outcomes,
    calculate_outcomes_2d,
    generate_report,
    load_time_series,
)


def _write_run(tmp_path):
    timeseries = tmp_path / "example_timeseries.csv"
    timeseries.write_text(
        "t_min,0,100,200\n"
        "0,1,1,1\n"
        "10,1,3,1\n"
        "20,1,4,3\n",
        encoding="utf-8",
    )
    summary = tmp_path / "example_summary.json"
    summary.write_text(
        json.dumps(
            {
                "solver": "fixture",
                "mass_inflow": 10,
                "mass_source": 4,
                "mass_outflow": 3,
                "mass_balance_error": 0.0002,
            }
        ),
        encoding="utf-8",
    )
    geometry = tmp_path / "geometry.csv"
    geometry.write_text(
        "station_m,bankfull_depth_m\n"
        "0,2\n"
        "200,2\n",
        encoding="utf-8",
    )
    return timeseries, summary, geometry


def test_load_time_series_rejects_invalid_coordinates_and_depth(tmp_path):
    invalid_stations = tmp_path / "stations.csv"
    invalid_stations.write_text("t_min,0,0\n0,1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strictly increasing"):
        load_time_series(invalid_stations)

    negative_depth = tmp_path / "depth.csv"
    negative_depth.write_text("t_min,0,1\n0,1,-1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-negative"):
        load_time_series(negative_depth)


def test_calculate_outcomes_uses_bankfull_threshold():
    series = TimeSeries(
        stations_m=np.array([0.0, 100.0, 200.0]),
        times_min=np.array([0.0, 10.0, 20.0]),
        depth_m=np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 3.0, 1.0],
                [1.0, 4.0, 3.0],
            ]
        ),
    )

    outcome = calculate_outcomes(
        series,
        {"mass_balance_error": 0.001},
        threshold_m=np.array([2.0, 2.0, 2.0]),
        threshold_source="bankfull_depth_profile",
    )

    assert outcome["metrics"]["max_depth_m"] == 4.0
    assert outcome["metrics"]["peak_time_min"] == 20.0
    assert outcome["metrics"]["peak_station_m"] == 100.0
    assert outcome["metrics"]["first_exceedance_time_min"] == 10.0
    assert outcome["metrics"]["max_exceedance_depth_m"] == 2.0
    assert outcome["metrics"]["max_exceedance_length_m"] == pytest.approx(200.0)
    assert outcome["metrics"]["max_exceedance_fraction"] == pytest.approx(2 / 3)


def test_generate_report_writes_html_and_machine_readable_outcomes(tmp_path):
    timeseries, summary, geometry = _write_run(tmp_path)

    report_path, outcome_path = generate_report(
        timeseries,
        summary_path=summary,
        geometry_path=geometry,
        title="Example River screening report",
    )

    report = report_path.read_text(encoding="utf-8")
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    assert "<title>Example River screening report</title>" in report
    assert 'id="time-slider"' in report
    assert "Screening output only" in report
    assert outcome["schema_version"] == 1
    assert outcome["metrics"]["threshold"]["source"] == "bankfull_depth_profile"
    assert outcome["sources"]["summary"] == str(summary)


def test_report_without_threshold_is_explicitly_unassessed(tmp_path):
    timeseries, _, _ = _write_run(tmp_path)

    _, outcome_path = generate_report(timeseries)

    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    assert outcome["metrics"]["threshold"] is None
    assert "not calculated" in outcome["warnings"][-1]


def test_reporting_cli_runs_from_repository_root(tmp_path):
    timeseries, summary, geometry = _write_run(tmp_path)
    report_path = tmp_path / "cli-report.html"

    completed = subprocess.run(
        [
            sys.executable,
            "src/rivers/reporting/generate_flood_report.py",
            str(timeseries),
            "--summary",
            str(summary),
            "--geometry",
            str(geometry),
            "--output",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert report_path.exists()
    assert report_path.with_suffix(".outcomes.json").exists()


def test_2d_outcomes_measure_area_and_report_plan_view(tmp_path):
    fields = tmp_path / "example_fields.npz"
    depth = np.array([
        [[0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.6], [0.8, 0.0]],
    ])
    np.savez_compressed(
        fields,
        x_m=[0.0, 10.0],
        y_m=[1.0, 3.0],
        dx_m=[10.0, 10.0],
        dy_m=[2.0, 2.0],
        times_min=[0.0, 1.0],
        depth_m=depth,
    )
    timeseries = tmp_path / "example_timeseries.csv"
    timeseries.write_text("t_min,0,10\n0,0,0\n1,0.3,0.4\n", encoding="utf-8")
    summary = tmp_path / "example_summary.json"
    summary.write_text(json.dumps({
        "dimension": 2,
        "fields_path": str(fields),
        "mass_balance_error": 0.0,
    }), encoding="utf-8")

    report_path, outcomes_path = generate_report(
        timeseries,
        summary_path=summary,
        depth_threshold_m=0.5,
    )
    outcomes = json.loads(outcomes_path.read_text(encoding="utf-8"))
    assert outcomes["dimension"] == 2
    assert outcomes["metrics"]["max_exceedance_area_m2"] == pytest.approx(40.0)
    assert outcomes["metrics"]["peak_x_m"] == 10.0
    assert 'id="depth-map"' in report_path.read_text(encoding="utf-8")
