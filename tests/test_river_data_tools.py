import csv
import json
import sqlite3

import pytest

from general.solvers import river_kinematic_wave
from rivers.ingest.common import haversine_m
from rivers.ingest.elevation import collect_elevations, parse_elevations
from rivers.ingest.export_profile import export_profile
from rivers.ingest.markers import create_reach, load_marker_rows
from rivers.ingest.parameters import import_geometry, import_roughness
from rivers.ingest.rainfall import collect_rainfall, parse_hourly_precipitation
from rivers.ingest.usgs_flow import (
    CFS_TO_M3_PER_MIN,
    collect_usgs_flow,
    discharge_to_m3_per_min,
    fetch_usgs_flow,
)


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_reach(tmp_path):
    marker_path = tmp_path / "markers.csv"
    _write_csv(
        marker_path,
        ["lat", "lon", "station_m", "label"],
        [
            {"lat": 40.0, "lon": -120.0, "station_m": 0, "label": "upstream"},
            {"lat": 39.99, "lon": -120.0, "station_m": 1000, "label": "middle"},
            {"lat": 39.98, "lon": -120.0, "station_m": 2000, "label": "downstream"},
        ],
    )
    db_path = tmp_path / "river.sqlite"
    reach_id = create_reach(
        "Test River", "Test Reach", marker_path, country="US", db_path=db_path
    )
    return db_path, reach_id


def test_marker_import_infers_stations(tmp_path):
    marker_path = tmp_path / "markers.json"
    marker_path.write_text(
        json.dumps(
            [
                {"lat": 40.0, "lon": -120.0},
                {"lat": 39.99, "lon": -120.0},
            ]
        ),
        encoding="utf-8",
    )

    rows = load_marker_rows(marker_path)

    assert rows[0]["station_m"] == 0
    assert rows[1]["station_m"] == pytest.approx(haversine_m(40.0, -120.0, 39.99, -120.0))


def test_provider_parsers_and_unit_conversion():
    assert parse_elevations({"elevation": [10, 9]}, 2) == [10.0, 9.0]
    rainfall = parse_hourly_precipitation(
        {"hourly": {"time": ["2020-01-01T00:00"], "precipitation": [1.5]}}
    )
    assert rainfall[0]["interval_min"] == 60
    assert discharge_to_m3_per_min(2, "ft3/s") == pytest.approx(2 * CFS_TO_M3_PER_MIN)
    assert discharge_to_m3_per_min(2, "m3/s") == pytest.approx(120)
    with pytest.raises(ValueError, match="Unsupported"):
        discharge_to_m3_per_min(2, "gallons/day")


def test_usgs_flow_follows_pagination():
    def feature(value, observed_at):
        return {
            "properties": {
                "parameter_code": "00060",
                "time": observed_at,
                "value": value,
                "unit_of_measure": "ft^3/s",
            }
        }

    def requester(url, params=None):
        if params is not None:
            return (
                {
                    "features": [feature("1", "2020-01-01T00:00:00Z")],
                    "links": [{"rel": "next", "href": "https://example.test/page-2"}],
                },
                "https://example.test/page-1?api_key=secret",
            )
        return {"features": [feature("2", "2020-01-01T00:15:00Z")]}, url

    observations, source_url, _ = fetch_usgs_flow(
        "01234567",
        "2020-01-01T00:00:00Z",
        "2020-01-01T01:00:00Z",
        api_key="secret",
        requester=requester,
    )

    assert [row["value"] for row in observations] == [1.0, 2.0]
    assert "secret" not in source_url
    assert "api_key=REDACTED" in source_url


def test_collection_and_profile_export_pipeline(tmp_path):
    db_path, reach_id = _make_reach(tmp_path)

    def elevation_requester(url, params):
        assert len(params["latitude"].split(",")) == 3
        return {"elevation": [30.0, 31.0, 20.0]}, "https://example.test/elevation"

    elevation_result = collect_elevations(
        reach_id, db_path=db_path, requester=elevation_requester
    )
    assert elevation_result["slope_count"] == 2

    roughness_path = tmp_path / "roughness.csv"
    _write_csv(
        roughness_path,
        ["start_station_m", "end_station_m", "manning_n", "method"],
        [
            {"start_station_m": 0, "end_station_m": 1000, "manning_n": 0.035, "method": "survey"},
            {"start_station_m": 1000, "end_station_m": 2000, "manning_n": 0.04, "method": "survey"},
        ],
    )
    import_roughness(reach_id, roughness_path, db_path=db_path)

    geometry_path = tmp_path / "geometry.csv"
    _write_csv(
        geometry_path,
        ["station_m", "width_m", "bankfull_depth_m", "method"],
        [{"station_m": 0, "width_m": 10, "bankfull_depth_m": 2, "method": "survey"}],
    )
    import_geometry(reach_id, geometry_path, db_path=db_path)

    def flow_requester(url, params):
        features = []
        for index, value in enumerate((100, 200)):
            features.append(
                {
                    "properties": {
                        "parameter_code": "00060",
                        "time": f"2020-01-01T0{index}:00:00Z",
                        "value": str(value),
                        "unit_of_measure": "ft3/s",
                        "approval_status": "Approved",
                    }
                }
            )
        return {"features": features}, "https://example.test/flow"

    collect_usgs_flow(
        reach_id,
        "01234567",
        "2020-01-01T00:00:00Z",
        "2020-01-01T02:00:00Z",
        db_path=db_path,
        requester=flow_requester,
    )

    def rainfall_requester(url, params):
        return (
            {
                "hourly": {
                    "time": ["2020-01-01T00:00", "2020-01-01T01:00"],
                    "precipitation": [1.2, 0.0],
                }
            },
            "https://example.test/rainfall",
        )

    collect_rainfall(
        reach_id,
        "2020-01-01",
        "2020-01-01",
        db_path=db_path,
        requester=rainfall_requester,
    )

    output_path = tmp_path / "profile.csv"
    metadata = export_profile(
        reach_id,
        output_path,
        db_path=db_path,
        initial_depth_m=0.1,
        rainfall_start="2020-01-01T00:00",
        rainfall_end="2020-01-01T01:00",
        flow_start="2020-01-01T00:00:00Z",
        flow_end="2020-01-01T02:00:00Z",
    )
    profile = river_kinematic_wave.load_profile(output_path)

    assert metadata["slope_values_adjusted"] == 1
    assert metadata["recommended_upstream_inflow"]["left_inflow_flux_m2_per_min"] == pytest.approx(
        150 * CFS_TO_M3_PER_MIN / 10
    )
    assert profile.slope[0] == pytest.approx(1e-6)
    assert profile.manning_n.tolist() == pytest.approx([0.035, 0.04, 0.04])
    assert profile.rainfall_rate_m_per_min.tolist() == pytest.approx([1e-5] * 3)
    assert output_path.with_suffix(".metadata.json").exists()

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM data_sources").fetchone()[0] == 6
