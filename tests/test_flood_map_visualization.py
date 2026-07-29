import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rivers.visualization import animate_flood_map


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_inundation_width_expands_symmetrically_above_bankfull():
    widths = animate_flood_map.inundation_widths(
        depths_m=np.array([1.0, 3.0]),
        channel_widths_m=np.array([100.0, 100.0]),
        bankfull_depths_m=np.array([2.0, 2.0]),
        floodplain_slope=0.02,
        max_width_m=1_000.0,
    )

    assert widths.tolist() == pytest.approx([100.0, 200.0])


def test_dry_depths_do_not_create_flood_extent():
    widths = animate_flood_map.inundation_widths(
        depths_m=np.array([0.0, 1e-10]),
        channel_widths_m=np.array([100.0, 100.0]),
        bankfull_depths_m=np.array([2.0, 2.0]),
        floodplain_slope=0.02,
        max_width_m=1_000.0,
    )

    assert widths.tolist() == [0.0, 0.0]

    frames = animate_flood_map.build_frames(
        stations=np.array([0.0, 1_000.0]),
        times=np.array([0.0]),
        depths=np.array([[0.0, 1e-10]]),
        centerline=np.array([[40.0, -120.0], [39.991, -120.0]]),
        channel_widths_m=np.array([100.0, 100.0]),
        bankfull_depths_m=np.array([2.0, 2.0]),
        floodplain_slope=0.02,
        max_width_m=1_000.0,
    )

    assert frames[0]["max_width_m"] == 0.0
    assert frames[0]["segments"] == []


def test_named_river_uses_paths_from_curated_definition():
    markers, geometry = animate_flood_map._named_map_inputs("columbia-hanford")

    assert markers == REPO_ROOT / "real_world_rivers" / "columbia_hanford_markers.csv"
    assert geometry == REPO_ROOT / "real_world_rivers" / "columbia_hanford_geometry.csv"
    assert markers.is_file()
    assert geometry.is_file()


def test_generator_uses_paired_summary_map_inputs(tmp_path):
    timeseries = tmp_path / "test_run_timeseries.csv"
    timeseries.write_text(
        "t_min,0,1000\n"
        "0,1,1\n"
        "10,3,3\n",
        encoding="utf-8",
    )
    markers = tmp_path / "markers.csv"
    _write_csv(
        markers,
        ["lat", "lon", "station_m", "label"],
        [
            {"lat": 40.0, "lon": -120.0, "station_m": 0, "label": "upstream"},
            {"lat": 39.991, "lon": -120.0, "station_m": 1000, "label": "downstream"},
        ],
    )
    geometry = tmp_path / "geometry.csv"
    _write_csv(
        geometry,
        ["station_m", "width_m", "bankfull_depth_m"],
        [{"station_m": 0, "width_m": 100, "bankfull_depth_m": 2}],
    )
    summary = tmp_path / "test_run_summary.json"
    summary.write_text(
        json.dumps(
            {
                "river": "test-river",
                "model": "saint-venant",
                "event": "extreme-rain",
                "map_inputs": {"markers": str(markers), "geometry": str(geometry)},
            }
        ),
        encoding="utf-8",
    )

    args = animate_flood_map.parse_args([str(timeseries)])
    output_path = animate_flood_map.run(args)
    document = output_path.read_text(encoding="utf-8")

    assert output_path.name == "test_run_flood_map.html"
    assert "Test River flood extent" in document
    assert "tile.opentopomap.org" in document
    assert '"cross_section_depth":"uniform"' in document
    assert '"max_width_m":200.0' in document


def test_centerline_rejects_simulation_stations_outside_markers():
    with pytest.raises(ValueError, match="extend beyond"):
        animate_flood_map.interpolate_centerline(
            stations=np.array([0.0, 1100.0]),
            marker_stations=np.array([0.0, 1000.0]),
            marker_coordinates=np.array([[40.0, -120.0], [39.99, -120.0]]),
        )


def test_mapped_centerline_retains_bends_between_solver_stations():
    stations, coordinates = animate_flood_map.mapped_centerline_samples(
        simulation_stations=np.array([0.0, 1000.0]),
        centerline_stations=np.array([0.0, 500.0, 1000.0]),
        centerline_coordinates=np.array(
            [[40.0, -120.0], [40.005, -119.995], [40.0, -119.99]]
        ),
    )

    assert stations.tolist() == [0.0, 500.0, 1000.0]
    assert coordinates[1].tolist() == pytest.approx([40.005, -119.995])


def test_visualization_script_runs_as_documented(tmp_path):
    timeseries = tmp_path / "cli_timeseries.csv"
    timeseries.write_text("t_min,0,1000\n0,1,1\n", encoding="utf-8")
    markers = tmp_path / "markers.csv"
    _write_csv(
        markers,
        ["lat", "lon", "station_m"],
        [
            {"lat": 40.0, "lon": -120.0, "station_m": 0},
            {"lat": 39.991, "lon": -120.0, "station_m": 1000},
        ],
    )
    geometry = tmp_path / "geometry.csv"
    _write_csv(
        geometry,
        ["station_m", "width_m", "bankfull_depth_m"],
        [{"station_m": 0, "width_m": 100, "bankfull_depth_m": 2}],
    )

    completed = subprocess.run(
        [
            sys.executable,
            "src/rivers/visualization/animate_flood_map.py",
            str(timeseries),
            "--markers",
            str(markers),
            "--geometry",
            str(geometry),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Animated flood map" in completed.stdout
    assert (tmp_path / "cli_flood_map.html").exists()
