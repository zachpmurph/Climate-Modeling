"""Export-time behaviour: validation gating, provenance-rich metadata,
adjustment records, and atomic writes.

Offline: the reach is built from injected provider requesters.
"""

import csv
import json

import pytest

from rivers.ingest import export_profile as export_module
from rivers.ingest.export_profile import export_profile
from rivers.ingest.markers import create_reach
from rivers.ingest.parameters import import_roughness
from rivers.ingest.elevation import collect_elevations
from rivers.ingest.rainfall import collect_rainfall
from rivers.ingest.validation import ValidationReport
from general.solvers.profile import load_profile


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_reach(tmp_path):
    marker_path = tmp_path / "markers.csv"
    _write_csv(
        marker_path,
        ["lat", "lon", "station_m", "label"],
        [
            {"lat": 46.6, "lon": -119.9, "station_m": 0, "label": "up"},
            {"lat": 46.5, "lon": -119.6, "station_m": 1000, "label": "mid"},
            {"lat": 46.4, "lon": -119.4, "station_m": 2000, "label": "down"},
        ],
    )
    db_path = tmp_path / "river.sqlite"
    reach_id = create_reach("Columbia", "Hanford", marker_path, country="US", db_path=db_path)

    collect_elevations(
        reach_id, db_path=db_path,
        requester=lambda url, params: ({"elevation": [30.0, 25.0, 20.0]}, "https://example.test/elev"),
    )
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
    collect_rainfall(
        reach_id, "2020-01-01", "2020-01-01", db_path=db_path, marker_order=1,
        requester=lambda url, params: (
            {"hourly": {"time": ["2020-01-01T00:00", "2020-01-01T01:00"], "precipitation": [1.2, 0.6]}},
            "https://example.test/rain",
        ),
    )
    return db_path, reach_id


def test_export_metadata_has_validation_and_per_value_classification(tmp_path):
    db_path, reach_id = _build_reach(tmp_path)
    output = tmp_path / "profile.csv"

    metadata = export_profile(reach_id, output, db_path=db_path)

    # The exported profile still loads through the documented interface.
    profile = load_profile(output)
    assert len(profile.station_m) == 3

    # Validation findings are retained in metadata.
    assert "validation" in metadata
    assert isinstance(metadata["validation"]["findings"], list)
    assert metadata["validation"]["counts"]["error"] == 0

    # Every exported value is classified.
    assert metadata["rows"][0]["classification"]["station_m"] == "observed"
    assert metadata["rows"][0]["classification"]["slope"] in {"derived", "fallback"}
    assert metadata["rows"][0]["classification"]["manning_n"] == "estimated"

    # Provenance sources are recorded.
    assert metadata["sources"], "expected data_sources provenance in metadata"


def test_slope_floor_adjustment_is_recorded_with_original_and_adjusted(tmp_path):
    db_path, reach_id = _build_reach(tmp_path)
    output = tmp_path / "profile.csv"

    # Flat elevations -> zero slope -> every interval floored.
    collect_elevations(
        reach_id, db_path=db_path, replace=True,
        requester=lambda url, params: ({"elevation": [20.0, 20.0, 20.0]}, "https://example.test/flat"),
    )
    metadata = export_profile(reach_id, output, db_path=db_path, minimum_slope=1e-6)

    adjustments = metadata["adjustments"]
    assert adjustments, "expected slope-floor adjustments to be recorded"
    first = adjustments[0]
    assert first["rule"] == "minimum_slope_floor"
    assert first["original"] == 0.0
    assert first["adjusted"] == pytest.approx(1e-6)
    assert metadata["slope_values_adjusted"] == len(adjustments)
    # Floored slopes are reclassified as fallback.
    assert any(row["classification"]["slope"] == "fallback" for row in metadata["rows"])


def test_export_aborts_and_writes_nothing_on_validation_error(tmp_path, monkeypatch):
    db_path, reach_id = _build_reach(tmp_path)
    output = tmp_path / "profile.csv"

    def failing_validate(rows, **kwargs):
        report = ValidationReport()
        report.add("error", "injected", "synthetic validation error for test")
        return report

    monkeypatch.setattr(export_module, "validate_profile", failing_validate)

    with pytest.raises(export_module.ProfileValidationError):
        export_profile(reach_id, output, db_path=db_path)

    # No misleading partial outputs.
    assert not output.exists()
    assert not output.with_suffix(".metadata.json").exists()


def test_export_overwrites_atomically_leaving_no_temp_files(tmp_path):
    db_path, reach_id = _build_reach(tmp_path)
    output = tmp_path / "profile.csv"

    export_profile(reach_id, output, db_path=db_path)
    export_profile(reach_id, output, db_path=db_path)  # second run overwrites

    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp" in p.name or p.name.endswith("~")]
    assert leftovers == [], f"temp files left behind: {leftovers}"
    assert output.exists()
    assert output.with_suffix(".metadata.json").exists()
