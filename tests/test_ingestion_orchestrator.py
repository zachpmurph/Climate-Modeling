"""Tests for config-driven ingestion: a curated JSON reach definition drives the
whole pipeline, one reach or a batch, with per-reach success/failure reporting.

Offline: providers are injected via the ``requesters`` mapping.
"""

import csv
import json
from pathlib import Path

import pytest

from rivers.ingest.orchestrator import ingest_all, ingest_reach, load_definition
from general.solvers.profile import load_profile

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _requesters():
    return {
        "elevation": lambda url, params: (
            {"elevation": [30.0, 25.0, 20.0]}, "https://example.test/elev"
        ),
        "rainfall": lambda url, params: (
            {"hourly": {"time": ["2020-01-01T00:00", "2020-01-01T01:00"], "precipitation": [1.2, 0.4]}},
            "https://example.test/rain",
        ),
    }


def _make_definition(dir_path, name="columbia", output_name="columbia.profile.csv"):
    _write_csv(
        dir_path / "markers.csv",
        ["lat", "lon", "station_m", "label"],
        [{"lat": 46.6, "lon": -119.9, "station_m": 0, "label": "up"},
         {"lat": 46.5, "lon": -119.6, "station_m": 1000, "label": "mid"},
         {"lat": 46.4, "lon": -119.4, "station_m": 2000, "label": "down"}],
    )
    _write_csv(
        dir_path / "roughness.csv",
        ["start_station_m", "end_station_m", "manning_n", "method"],
        [{"start_station_m": 0, "end_station_m": 1000, "manning_n": 0.035 / 60, "method": "survey"},
         {"start_station_m": 1000, "end_station_m": 2000, "manning_n": 0.04 / 60, "method": "survey"}],
    )
    definition = {
        "river": {"name": "Columbia", "region": "WA", "country": "US"},
        "reach": {"name": name},
        "markers": "markers.csv",
        "elevation": {"provider": "open-meteo-dem"},
        "roughness": {"file": "roughness.csv"},
        "rainfall": {"start_date": "2020-01-01", "end_date": "2020-01-01", "marker_order": 1},
        "export": {"output": output_name, "minimum_slope": 1e-6},
    }
    def_path = dir_path / f"{name}.json"
    def_path.write_text(json.dumps(definition, indent=2), encoding="utf-8")
    return def_path


def test_ingest_reach_produces_loadable_validated_profile(tmp_path):
    def_path = _make_definition(tmp_path)
    db_path = tmp_path / "river.sqlite"

    result = ingest_reach(def_path, db_path=db_path, requesters=_requesters())

    assert result["status"] == "ok"
    output = tmp_path / "columbia.profile.csv"
    assert output.exists()
    profile = load_profile(output)
    assert len(profile.station_m) == 3
    assert result["export"]["validation"]["counts"]["error"] == 0


def test_ingest_reach_is_rerunnable_without_duplicates(tmp_path):
    def_path = _make_definition(tmp_path)
    db_path = tmp_path / "river.sqlite"

    first = ingest_reach(def_path, db_path=db_path, requesters=_requesters(), replace=True)
    second = ingest_reach(def_path, db_path=db_path, requesters=_requesters(), replace=True)

    assert first["status"] == "ok"
    assert second["status"] == "ok"
    # Same reach reused, not duplicated.
    assert first["reach_id"] == second["reach_id"]


def test_ingest_reach_rerun_without_replace_is_idempotent(tmp_path):
    def_path = _make_definition(tmp_path)
    db_path = tmp_path / "river.sqlite"

    first = ingest_reach(def_path, db_path=db_path, requesters=_requesters())
    second = ingest_reach(def_path, db_path=db_path, requesters=_requesters())  # no replace

    assert first["status"] == "ok"
    assert second["status"] == "ok", "re-running without --replace must reuse the reach, not fail"
    assert first["reach_id"] == second["reach_id"]
    assert second["steps"]["reach"]["reused"] is True


def test_ingest_all_ignores_generated_sidecars_in_curated_dir(tmp_path):
    curated = tmp_path / "curated"
    curated.mkdir()
    _make_definition(curated, name="one", output_name="one.profile.csv")

    # First run writes one.profile.metadata.json (and .profile.csv) into curated/.
    ingest_all(curated, db_path=tmp_path / "river.sqlite", requesters=_requesters())
    assert (curated / "one.profile.metadata.json").exists()

    # A second run must NOT treat that generated sidecar as a definition.
    summary = ingest_all(curated, db_path=tmp_path / "river.sqlite", requesters=_requesters())
    assert summary["total"] == 1, "generated .metadata.json must not be globbed as a definition"
    assert summary["overall_success"] is True


def test_ingest_all_empty_directory_is_not_success(tmp_path):
    empty = tmp_path / "curated"
    empty.mkdir()
    summary = ingest_all(empty, db_path=tmp_path / "river.sqlite", requesters=_requesters())
    assert summary["total"] == 0
    assert summary["overall_success"] is False


def test_ingest_all_reports_both_successes_and_failures(tmp_path):
    curated = tmp_path / "curated"
    curated.mkdir()
    _make_definition(curated, name="good", output_name="good.profile.csv")

    # A broken definition: its markers file does not exist.
    broken = {
        "river": {"name": "Nowhere"},
        "reach": {"name": "bad"},
        "markers": "missing.csv",
        "roughness": {"file": "missing.csv"},
        "export": {"output": "bad.profile.csv"},
    }
    (curated / "bad.json").write_text(json.dumps(broken), encoding="utf-8")

    summary = ingest_all(curated, db_path=tmp_path / "river.sqlite", requesters=_requesters())

    assert summary["total"] == 2
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1
    assert summary["overall_success"] is False
    statuses = {r["reach"]: r["status"] for r in summary["reaches"]}
    assert statuses["good"] == "ok"
    assert statuses["bad"] == "error"


def test_ingest_all_overall_success_when_all_exports_present(tmp_path):
    curated = tmp_path / "curated"
    curated.mkdir()
    _make_definition(curated, name="one", output_name="one.profile.csv")

    summary = ingest_all(curated, db_path=tmp_path / "river.sqlite", requesters=_requesters())

    assert summary["overall_success"] is True
    assert (curated / "one.profile.csv").exists()


def test_geometry_without_flow_yields_no_inflow_recommendation(tmp_path):
    def_path = _make_definition(tmp_path, name="geo", output_name="geo.profile.csv")
    _write_csv(
        tmp_path / "geometry.csv",
        ["station_m", "width_m", "bankfull_depth_m"],
        [{"station_m": 0, "width_m": 300, "bankfull_depth_m": 4.5}],
    )
    definition = json.loads(def_path.read_text())
    definition["geometry"] = {"file": "geometry.csv"}  # geometry present, but no "flow" section
    def_path.write_text(json.dumps(definition), encoding="utf-8")

    result = ingest_reach(def_path, db_path=tmp_path / "river.sqlite", requesters=_requesters())
    assert result["status"] == "ok"

    metadata = json.loads((tmp_path / "geo.profile.metadata.json").read_text())
    assert metadata["recommended_upstream_inflow"] is None, \
        "without flow observations there is no basis for an inflow recommendation"


def test_shipped_columbia_definition_is_valid_and_its_files_exist():
    def_path = REPO_ROOT / "real_world_rivers" / "curated" / "columbia_hanford.json"
    definition = load_definition(def_path)
    base = def_path.parent

    assert (base / definition["markers"]).resolve().exists()
    assert (base / definition["roughness"]["file"]).resolve().exists()
    if "geometry" in definition:
        assert (base / definition["geometry"]["file"]).resolve().exists()
    assert definition["export"]["output"]
