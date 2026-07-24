"""Reliability tests for the ingestion pipeline: deduplication, atomic writes,
retry/error behaviour, and provenance classification.

All tests are offline: providers are supplied as injected ``requester`` callables
or fixture payloads. No test in this module touches the network.
"""

import csv
import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from rivers.ingest import common
from rivers.ingest.common import add_source, connect_database, redact_url, request_json
from rivers.ingest.elevation import collect_elevations
from rivers.ingest.markers import create_reach

REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeResponse:
    def __init__(self, payload, url):
        self._payload = json.dumps(payload).encode("utf-8")
        self._url = url

    def read(self, *args):
        return self._payload

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
from rivers.ingest.rainfall import collect_rainfall
from rivers.ingest.usgs_flow import collect_usgs_flow


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
    reach_id = create_reach("Test River", "Test Reach", marker_path, country="US", db_path=db_path)
    return db_path, reach_id


def _elevation_requester(url, params):
    return {"elevation": [30.0, 31.0, 20.0]}, "https://example.test/elevation"


def _flow_requester(url, params=None):
    features = [
        {
            "properties": {
                "parameter_code": "00060",
                "time": f"2020-01-01T0{index}:00:00Z",
                "value": str(value),
                "unit_of_measure": "ft3/s",
                "approval_status": "Approved",
            }
        }
        for index, value in enumerate((100, 200))
    ]
    return {"features": features}, "https://example.test/flow"


def _rainfall_requester(url, params=None):
    return (
        {"hourly": {"time": ["2020-01-01T00:00", "2020-01-01T01:00"], "precipitation": [1.2, 0.0]}},
        "https://example.test/rainfall",
    )


def _count(db_path, table, reach_id):
    with connect_database(db_path=db_path) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE reach_id = ?", (reach_id,)).fetchone()[0]


def test_rerunning_flow_ingestion_does_not_duplicate_observations(tmp_path):
    db_path, reach_id = _make_reach(tmp_path)

    collect_usgs_flow(reach_id, "01234567", "2020-01-01T00:00:00Z", "2020-01-01T02:00:00Z",
                      db_path=db_path, requester=_flow_requester)
    first = _count(db_path, "flow_observations", reach_id)

    # Re-run the identical ingestion WITHOUT --replace.
    collect_usgs_flow(reach_id, "01234567", "2020-01-01T00:00:00Z", "2020-01-01T02:00:00Z",
                      db_path=db_path, requester=_flow_requester)
    second = _count(db_path, "flow_observations", reach_id)

    assert first == 2
    assert second == 2, "re-running the same flow ingestion must not create duplicate observations"


def test_rerunning_rainfall_ingestion_does_not_duplicate_observations(tmp_path):
    db_path, reach_id = _make_reach(tmp_path)

    collect_rainfall(reach_id, "2020-01-01", "2020-01-01", db_path=db_path, requester=_rainfall_requester)
    collect_rainfall(reach_id, "2020-01-01", "2020-01-01", db_path=db_path, requester=_rainfall_requester)

    assert _count(db_path, "rainfall_observations", reach_id) == 2


def test_rerunning_elevation_ingestion_does_not_duplicate_samples(tmp_path):
    db_path, reach_id = _make_reach(tmp_path)

    collect_elevations(reach_id, db_path=db_path, requester=_elevation_requester)
    collect_elevations(reach_id, db_path=db_path, requester=_elevation_requester)

    assert _count(db_path, "elevation_samples", reach_id) == 3
    assert _count(db_path, "slope_samples", reach_id) == 2


def test_redact_url_scrubs_sensitive_query_parameters():
    redacted = redact_url("https://api.example.test/data?api_key=SECRET&token=TOP&x=1")
    assert "SECRET" not in redacted
    assert "TOP" not in redacted
    assert "x=1" in redacted
    assert "api_key=REDACTED" in redacted


def test_request_json_redacts_credentials_in_raised_exception(monkeypatch):
    def failing_urlopen(request, timeout=None):
        raise URLError(f"cannot reach {request.full_url}")

    # No retry delay: avoid slowing the test.
    monkeypatch.setattr(common, "urlopen", failing_urlopen)

    with pytest.raises(Exception) as excinfo:
        request_json(
            "https://api.example.test/data",
            {"api_key": "SUPERSECRET", "site": "01234567"},
            max_retries=0,
        )

    message = str(excinfo.value)
    assert "SUPERSECRET" not in message
    assert "REDACTED" in message


def test_failed_provider_call_preserves_previously_stored_data(tmp_path):
    db_path, reach_id = _make_reach(tmp_path)
    collect_rainfall(reach_id, "2020-01-01", "2020-01-01", db_path=db_path, requester=_rainfall_requester)

    def exploding_requester(url, params=None):
        raise RuntimeError("provider is down")

    with pytest.raises(RuntimeError):
        collect_elevations(reach_id, db_path=db_path, requester=exploding_requester)

    # The failed elevation fetch must not have written partial rows, and the
    # earlier rainfall data must survive.
    assert _count(db_path, "elevation_samples", reach_id) == 0
    assert _count(db_path, "rainfall_observations", reach_id) == 2

    # The database must remain writable after the failure (no dangling lock).
    collect_elevations(reach_id, db_path=db_path, requester=_elevation_requester)
    assert _count(db_path, "elevation_samples", reach_id) == 3


def test_request_json_retries_5xx_then_succeeds(monkeypatch):
    attempts = {"n": 0}

    def flaky_urlopen(request, timeout=None):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise HTTPError(request.full_url, 503, "busy", {}, None)
        return _FakeResponse({"ok": True}, "https://example.test/final")

    monkeypatch.setattr(common, "urlopen", flaky_urlopen)
    payload, url = request_json("https://example.test/x", {"a": "1"}, retry_delay=0)

    assert payload == {"ok": True}
    assert attempts["n"] == 3, "a 5xx response should be retried"


def test_request_json_does_not_retry_4xx(monkeypatch):
    attempts = {"n": 0}

    def urlopen_404(request, timeout=None):
        attempts["n"] += 1
        raise HTTPError(request.full_url, 404, "not found", {}, None)

    monkeypatch.setattr(common, "urlopen", urlopen_404)
    with pytest.raises(common.ProviderRequestError):
        request_json("https://example.test/x", {"a": "1"}, retry_delay=0)

    assert attempts["n"] == 1, "a 4xx response must not be retried"


def test_columbia_markers_rerun_dedupes_with_derived_float_stations(tmp_path):
    # The real curated markers file has NO station_m column, so stations are
    # derived from haversine (float values like 24107.839...). Re-running must
    # still dedupe against those derived floats.
    markers_csv = REPO_ROOT / "real_world_rivers" / "columbia_hanford_markers.csv"
    db_path = tmp_path / "river.sqlite"
    reach_id = create_reach("Columbia", "Hanford", markers_csv, country="US", db_path=db_path)

    def elevation_requester(url, params):
        count = len(params["latitude"].split(","))
        return {"elevation": [100.0 - 10.0 * i for i in range(count)]}, "https://example.test/elev"

    collect_elevations(reach_id, db_path=db_path, requester=elevation_requester)
    collect_elevations(reach_id, db_path=db_path, requester=elevation_requester)

    with connect_database(db_path=db_path) as conn:
        elev = conn.execute("SELECT COUNT(*) FROM elevation_samples WHERE reach_id = ?", (reach_id,)).fetchone()[0]
        slope = conn.execute("SELECT COUNT(*) FROM slope_samples WHERE reach_id = ?", (reach_id,)).fetchone()[0]
    assert elev == 5, "derived-float stations must dedupe on re-run"
    assert slope == 4


def test_add_source_is_idempotent_on_natural_key(tmp_path):
    db_path, _ = _make_reach(tmp_path)
    with connect_database(db_path=db_path) as conn:
        first = add_source(conn, "USGS gauge X", "stream gauge", url="https://example.test/x",
                           citation="cite")
        second = add_source(conn, "USGS gauge X", "stream gauge", url="https://example.test/x",
                            citation="cite")
        assert first == second, "identical source metadata must reuse the same source row"
