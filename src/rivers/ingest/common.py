import json
import math
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from .database import DEFAULT_DB_PATH, initialize_database


USER_AGENT = "Climate-Modeling river-data collector/1.0"
EARTH_RADIUS_M = 6_371_008.8

# Query-string keys whose values are credentials and must never appear in
# stored URLs, logs, metadata, or exception messages.
SENSITIVE_QUERY_KEYS = frozenset(
    {"api_key", "apikey", "key", "token", "access_token", "auth", "password", "secret"}
)


class ProviderRequestError(RuntimeError):
    """A provider HTTP request failed. Its message is credential-redacted."""


def redact_url(url):
    """Return ``url`` with the value of any sensitive query parameter replaced.

    Redaction covers credentials passed in the query string of provider URLs so
    they cannot leak through ``data_sources.url``, logs, metadata sidecars, or
    exception text. URLs without a query string are returned unchanged.
    """
    if not url:
        return url
    parts = urlsplit(url)
    if not parts.query:
        return url
    query = [
        (key, "REDACTED" if key.lower() in SENSITIVE_QUERY_KEYS else value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
    ]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


@contextmanager
def connect_database(db_path=DEFAULT_DB_PATH):
    """Yield a configured SQLite connection, committing on success and always closing.

    Used as ``with connect_database(...) as conn:``. The surrounding transaction
    commits on a clean exit and rolls back on exception, and the connection is
    closed either way — so a failed provider call leaves no partial writes and
    no dangling file handle (which on Windows would lock the database file).
    """
    initialize_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def request_json(base_url, params=None, timeout=30, max_retries=3, retry_delay=1.0):
    url = f"{base_url}?{urlencode(params)}" if params else base_url
    safe_url = redact_url(url)
    request = Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response), redact_url(response.geturl())
        except HTTPError as exc:
            if exc.code < 500:
                # 4xx errors won't be fixed by retrying. Raise a redacted error
                # so the credential in `url` never surfaces in the traceback.
                raise ProviderRequestError(f"HTTP {exc.code} from {safe_url}") from None
            last_error = exc
        except (URLError, OSError) as exc:
            last_error = exc
        if attempt < max_retries:
            time.sleep(retry_delay * (2 ** attempt))
    raise ProviderRequestError(
        f"Provider request to {safe_url} failed after {max_retries + 1} attempt(s): "
        f"{type(last_error).__name__}"
    ) from None


def add_source(conn, name, source_type, url=None, citation=None, notes=None):
    """Insert a provenance source, reusing an existing row with identical metadata.

    ``add_source`` is idempotent on the natural key (name, source_type, url,
    citation). Re-running an ingestion therefore does not mint a fresh
    ``source_id`` every time, which is what keeps the observation-level UNIQUE
    constraints (which do NOT include source_id) effective across runs. The
    match uses SQL ``IS`` so NULL url/citation values compare correctly.
    """
    existing = conn.execute(
        """
        SELECT id FROM data_sources
        WHERE name = ? AND source_type = ? AND url IS ? AND citation IS ?
        """,
        (name, source_type, url, citation),
    ).fetchone()
    if existing is not None:
        return existing["id"]
    cursor = conn.execute(
        """
        INSERT INTO data_sources (name, source_type, url, citation, accessed_at, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (name, source_type, url, citation, utc_now(), notes),
    )
    return cursor.lastrowid


def get_reach(conn, reach_id):
    reach = conn.execute("SELECT * FROM reaches WHERE id = ?", (reach_id,)).fetchone()
    if reach is None:
        raise ValueError(f"Reach {reach_id} does not exist")
    return reach


def get_markers(conn, reach_id):
    get_reach(conn, reach_id)
    rows = conn.execute(
        "SELECT * FROM reach_markers WHERE reach_id = ? ORDER BY marker_order",
        (reach_id,),
    ).fetchall()
    if len(rows) < 2:
        raise ValueError(f"Reach {reach_id} needs at least two ordered markers")
    return rows


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def stations_from_coordinates(rows):
    stations = [0.0]
    for previous, current in zip(rows, rows[1:]):
        distance = haversine_m(previous["lat"], previous["lon"], current["lat"], current["lon"])
        stations.append(stations[-1] + distance)
    return stations


def read_structured_rows(path):
    import csv

    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))

    data = json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() in {".geojson", ".json"} and isinstance(data, dict):
        if data.get("type") == "FeatureCollection":
            features = data.get("features", [])
            if len(features) != 1:
                raise ValueError("Marker GeoJSON must contain exactly one LineString feature")
            data = features[0]
        if data.get("type") == "Feature":
            data = data.get("geometry")
        if isinstance(data, dict) and data.get("type") == "LineString":
            return [
                {"lon": coordinate[0], "lat": coordinate[1], "label": f"marker-{index}"}
                for index, coordinate in enumerate(data["coordinates"])
            ]
        if isinstance(data, dict):
            data = data.get("markers", data.get("rows"))
    if not isinstance(data, list):
        raise ValueError("Expected CSV rows, a JSON list, or a GeoJSON LineString")
    return data
