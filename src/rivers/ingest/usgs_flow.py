import warnings
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .common import add_source, connect_database, get_markers, request_json


CONTINUOUS_URL = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous/items"
USGS_CITATION = "U.S. Geological Survey Water Data APIs, continuous values."
DISCHARGE_PARAMETER = "00060"
CFS_TO_M3_PER_MIN = 0.028316846592 * 60.0


def _redact_api_key(url):
    parts = urlsplit(url)
    query = [
        (key, "REDACTED" if key.lower() == "api_key" else value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
    ]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def discharge_to_m3_per_min(value, unit):
    # USGS API sometimes delivers "ft³/s" with the Unicode superscript-3 (U+00B3)
    # corrupted to "?" due to encoding issues in older responses. Detect and recover
    # before normalizing, so the substitution is visible rather than silent.
    normalized = unit.lower().replace(" ", "").replace("^", "")
    if "?" in normalized:
        warnings.warn(
            f"USGS unit string {unit!r} contains '?' — likely a mangled superscript (U+00B3 → ?). "
            "Treating '?' as '3' to recover the unit.",
            stacklevel=2,
        )
        normalized = normalized.replace("?", "3")
    if normalized in {"ft3/s", "cfs", "cubicfeetpersecond"}:
        return float(value) * CFS_TO_M3_PER_MIN
    if normalized in {"m3/s", "cubicmeterspersecond"}:
        return float(value) * 60.0
    if normalized in {"m3/min", "cubicmetersperminute"}:
        return float(value)
    raise ValueError(f"Unsupported USGS discharge unit: {unit}")


def parse_flow_features(payload):
    observations = []
    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        if properties.get("parameter_code") != DISCHARGE_PARAMETER:
            continue
        value = properties.get("value")
        if value is None:
            continue
        unit = properties.get("unit_of_measure", "ft3/s")
        observations.append(
            {
                "observed_at": properties["time"],
                "value": float(value),
                "unit": unit,
                "discharge_m3_per_min": discharge_to_m3_per_min(value, unit),
                "approval_status": properties.get("approval_status"),
            }
        )
    return observations


def fetch_usgs_flow(site_id, start, end, *, api_key=None, requester=request_json):
    monitoring_location_id = site_id if "-" in site_id else f"USGS-{site_id}"
    params = {
        "f": "json",
        "monitoring_location_id": monitoring_location_id,
        "parameter_code": DISCHARGE_PARAMETER,
        "datetime": f"{start}/{end}",
        "limit": 50000,
    }
    if api_key:
        params["api_key"] = api_key
    payload, url = requester(CONTINUOUS_URL, params)
    observations = parse_flow_features(payload)
    next_url = next(
        (link.get("href") for link in payload.get("links", []) if link.get("rel") == "next"),
        None,
    )
    while next_url:
        payload, _ = requester(next_url)
        observations.extend(parse_flow_features(payload))
        next_url = next(
            (link.get("href") for link in payload.get("links", []) if link.get("rel") == "next"),
            None,
        )
    if not observations:
        raise ValueError(f"No discharge observations returned for {monitoring_location_id}")
    return observations, _redact_api_key(url), monitoring_location_id


def collect_usgs_flow(
    reach_id,
    site_id,
    start,
    end,
    *,
    marker_order=0,
    api_key=None,
    db_path=None,
    replace=False,
    requester=request_json,
):
    observations, url, gauge_id = fetch_usgs_flow(
        site_id, start, end, api_key=api_key, requester=requester
    )
    options = {} if db_path is None else {"db_path": db_path}
    with connect_database(**options) as conn:
        markers = get_markers(conn, reach_id)
        try:
            marker = next(row for row in markers if row["marker_order"] == marker_order)
        except StopIteration as exc:
            raise ValueError(f"Marker order {marker_order} does not exist") from exc
        if replace:
            conn.execute(
                "DELETE FROM flow_observations WHERE reach_id = ? AND gauge_id = ?",
                (reach_id, gauge_id),
            )
        source_id = add_source(
            conn,
            f"USGS gauge {gauge_id}",
            "stream gauge",
            url=url,
            citation=USGS_CITATION,
            notes="Parameter 00060 continuous discharge. Approval status is retained in notes.",
        )
        conn.executemany(
            """
            INSERT INTO flow_observations
                (reach_id, marker_id, gauge_id, observed_at, discharge_m3_per_min,
                 discharge_original_value, discharge_original_unit, source_id, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    reach_id,
                    marker["id"],
                    gauge_id,
                    item["observed_at"],
                    item["discharge_m3_per_min"],
                    item["value"],
                    item["unit"],
                    source_id,
                    f"approval_status={item['approval_status']}",
                )
                for item in observations
            ],
        )
    return {"observation_count": len(observations), "source_id": source_id, "gauge_id": gauge_id}
