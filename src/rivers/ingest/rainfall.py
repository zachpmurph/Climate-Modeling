import warnings

from .common import add_source, connect_database, get_markers, request_json


ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
RAINFALL_CITATION = "Open-Meteo Historical Weather API; hourly precipitation reanalysis."


def parse_hourly_precipitation(payload):
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    values = hourly.get("precipitation", [])
    if len(times) != len(values) or not times:
        raise ValueError("Rainfall provider returned invalid hourly precipitation")
    return [
        {"observed_at": observed_at, "precipitation_mm": float(value), "interval_min": 60.0}
        for observed_at, value in zip(times, values)
        if value is not None
    ]


def fetch_rainfall(lat, lon, start_date, end_date, *, requester=request_json):
    payload, url = requester(
        ARCHIVE_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "precipitation",
            "precipitation_unit": "mm",
            "timezone": "GMT",
        },
    )
    return parse_hourly_precipitation(payload), url


def collect_rainfall(
    reach_id,
    start_date,
    end_date,
    *,
    marker_order=None,
    db_path=None,
    replace=False,
    requester=request_json,
):
    options = {} if db_path is None else {"db_path": db_path}
    with connect_database(**options) as conn:
        markers = get_markers(conn, reach_id)
        if marker_order is None:
            marker = markers[len(markers) // 2]
            if len(markers) > 2:
                warnings.warn(
                    f"collect_rainfall: sampling rainfall at middle marker only "
                    f"(order={marker['marker_order']}, station={marker['station_m']:.0f} m) "
                    f"for a {len(markers)}-marker reach. For spatially varying rainfall, "
                    "call collect_rainfall with an explicit marker_order for each marker.",
                    stacklevel=2,
                )
        else:
            try:
                marker = next(row for row in markers if row["marker_order"] == marker_order)
            except StopIteration as exc:
                raise ValueError(f"Marker order {marker_order} does not exist") from exc
        observations, url = fetch_rainfall(
            marker["lat"], marker["lon"], start_date, end_date, requester=requester
        )
        if replace:
            conn.execute(
                """
                DELETE FROM rainfall_observations
                WHERE reach_id = ? AND marker_id = ?
                  AND substr(observed_at, 1, 10) BETWEEN ? AND ?
                """,
                (reach_id, marker["id"], start_date, end_date),
            )
        source_id = add_source(
            conn,
            "Open-Meteo hourly precipitation",
            "weather reanalysis",
            url=url,
            citation=RAINFALL_CITATION,
            notes=(
                f"Sampled at marker {marker['marker_order']} ({marker['lat']}, {marker['lon']}); "
                "hourly precipitation is a preceding-hour sum."
            ),
        )
        conn.executemany(
            """
            INSERT INTO rainfall_observations
                (reach_id, marker_id, observed_at, precipitation_mm, interval_min, source_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    reach_id,
                    marker["id"],
                    item["observed_at"],
                    item["precipitation_mm"],
                    item["interval_min"],
                    source_id,
                )
                for item in observations
            ],
        )
    return {"observation_count": len(observations), "source_id": source_id}
