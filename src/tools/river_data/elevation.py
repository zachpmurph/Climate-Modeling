from .common import add_source, connect_database, get_markers, request_json


ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
ELEVATION_CITATION = "Open-Meteo Elevation API; Copernicus DEM 2021 GLO-90."
# Open-Meteo elevation API accepts at most 100 coordinate pairs per request.
# https://open-meteo.com/en/docs/elevation-api
ELEVATION_BATCH_SIZE = 100


def parse_elevations(payload, expected_count):
    values = payload.get("elevation")
    if not isinstance(values, list) or len(values) != expected_count:
        raise ValueError("Elevation provider returned an unexpected number of values")
    if any(value is None for value in values):
        raise ValueError("Elevation provider returned a missing elevation")
    return [float(value) for value in values]


def fetch_elevations(markers, requester=request_json):
    elevations = []
    urls = []
    for start in range(0, len(markers), ELEVATION_BATCH_SIZE):
        batch = markers[start : start + ELEVATION_BATCH_SIZE]
        payload, url = requester(
            ELEVATION_URL,
            {
                "latitude": ",".join(str(row["lat"]) for row in batch),
                "longitude": ",".join(str(row["lon"]) for row in batch),
            },
        )
        elevations.extend(parse_elevations(payload, len(batch)))
        urls.append(url)
    return elevations, urls


def collect_elevations(reach_id, *, db_path=None, replace=False, requester=request_json):
    options = {} if db_path is None else {"db_path": db_path}
    with connect_database(**options) as conn:
        markers = get_markers(conn, reach_id)
        elevations, urls = fetch_elevations(markers, requester=requester)
        if replace:
            conn.execute("DELETE FROM slope_samples WHERE reach_id = ?", (reach_id,))
            conn.execute("DELETE FROM elevation_samples WHERE reach_id = ?", (reach_id,))
        source_id = add_source(
            conn,
            "Open-Meteo Copernicus DEM elevation",
            "digital elevation model",
            url=" | ".join(urls),
            citation=ELEVATION_CITATION,
            notes="GLO-90 terrain elevation sampled at reach markers; nominal resolution 90 m.",
        )
        conn.executemany(
            """
            INSERT INTO elevation_samples
                (reach_id, marker_id, station_m, elevation_m, method, source_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (reach_id, marker["id"], marker["station_m"], elevation, "GLO-90 point sample", source_id)
                for marker, elevation in zip(markers, elevations)
            ],
        )
        slopes = []
        for left, right, elevation_left, elevation_right in zip(
            markers, markers[1:], elevations, elevations[1:]
        ):
            distance = right["station_m"] - left["station_m"]
            slope = (elevation_left - elevation_right) / distance
            slopes.append((left["station_m"], right["station_m"], slope, elevation_left, elevation_right))
        conn.executemany(
            """
            INSERT INTO slope_samples
                (reach_id, start_station_m, end_station_m, slope,
                 elevation_start_m, elevation_end_m, method, source_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (reach_id, start, end, slope, elev_start, elev_end, "DEM endpoint difference", source_id)
                for start, end, slope, elev_start, elev_end in slopes
            ],
        )
    return {"elevation_count": len(elevations), "slope_count": len(slopes), "source_id": source_id}
