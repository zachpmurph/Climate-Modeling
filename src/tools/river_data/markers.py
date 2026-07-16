from pathlib import Path

from .common import add_source, connect_database, read_structured_rows, stations_from_coordinates


def load_marker_rows(path):
    raw_rows = read_structured_rows(path)
    if len(raw_rows) < 2:
        raise ValueError("A reach requires at least two markers")

    rows = []
    for index, row in enumerate(raw_rows):
        try:
            lat = float(row["lat"])
            lon = float(row["lon"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Marker {index} requires numeric lat and lon") from exc
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError(f"Marker {index} has invalid WGS84 coordinates")
        rows.append(
            {
                "lat": lat,
                "lon": lon,
                "station_m": row.get("station_m"),
                "label": str(row.get("label", "")),
                "notes": row.get("notes"),
            }
        )

    supplied = [row["station_m"] not in (None, "") for row in rows]
    if any(supplied) and not all(supplied):
        raise ValueError("Either provide station_m for every marker or omit it for every marker")
    stations = [float(row["station_m"]) for row in rows] if all(supplied) else stations_from_coordinates(rows)
    if any(right <= left for left, right in zip(stations, stations[1:])):
        raise ValueError("Marker stations must be strictly increasing")
    for row, station in zip(rows, stations):
        row["station_m"] = station
    return rows


def create_reach(
    river_name,
    reach_name,
    marker_path,
    *,
    region=None,
    country=None,
    notes=None,
    db_path=None,
    replace=False,
):
    rows = load_marker_rows(marker_path)
    options = {} if db_path is None else {"db_path": db_path}
    with connect_database(**options) as conn:
        river = conn.execute(
            "SELECT id FROM rivers WHERE name = ? AND region IS ? AND country IS ?",
            (river_name, region, country),
        ).fetchone()
        if river is None:
            river_id = conn.execute(
                "INSERT INTO rivers (name, region, country) VALUES (?, ?, ?)",
                (river_name, region, country),
            ).lastrowid
        else:
            river_id = river["id"]
        existing = conn.execute(
            "SELECT id FROM reaches WHERE river_id = ? AND name = ?",
            (river_id, reach_name),
        ).fetchone()
        if existing and not replace:
            raise ValueError(f"Reach already exists with id {existing['id']}; pass --replace to overwrite it")
        if existing:
            conn.execute("DELETE FROM reaches WHERE id = ?", (existing["id"],))

        cursor = conn.execute(
            """
            INSERT INTO reaches
                (river_id, name, start_lat, start_lon, end_lat, end_lon, length_m, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                river_id,
                reach_name,
                rows[0]["lat"],
                rows[0]["lon"],
                rows[-1]["lat"],
                rows[-1]["lon"],
                rows[-1]["station_m"] - rows[0]["station_m"],
                notes,
            ),
        )
        reach_id = cursor.lastrowid
        source_id = add_source(
            conn,
            Path(marker_path).name,
            "reviewed centerline",
            url=str(Path(marker_path).resolve()),
            notes="Marker order is upstream to downstream.",
        )
        conn.executemany(
            """
            INSERT INTO reach_markers
                (reach_id, marker_order, lat, lon, station_m, label, source_id, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    reach_id,
                    index,
                    row["lat"],
                    row["lon"],
                    row["station_m"],
                    row["label"],
                    source_id,
                    row["notes"],
                )
                for index, row in enumerate(rows)
            ],
        )
    return reach_id
