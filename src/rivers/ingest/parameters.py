from pathlib import Path

from .common import add_source, connect_database, get_markers, read_structured_rows


def _optional_float(row, name):
    value = row.get(name)
    return None if value in (None, "") else float(value)


def import_roughness(reach_id, path, *, db_path=None, replace=False):
    rows = read_structured_rows(path)
    values = []
    for index, row in enumerate(rows):
        start = float(row["start_station_m"])
        end = float(row["end_station_m"])
        manning_n = float(row["manning_n"])
        if end <= start or manning_n <= 0:
            raise ValueError(f"Invalid roughness row {index}: require end > start and manning_n > 0")
        values.append((start, end, manning_n, row.get("method", "reviewed input"), row.get("notes")))

    options = {} if db_path is None else {"db_path": db_path}
    with connect_database(**options) as conn:
        get_markers(conn, reach_id)
        if replace:
            conn.execute("DELETE FROM roughness_samples WHERE reach_id = ?", (reach_id,))
        source_id = add_source(
            conn,
            Path(path).name,
            "reviewed roughness input",
            url=str(Path(path).resolve()),
            notes="Manning n values must be reviewed against channel material and calibration evidence.",
        )
        conn.executemany(
            """
            INSERT INTO roughness_samples
                (reach_id, start_station_m, end_station_m, manning_n, method, source_id, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [(reach_id, start, end, value, method, source_id, notes) for start, end, value, method, notes in values],
        )
    return {"sample_count": len(values), "source_id": source_id}


def import_geometry(reach_id, path, *, db_path=None, replace=False):
    rows = read_structured_rows(path)
    values = []
    for index, row in enumerate(rows):
        station = float(row["station_m"])
        width = _optional_float(row, "width_m")
        depth = _optional_float(row, "bankfull_depth_m")
        if width is None and depth is None:
            raise ValueError(f"Geometry row {index} needs width_m or bankfull_depth_m")
        if (width is not None and width <= 0) or (depth is not None and depth <= 0):
            raise ValueError(f"Geometry row {index} values must be positive")
        values.append((station, width, depth, row.get("method", "reviewed input"), row.get("notes")))

    options = {} if db_path is None else {"db_path": db_path}
    with connect_database(**options) as conn:
        markers = get_markers(conn, reach_id)
        if replace:
            conn.execute("DELETE FROM channel_geometry_samples WHERE reach_id = ?", (reach_id,))
        source_id = add_source(
            conn,
            Path(path).name,
            "reviewed channel geometry",
            url=str(Path(path).resolve()),
        )
        for station, width, depth, method, notes in values:
            marker = min(markers, key=lambda row: abs(row["station_m"] - station))
            conn.execute(
                """
                INSERT INTO channel_geometry_samples
                    (reach_id, marker_id, station_m, width_m, bankfull_depth_m,
                     method, source_id, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (reach_id, marker["id"], station, width, depth, method, source_id, notes),
            )
    return {"sample_count": len(values), "source_id": source_id}
