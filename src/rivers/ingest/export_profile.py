import csv
import json
import statistics
import warnings
from pathlib import Path

from .common import connect_database, get_markers, get_reach


def _interval_value(conn, table, column, reach_id, station, is_last):
    # For interior stations, end_station_m > station selects the interval whose
    # end is strictly past the station — meaning a station at an interval boundary
    # (e.g., station=1000 with intervals [0,1000] and [1000,2000]) uses the
    # interval that *starts* at that boundary, not the one that ends there.
    # The final station uses >= so its own interval end is inclusive.
    end_operator = ">=" if is_last else ">"
    row = conn.execute(
        f"""
        SELECT {column} AS value FROM {table}
        WHERE reach_id = ?
          AND start_station_m <= ?
          AND end_station_m {end_operator} ?
        ORDER BY id DESC LIMIT 1
        """,
        (reach_id, station, station),
    ).fetchone()
    if row is None:
        raise ValueError(f"No {column} sample covers station {station:g} m")
    return float(row["value"])


def _rainfall_rate(conn, reach_id, start, end):
    if start is None and end is None:
        return None
    if not start or not end:
        raise ValueError("Rainfall export requires both start and end timestamps")
    row = conn.execute(
        """
        SELECT COUNT(*) AS obs_count, AVG(precipitation_mm / 1000.0 / interval_min) AS rate
        FROM rainfall_observations
        WHERE reach_id = ? AND observed_at >= ? AND observed_at <= ?
        """,
        (reach_id, start, end),
    ).fetchone()
    if row["rate"] is None:
        raise ValueError("No rainfall observations exist in the requested interval")
    return float(row["rate"]), int(row["obs_count"])


def _recommended_inflow(conn, reach_id, start=None, end=None):
    upstream_marker = conn.execute(
        """
        SELECT observations.marker_id
        FROM flow_observations AS observations
        JOIN reach_markers AS markers ON markers.id = observations.marker_id
        WHERE observations.reach_id = ?
        ORDER BY markers.marker_order
        LIMIT 1
        """,
        (reach_id,),
    ).fetchone()
    if upstream_marker is None:
        return None
    clauses = ["reach_id = ?", "marker_id = ?"]
    values = [reach_id, upstream_marker["marker_id"]]
    if start:
        clauses.append("observed_at >= ?")
        values.append(start)
    if end:
        clauses.append("observed_at <= ?")
        values.append(end)
    flows = [
        float(row["discharge_m3_per_min"])
        for row in conn.execute(
            f"SELECT discharge_m3_per_min FROM flow_observations WHERE {' AND '.join(clauses)}",
            values,
        )
    ]
    width_row = conn.execute(
        """
        SELECT width_m FROM channel_geometry_samples
        WHERE reach_id = ? AND width_m IS NOT NULL
        ORDER BY ABS(COALESCE(station_m, 0.0)), id DESC LIMIT 1
        """,
        (reach_id,),
    ).fetchone()
    if not flows or width_row is None:
        return None
    discharge = statistics.median(flows)
    width = float(width_row["width_m"])
    return {
        "median_discharge_m3_per_min": discharge,
        "upstream_width_m": width,
        "left_inflow_flux_m2_per_min": discharge / width,
        "observation_count": len(flows),
    }


def export_profile(
    reach_id,
    output_path,
    *,
    db_path=None,
    minimum_slope=1e-6,
    initial_depth_m=None,
    rainfall_start=None,
    rainfall_end=None,
    flow_start=None,
    flow_end=None,
):
    if minimum_slope <= 0:
        raise ValueError("minimum_slope must be positive")
    options = {} if db_path is None else {"db_path": db_path}
    with connect_database(**options) as conn:
        reach = get_reach(conn, reach_id)
        markers = get_markers(conn, reach_id)
        rainfall_result = _rainfall_rate(conn, reach_id, rainfall_start, rainfall_end)
        rainfall_rate, rainfall_obs_count = rainfall_result if rainfall_result is not None else (None, None)
        rows = []
        adjusted_slopes = 0
        for index, marker in enumerate(markers):
            raw_slope = _interval_value(
                conn, "slope_samples", "slope", reach_id, marker["station_m"], index == len(markers) - 1
            )
            slope = max(raw_slope, minimum_slope)
            adjusted_slopes += slope != raw_slope
            row = {
                "station_m": float(marker["station_m"]),
                "slope": slope,
                "manning_n": _interval_value(
                    conn,
                    "roughness_samples",
                    "manning_n",
                    reach_id,
                    marker["station_m"],
                    index == len(markers) - 1,
                ),
                "label": marker["label"] or f"marker-{marker['marker_order']}",
            }
            if initial_depth_m is not None:
                row["initial_depth_m"] = float(initial_depth_m)
            if rainfall_rate is not None:
                row["rainfall_rate_m_per_min"] = rainfall_rate
            rows.append(row)
        recommendation = _recommended_inflow(conn, reach_id, flow_start, flow_end)

    if adjusted_slopes:
        warnings.warn(
            f"{adjusted_slopes} of {len(markers)} slope value(s) were below minimum_slope "
            f"({minimum_slope:g}) and were raised to that floor. Check the slope data or "
            "lower minimum_slope if the adjustment is unintended.",
            stacklevel=2,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        output_path.write_text(json.dumps({"segments": rows}, indent=2) + "\n", encoding="utf-8")
    elif output_path.suffix.lower() == ".csv":
        fields = ["station_m", "slope", "manning_n"]
        if initial_depth_m is not None:
            fields.append("initial_depth_m")
        if rainfall_rate is not None:
            fields.append("rainfall_rate_m_per_min")
        fields.append("label")
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        raise ValueError("Profile output must end in .csv or .json")

    metadata = {
        "reach_id": reach_id,
        "reach_name": reach["name"],
        "profile_path": str(output_path),
        "segments": len(rows),
        "minimum_slope": minimum_slope,
        "slope_values_adjusted": adjusted_slopes,
        "rainfall_rate_m_per_min": rainfall_rate,
        "rainfall_observation_count": rainfall_obs_count,
        "recommended_upstream_inflow": recommendation,
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
