import csv
import io
import json
import os
import statistics
import warnings
from pathlib import Path

from .common import connect_database, get_markers, get_reach
from .validation import (
    ProfileValidationError,
    validate_profile,
    validate_temporal_coverage,
)


def _interval_value(conn, table, column, reach_id, station, is_last):
    # For interior stations, end_station_m > station selects the interval whose
    # end is strictly past the station — meaning a station at an interval boundary
    # (e.g., station=1000 with intervals [0,1000] and [1000,2000]) uses the
    # interval that *starts* at that boundary, not the one that ends there.
    # The final station uses >= so its own interval end is inclusive.
    end_operator = ">=" if is_last else ">"
    row = conn.execute(
        f"""
        SELECT {column} AS value, classification FROM {table}
        WHERE reach_id = ?
          AND start_station_m <= ?
          AND end_station_m {end_operator} ?
        ORDER BY id DESC LIMIT 1
        """,
        (reach_id, station, station),
    ).fetchone()
    if row is None:
        raise ValueError(f"No {column} sample covers station {station:g} m")
    return float(row["value"]), row["classification"]


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


def _rainfall_times(conn, reach_id, start, end):
    return [
        r["observed_at"]
        for r in conn.execute(
            """
            SELECT observed_at FROM rainfall_observations
            WHERE reach_id = ? AND observed_at >= ? AND observed_at <= ?
            ORDER BY observed_at
            """,
            (reach_id, start, end),
        )
    ]


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
        "classification": "derived",
    }


def _reach_sources(conn, reach_id):
    """Return only the data_sources actually referenced by this reach's rows.

    Provenance must be traceable and reach-specific: a shared database can hold
    many reaches, so listing every source row would let one reach's sidecar
    claim provenance it does not have.
    """
    source_id_rows = conn.execute(
        """
        SELECT DISTINCT source_id FROM (
            SELECT source_id FROM reach_markers            WHERE reach_id = ?
            UNION SELECT source_id FROM elevation_samples  WHERE reach_id = ?
            UNION SELECT source_id FROM slope_samples      WHERE reach_id = ?
            UNION SELECT source_id FROM roughness_samples  WHERE reach_id = ?
            UNION SELECT source_id FROM channel_geometry_samples WHERE reach_id = ?
            UNION SELECT source_id FROM flow_observations  WHERE reach_id = ?
            UNION SELECT source_id FROM rainfall_observations    WHERE reach_id = ?
        )
        WHERE source_id IS NOT NULL
        """,
        (reach_id,) * 7,
    ).fetchall()
    ids = [row["source_id"] for row in source_id_rows]
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    return [
        dict(row)
        for row in conn.execute(
            f"SELECT id, name, source_type, url, citation, accessed_at "
            f"FROM data_sources WHERE id IN ({placeholders}) ORDER BY id",
            ids,
        )
    ]


def _atomic_write_text(path, text):
    """Write ``text`` to ``path`` atomically via a temp file + os.replace."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)  # atomic for same-directory paths on POSIX and Windows


def _render_csv(rows, fields):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


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
        coordinates = []
        adjustments = []
        for index, marker in enumerate(markers):
            is_last = index == len(markers) - 1
            raw_slope, slope_class = _interval_value(
                conn, "slope_samples", "slope", reach_id, marker["station_m"], is_last
            )
            slope = max(raw_slope, minimum_slope)
            if slope != raw_slope:
                slope_class = "fallback"
                adjustments.append({
                    "rule": "minimum_slope_floor",
                    "field": "slope",
                    "station_m": float(marker["station_m"]),
                    "original": raw_slope,
                    "adjusted": slope,
                    "minimum_slope": minimum_slope,
                })
            manning_n, manning_class = _interval_value(
                conn, "roughness_samples", "manning_n", reach_id, marker["station_m"], is_last
            )
            classification = {
                "station_m": "observed",
                "slope": slope_class or "derived",
                "manning_n": manning_class or "estimated",
            }
            row = {
                "station_m": float(marker["station_m"]),
                "slope": slope,
                "manning_n": manning_n,
                "label": marker["label"] or f"marker-{marker['marker_order']}",
                "classification": classification,
            }
            if initial_depth_m is not None:
                row["initial_depth_m"] = float(initial_depth_m)
                classification["initial_depth_m"] = "fallback"
            if rainfall_rate is not None:
                row["rainfall_rate_m_per_min"] = rainfall_rate
                classification["rainfall_rate_m_per_min"] = "derived"
            rows.append(row)
            coordinates.append((marker["lat"], marker["lon"]))

        adjusted_slopes = len(adjustments)
        recommendation = _recommended_inflow(conn, reach_id, flow_start, flow_end)
        sources = _reach_sources(conn, reach_id)

        # Validate BEFORE writing anything: errors block export so we never leave
        # a misleading partial artifact on disk.
        report = validate_profile(rows, coordinates=coordinates, adjustments=adjustments)
        if rainfall_rate is not None:
            times = _rainfall_times(conn, reach_id, rainfall_start, rainfall_end)
            report.extend(
                validate_temporal_coverage(times, rainfall_start, rainfall_end, label="rainfall")
            )

    if report.has_errors:
        raise ProfileValidationError(report)

    if adjusted_slopes:
        warnings.warn(
            f"{adjusted_slopes} of {len(markers)} slope value(s) were below minimum_slope "
            f"({minimum_slope:g}) and were raised to that floor. Check the slope data or "
            "lower minimum_slope if the adjustment is unintended.",
            stacklevel=2,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    # Build the profile payload (model-facing columns only — no new required
    # columns; classification and provenance live in the metadata sidecar).
    if suffix == ".json":
        segments = [{k: v for k, v in row.items() if k != "classification"} for row in rows]
        profile_text = json.dumps({"segments": segments}, indent=2) + "\n"
    elif suffix == ".csv":
        fields = ["station_m", "slope", "manning_n"]
        if initial_depth_m is not None:
            fields.append("initial_depth_m")
        if rainfall_rate is not None:
            fields.append("rainfall_rate_m_per_min")
        fields.append("label")
        profile_text = _render_csv(rows, fields)
    else:
        raise ValueError("Profile output must end in .csv or .json")

    metadata = {
        "reach_id": reach_id,
        "reach_name": reach["name"],
        "profile_path": str(output_path),
        "segments": len(rows),
        "generated_artifact": True,
        "minimum_slope": minimum_slope,
        "slope_values_adjusted": adjusted_slopes,
        "adjustments": adjustments,
        "rainfall_rate_m_per_min": rainfall_rate,
        "rainfall_observation_count": rainfall_obs_count,
        "recommended_upstream_inflow": recommendation,
        "rows": [
            {
                "station_m": row["station_m"],
                "label": row["label"],
                "classification": row["classification"],
            }
            for row in rows
        ],
        "sources": sources,
        "validation": {
            "counts": report.counts(),
            "findings": report.to_metadata(),
        },
    }
    metadata_path = output_path.with_suffix(".metadata.json")

    # Write the sidecar first, then the profile last, both atomically. A reader
    # keys off the profile, so it only appears once its metadata is in place.
    _atomic_write_text(metadata_path, json.dumps(metadata, indent=2) + "\n")
    _atomic_write_text(output_path, profile_text)
    return metadata
