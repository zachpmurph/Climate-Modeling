"""Fetch reviewed USGS channel measurements for validation geometry."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rivers.ingest.common import request_json
from rivers.ingest.usgs_flow import CFS_TO_M3_PER_MIN


CHANNEL_URL = (
    "https://api.waterdata.usgs.gov/ogcapi/v0/"
    "collections/channel-measurements/items"
)
FIELD_URL = (
    "https://api.waterdata.usgs.gov/ogcapi/v0/"
    "collections/field-measurements/items"
)
FT_TO_M = 0.3048
FT2_TO_M2 = FT_TO_M**2
FIELDNAMES = (
    "station_m",
    "monitoring_location_id",
    "field_visit_id",
    "measured_at",
    "active_width_m",
    "channel_area_m2",
    "hydraulic_mean_depth_m",
    "mean_gage_height_ft",
    "gage_datum_ft",
    "water_surface_elevation_ft",
    "effective_bed_elevation_ft",
    "represented_discharge_m3_per_min",
    "total_visit_discharge_m3_per_min",
    "represented_flow_fraction",
    "represented_channel_count",
    "published_channel_count",
    "inferred_manning_n_model",
    "inferred_manning_n_si",
    "roughness_assumption",
)


def _number(value):
    if value is None or str(value).strip() == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _require_unit(properties, field, expected):
    unit = str(properties.get(field, "")).strip()
    if unit != expected:
        raise ValueError(
            f"Unsupported {field} {unit!r}; expected {expected!r}"
        )


def collect_channel_geometry_rows(config, *, requester=request_json):
    """Return one aggregated geometry row per configured field visit."""
    sources = config.get("field_measurement_sources", [])
    if len(sources) < 2:
        raise ValueError(
            "field_measurement_sources needs at least two reviewed visits"
        )
    rows = []
    resolved_urls = {}
    for source in sources:
        gauge = source["gauge"]
        gauge_id = gauge if "-" in gauge else f"USGS-{gauge}"
        visit_id = source["field_visit_id"]
        params = {
            "f": "json",
            "monitoring_location_id": gauge_id,
            "field_visit_id": visit_id,
            "limit": 50000,
        }
        payload, url = requester(CHANNEL_URL, params)
        configured_url = source.get("url")
        if configured_url is not None and configured_url != url:
            raise ValueError(
                "Resolved channel-measurement URL does not match the "
                "committed source URL"
            )
        features = [
            feature
            for feature in payload.get("features", [])
            if feature.get("properties", {}).get("field_visit_id")
            == visit_id
        ]
        if not features:
            raise ValueError(f"No channel measurements returned for {visit_id}")
        field_payload, field_url = requester(FIELD_URL, params)
        configured_field_url = source.get("field_measurement_url")
        if (
            configured_field_url is not None
            and configured_field_url != field_url
        ):
            raise ValueError(
                "Resolved field-measurement URL does not match the "
                "committed source URL"
            )
        gage_height_features = [
            feature.get("properties", {})
            for feature in field_payload.get("features", [])
            if (
                feature.get("properties", {}).get("field_visit_id")
                == visit_id
                and feature.get("properties", {}).get("parameter_code")
                == "00065"
                and feature.get("properties", {}).get("reading_type")
                == "MeanGageHeight"
            )
        ]
        if len(gage_height_features) != 1:
            raise ValueError(
                f"Field visit {visit_id} needs exactly one MeanGageHeight"
            )
        gage_height_properties = gage_height_features[0]
        if gage_height_properties.get("approval_status") != "Approved":
            raise ValueError(
                f"Field visit {visit_id} MeanGageHeight is not approved"
            )
        _require_unit(
            gage_height_properties, "unit_of_measure", "ft"
        )
        mean_gage_height_ft = float(gage_height_properties["value"])
        gage_datum_ft = float(source["gage_datum_ft"])
        total_flow_cfs = 0.0
        represented_flow_cfs = 0.0
        width_ft = 0.0
        area_ft2 = 0.0
        represented_count = 0
        times = set()
        for feature in features:
            properties = feature.get("properties", {})
            times.add(properties["time"])
            flow = _number(properties.get("channel_flow"))
            width = _number(properties.get("channel_width"))
            area = _number(properties.get("channel_area"))
            if flow is not None:
                _require_unit(properties, "channel_flow_unit", "ft^3/s")
                total_flow_cfs += flow
            if width is None or area is None:
                continue
            _require_unit(properties, "channel_width_unit", "ft")
            _require_unit(properties, "channel_area_unit", "ft^2")
            if width <= 0.0 or area <= 0.0:
                raise ValueError("Published channel width and area must be positive")
            width_ft += width
            area_ft2 += area
            represented_count += 1
            if flow is not None:
                represented_flow_cfs += flow
        if len(times) != 1:
            raise ValueError(
                f"Field visit {visit_id} has inconsistent measurement times"
            )
        if (
            represented_count == 0
            or width_ft <= 0.0
            or area_ft2 <= 0.0
            or total_flow_cfs <= 0.0
        ):
            raise ValueError(
                f"Field visit {visit_id} has no usable width/area/flow geometry"
            )
        width_m = width_ft * FT_TO_M
        area_m2 = area_ft2 * FT2_TO_M2
        hydraulic_mean_depth_m = area_m2 / width_m
        water_surface_elevation_ft = (
            gage_datum_ft + mean_gage_height_ft
        )
        effective_bed_elevation_ft = (
            water_surface_elevation_ft
            - hydraulic_mean_depth_m / FT_TO_M
        )
        rows.append(
            {
                "station_m": f"{float(source['station_m']):.6f}",
                "monitoring_location_id": gauge_id,
                "field_visit_id": visit_id,
                "measured_at": times.pop(),
                "active_width_m": f"{width_m:.12f}",
                "channel_area_m2": f"{area_m2:.12f}",
                "hydraulic_mean_depth_m": (
                    f"{hydraulic_mean_depth_m:.12f}"
                ),
                "mean_gage_height_ft": f"{mean_gage_height_ft:.6f}",
                "gage_datum_ft": f"{gage_datum_ft:.6f}",
                "water_surface_elevation_ft": (
                    f"{water_surface_elevation_ft:.12f}"
                ),
                "effective_bed_elevation_ft": (
                    f"{effective_bed_elevation_ft:.12f}"
                ),
                "represented_discharge_m3_per_min": (
                    f"{represented_flow_cfs * CFS_TO_M3_PER_MIN:.12f}"
                ),
                "total_visit_discharge_m3_per_min": (
                    f"{total_flow_cfs * CFS_TO_M3_PER_MIN:.12f}"
                ),
                "represented_flow_fraction": (
                    f"{represented_flow_cfs / total_flow_cfs:.12f}"
                ),
                "represented_channel_count": represented_count,
                "published_channel_count": len(features),
            }
        )
        resolved_urls[f"{gauge_id}:channel"] = url
        resolved_urls[f"{gauge_id}:field"] = field_url
    rows.sort(key=lambda row: float(row["station_m"]))
    stations = [float(row["station_m"]) for row in rows]
    if any(
        right <= left for left, right in zip(stations, stations[1:])
    ):
        raise ValueError(
            "Field-measurement geometry stations must be strictly increasing"
        )
    upstream_bed_ft = float(rows[0]["effective_bed_elevation_ft"])
    downstream_bed_ft = float(rows[-1]["effective_bed_elevation_ft"])
    reach_length_m = stations[-1] - stations[0]
    inferred_slope = (
        (upstream_bed_ft - downstream_bed_ft) * FT_TO_M / reach_length_m
    )
    if not math.isfinite(inferred_slope) or inferred_slope <= 0.0:
        raise ValueError(
            "Field measurements must imply a positive downstream bed slope"
        )
    for row in rows:
        width_m = float(row["active_width_m"])
        area_m2 = float(row["channel_area_m2"])
        mean_depth_m = float(row["hydraulic_mean_depth_m"])
        discharge = float(row["represented_discharge_m3_per_min"])
        hydraulic_radius = area_m2 / (
            width_m + 2.0 * mean_depth_m
        )
        manning_model = (
            area_m2
            * hydraulic_radius ** (2.0 / 3.0)
            * math.sqrt(inferred_slope)
            / discharge
        )
        row["inferred_manning_n_model"] = f"{manning_model:.15f}"
        row["inferred_manning_n_si"] = f"{manning_model * 60.0:.12f}"
        row["roughness_assumption"] = (
            "Manning inversion with rectangular wetted perimeter, "
            "field-measured area and represented discharge, and reach-average "
            "effective-bed slope"
        )
    return rows, resolved_urls


def fetch_channel_geometry(
    config_path, *, output_path=None, requester=request_json
):
    """Fetch configured visits and write their aggregated reviewed geometry."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows, urls = collect_channel_geometry_rows(config, requester=requester)
    destination = (
        Path(output_path)
        if output_path is not None
        else config_path.parent
        / config["reach"]["field_measurement_geometry"]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "output": str(destination),
        "rows": len(rows),
        "urls": urls,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fetch and aggregate reviewed USGS channel measurements."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            fetch_channel_geometry(args.config, output_path=args.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
