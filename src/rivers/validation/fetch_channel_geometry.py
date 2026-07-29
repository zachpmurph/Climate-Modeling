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
    "represented_discharge_m3_per_min",
    "total_visit_discharge_m3_per_min",
    "represented_flow_fraction",
    "represented_channel_count",
    "published_channel_count",
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
        rows.append(
            {
                "station_m": f"{float(source['station_m']):.6f}",
                "monitoring_location_id": gauge_id,
                "field_visit_id": visit_id,
                "measured_at": times.pop(),
                "active_width_m": f"{width_m:.12f}",
                "channel_area_m2": f"{area_m2:.12f}",
                "hydraulic_mean_depth_m": f"{area_m2 / width_m:.12f}",
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
        resolved_urls[gauge_id] = url
    rows.sort(key=lambda row: float(row["station_m"]))
    stations = [float(row["station_m"]) for row in rows]
    if any(
        right <= left for left, right in zip(stations, stations[1:])
    ):
        raise ValueError(
            "Field-measurement geometry stations must be strictly increasing"
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
