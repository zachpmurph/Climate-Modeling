"""Fetch approved USGS tributary/diversion hydrographs for one validation case."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rivers.validation.fetch_event import _as_query_time, _as_utc_iso
from rivers.validation.providers import fetch_usgs_flow, request_json


FIELDNAMES = (
    "source_name",
    "station_m",
    "site_id",
    "observed_at",
    "discharge_m3_per_min",
    "original_cfs",
    "approval_status",
)


def collect_point_flow_rows(config, *, requester=request_json):
    """Return normalized point-flow rows and resolved provider URLs."""
    source_definitions = config.get("internal_sources", [])
    if not source_definitions:
        raise ValueError("Event config must contain at least one internal_sources entry")
    start, end = config["case"]["observation_window"]
    start_time = datetime.fromisoformat(start.replace("Z", "+00:00"))
    warmup_min = float(config.get("warmup", {}).get("duration_min", 0.0))
    query_start = _as_query_time(start_time - timedelta(minutes=warmup_min))

    rows = []
    resolved_urls = {}
    seen_names = set()
    for source in source_definitions:
        name = str(source["name"])
        if name in seen_names:
            raise ValueError(f"Duplicate internal source name: {name}")
        seen_names.add(name)
        station = float(source["station_m"])
        if not math.isfinite(station) or station < 0.0:
            raise ValueError(f"Internal source {name} has invalid station_m")
        observations, url, gauge_id = fetch_usgs_flow(
            source["gauge"], query_start, end, requester=requester
        )
        configured_url = source.get("url")
        if configured_url is not None and configured_url != url:
            raise ValueError(
                f"Resolved provider URL for {name} does not match committed URL"
            )
        resolved_urls[name] = url
        normalized = []
        for observation in observations:
            if observation["approval_status"] != "Approved":
                raise ValueError(
                    f"{gauge_id} returned non-approved observation at "
                    f"{observation['observed_at']}"
                )
            normalized.append(
                {
                    "source_name": name,
                    "station_m": f"{station:.3f}",
                    "site_id": gauge_id,
                    "observed_at": _as_utc_iso(observation["observed_at"]),
                    "discharge_m3_per_min": (
                        f"{observation['discharge_m3_per_min']:.12f}"
                    ),
                    "original_cfs": f"{observation['value']:.6f}",
                    "approval_status": observation["approval_status"],
                }
            )
        normalized.sort(key=lambda row: row["observed_at"])
        if len(normalized) < 2:
            raise ValueError(f"Internal source {name} needs at least two observations")
        if normalized[0]["observed_at"] != _as_utc_iso(query_start):
            raise ValueError(f"Internal source {name} does not cover warm-up start")
        if normalized[-1]["observed_at"] != _as_utc_iso(end):
            raise ValueError(f"Internal source {name} does not cover event end")
        rows.extend(normalized)

    rows.sort(key=lambda row: (float(row["station_m"]), row["observed_at"]))
    return rows, resolved_urls


def fetch_point_flows(config_path, *, output_path=None, requester=request_json):
    """Fetch all configured internal sources and write their offline CSV."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "point_flow_series" not in config:
        raise ValueError("Event config must declare point_flow_series")
    rows, urls = collect_point_flow_rows(config, requester=requester)
    destination = (
        Path(output_path)
        if output_path is not None
        else config_path.parent / config["point_flow_series"]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return {"output": str(destination), "rows": len(rows), "urls": urls}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fetch approved observed internal flows for a validation case."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            fetch_point_flows(args.config, output_path=args.output), indent=2
        )
    )


if __name__ == "__main__":
    main()
