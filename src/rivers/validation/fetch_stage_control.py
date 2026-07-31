"""Fetch and datum-normalize an approved USGS downstream-stage control."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rivers.validation.providers import request_json


CONTINUOUS_URL = (
    "https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous/items"
)
STAGE_PARAMETER = "00065"
FT_TO_M = 0.3048
FIELDNAMES = (
    "observed_at",
    "downstream_stage_m",
    "original_gage_height_ft",
    "approval_status",
)


def _utc(value):
    timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if timestamp.utcoffset() is None:
        raise ValueError("Stage timestamp must include a UTC offset")
    return timestamp.astimezone(timezone.utc)


def _query_time(timestamp):
    return timestamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso(timestamp):
    return timestamp.astimezone(timezone.utc).isoformat()


def _stage_window(config):
    start, end = config["case"]["observation_window"]
    event_start = _utc(start)
    event_end = _utc(end)
    warmup_min = float(
        config.get("warmup", {}).get(
            "duration_min", config.get("warmup_min", 0.0)
        )
    )
    return event_start - timedelta(minutes=warmup_min), event_end


def collect_stage_rows(config, *, requester=request_json):
    """Return normalized stage rows and the resolved provider URL."""
    source = config["downstream_stage_source"]
    if source.get("parameter", STAGE_PARAMETER) != STAGE_PARAMETER:
        raise ValueError("Downstream stage source must use USGS parameter 00065")
    gauge = source["gauge"]
    gauge_id = gauge if "-" in gauge else f"USGS-{gauge}"
    start, end = _stage_window(config)
    params = {
        "f": "json",
        "monitoring_location_id": gauge_id,
        "parameter_code": STAGE_PARAMETER,
        "datetime": f"{_query_time(start)}/{_query_time(end)}",
        "limit": 50000,
    }
    payload, url = requester(CONTINUOUS_URL, params)
    features = list(payload.get("features", []))
    next_url = next(
        (
            link.get("href")
            for link in payload.get("links", [])
            if link.get("rel") == "next"
        ),
        None,
    )
    while next_url:
        payload, _ = requester(next_url)
        features.extend(payload.get("features", []))
        next_url = next(
            (
                link.get("href")
                for link in payload.get("links", [])
                if link.get("rel") == "next"
            ),
            None,
        )

    configured_url = source.get("url")
    if configured_url is not None and configured_url != url:
        raise ValueError(
            "Resolved downstream-stage provider URL does not match the "
            "committed source URL"
        )
    gauge_datum_ft = float(source["gage_datum_ft"])
    model_datum_ft = float(source["model_vertical_datum_ft"])
    rows = []
    for feature in features:
        properties = feature.get("properties", {})
        if properties.get("parameter_code") != STAGE_PARAMETER:
            continue
        value = properties.get("value")
        if value is None:
            continue
        unit = str(properties.get("unit_of_measure", "ft")).strip().lower()
        if unit not in {"ft", "feet", "foot"}:
            raise ValueError(f"Unsupported USGS gage-height unit: {unit}")
        approval = properties.get("approval_status")
        if approval != "Approved":
            raise ValueError(
                f"{gauge_id} returned non-approved stage at "
                f"{properties.get('time')}"
            )
        height_ft = float(value)
        model_stage_m = (
            gauge_datum_ft + height_ft - model_datum_ft
        ) * FT_TO_M
        rows.append(
            {
                "observed_at": _iso(_utc(properties["time"])),
                "downstream_stage_m": f"{model_stage_m:.12f}",
                "original_gage_height_ft": f"{height_ft:.6f}",
                "approval_status": approval,
            }
        )
    rows.sort(key=lambda row: row["observed_at"])
    if len(rows) < 2:
        raise ValueError("Stage control needs at least two approved observations")
    if rows[0]["observed_at"] != _iso(start):
        raise ValueError("Stage observations do not begin at the warm-up start")
    if rows[-1]["observed_at"] != _iso(end):
        raise ValueError("Stage observations do not end at the event end")
    return rows, url


def fetch_stage_control(config_path, *, output_path=None, requester=request_json):
    """Fetch one configured stage series and write its normalized offline CSV."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows, url = collect_stage_rows(config, requester=requester)
    destination = (
        Path(output_path)
        if output_path is not None
        else config_path.parent / config["downstream_stage_series"]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "output": str(destination),
        "rows": len(rows),
        "url": url,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fetch approved USGS stage and convert it to model datum."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            fetch_stage_control(args.config, output_path=args.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
