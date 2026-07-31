"""Fetch and normalize an approved USGS two-gauge validation event."""

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

from rivers.validation.providers import fetch_usgs_flow, request_json


FIELDNAMES = (
    "role",
    "site_id",
    "observed_at",
    "discharge_m3_per_min",
    "original_cfs",
    "approval_status",
)


def _as_utc_iso(value):
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.utcoffset() is None:
        raise ValueError("USGS observation timestamp must include a UTC offset")
    return timestamp.astimezone(timezone.utc).isoformat()


def _as_query_time(timestamp):
    return timestamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def collect_event_rows(config, *, requester=request_json):
    """Return validated CSV rows and the resolved provider URLs."""
    start, end = config["case"]["observation_window"]
    start_time = datetime.fromisoformat(start.replace("Z", "+00:00"))
    warmup = config.get("warmup", {})
    warmup_min = float(
        warmup.get("duration_min", config.get("warmup_min", 0.0))
    )
    observed_warmup = warmup.get("upstream_forcing") == "observed"
    expected_roles = {"upstream", "downstream"}
    sources = config.get("sources", [])
    roles = {source.get("role") for source in sources}
    if roles != expected_roles or len(sources) != 2:
        raise ValueError("Event config must contain one upstream and one downstream source")

    rows = []
    resolved_urls = {}
    for source in sources:
        role = source["role"]
        gauge = source["gauge"]
        query_start = start
        if role == "upstream" and observed_warmup:
            query_start = _as_query_time(
                start_time - timedelta(minutes=warmup_min)
            )
        observations, url, gauge_id = fetch_usgs_flow(
            gauge,
            query_start,
            end,
            requester=requester,
        )
        configured_url = source.get("url")
        if configured_url is not None and configured_url != url:
            raise ValueError(
                f"Resolved {role} provider URL does not match the committed source URL"
            )
        resolved_urls[role] = url
        for observation in observations:
            if observation["approval_status"] != "Approved":
                raise ValueError(
                    f"{gauge_id} returned non-approved observation at "
                    f"{observation['observed_at']}"
                )
            rows.append(
                {
                    "role": role,
                    "site_id": gauge_id,
                    "observed_at": _as_utc_iso(observation["observed_at"]),
                    "discharge_m3_per_min": f"{observation['discharge_m3_per_min']:.12f}",
                    "original_cfs": f"{observation['value']:.6f}",
                    "approval_status": observation["approval_status"],
                }
            )

    role_order = {"upstream": 0, "downstream": 1}
    rows.sort(key=lambda row: (role_order[row["role"]], row["observed_at"]))
    for role in expected_roles:
        selected = [row for row in rows if row["role"] == role]
        if len(selected) < 2:
            raise ValueError(f"Event needs at least two {role} observations")
        expected_start = start
        if role == "upstream" and observed_warmup:
            expected_start = _as_query_time(
                start_time - timedelta(minutes=warmup_min)
            )
        if selected[0]["observed_at"] != _as_utc_iso(expected_start):
            raise ValueError(f"{role} observations do not begin at the configured start")
        if selected[-1]["observed_at"] != _as_utc_iso(end):
            raise ValueError(f"{role} observations do not end at the configured end")
    return rows, resolved_urls


def fetch_event(config_path, *, output_path=None, requester=request_json):
    """Fetch one configured event and write its normalized offline CSV."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows, resolved_urls = collect_event_rows(config, requester=requester)
    destination = (
        Path(output_path)
        if output_path is not None
        else config_path.parent / config["observations"]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "output": str(destination),
        "rows": len(rows),
        "urls": resolved_urls,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fetch one approved USGS two-gauge validation event."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(fetch_event(args.config, output_path=args.output), indent=2))


if __name__ == "__main__":
    main()
