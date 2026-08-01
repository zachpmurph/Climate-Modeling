"""Create and optionally fetch fixed longer-window validation variants."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rivers.validation.fetch_event import fetch_event
from rivers.validation.fetch_point_flows import fetch_point_flows
from rivers.validation.providers import CONTINUOUS_URL, DISCHARGE_PARAMETER


def _timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("Extended-window timestamps must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _iso(timestamp):
    return timestamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _flow_url(gauge, start, end):
    gauge_id = gauge if "-" in gauge else f"USGS-{gauge}"
    return f"{CONTINUOUS_URL}?{urlencode({'f': 'json', 'monitoring_location_id': gauge_id, 'parameter_code': DISCHARGE_PARAMETER, 'datetime': f'{start}/{end}', 'limit': 50000})}"


def build_extended_config(base, *, extension_hours, output_stem):
    """Return a deterministic longer-window copy of one case config."""
    config = copy.deepcopy(base)
    start_text, end_text = config["case"]["observation_window"]
    start = _timestamp(start_text)
    end = _timestamp(end_text) + timedelta(hours=float(extension_hours))
    new_end = _iso(end)
    warmup_min = float(config.get("warmup", {}).get("duration_min", 0.0))
    warmup_start = _iso(start - timedelta(minutes=warmup_min))
    config["case"]["name"] += f" — extended by {extension_hours:g} hours"
    config["case"]["observation_window"] = [_iso(start), new_end]
    config["case"]["purpose"] = (
        "Fixed longer-window diagnostic retaining all baseline hydraulic "
        "assumptions; downstream observations remain scoring-only."
    )
    config["validation_status"] = "extended_window_diagnostic"
    config["observation_endpoint_tolerance_min"] = 15.0
    config["observations"] = f"{output_stem}.csv"
    config["retrieved_at"] = datetime.now(timezone.utc).date().isoformat()
    context = config.setdefault("reach_context", {})
    context["window_extension_hours"] = float(extension_hours)
    context["window_extension_policy"] = (
        "Append the same fixed duration to the pre-existing scored end; no "
        "parameter or endpoint is selected from model skill."
    )

    for source in config["sources"]:
        source_start = warmup_start if source["role"] == "upstream" else _iso(start)
        source["url"] = _flow_url(source["gauge"], source_start, new_end)
    if config.get("point_flow_series") is not None:
        config["point_flow_series"] = f"{output_stem}_point_flows.csv"
        for source in config.get("internal_sources", []):
            source["url"] = _flow_url(source["gauge"], warmup_start, new_end)
    return config


def generate_extended_cases(manifest_path, *, fetch=False):
    """Write all declared variants and a run-suite manifest."""
    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    extension_hours = float(manifest["extension_hours"])
    if extension_hours <= 0.0:
        raise ValueError("extension_hours must be positive")
    generated = []
    for relative in manifest["base_cases"]:
        base_path = root / relative
        base = json.loads(base_path.read_text(encoding="utf-8"))
        output_stem = f"{base_path.stem}_extended_{extension_hours:g}h"
        destination = root / f"{output_stem}.json"
        extended = build_extended_config(
            base, extension_hours=extension_hours, output_stem=output_stem
        )
        destination.write_text(
            json.dumps(extended, indent=2) + "\n", encoding="utf-8"
        )
        if fetch:
            fetch_event(destination)
            if extended.get("point_flow_series") is not None:
                fetch_point_flows(destination)
        generated.append(destination.name)

    suite_path = root / manifest["generated_suite"]
    suite = {
        "schema_version": 1,
        "suite": {
            "name": manifest["title"],
            "window_extension_hours": extension_hours,
        },
        "parameter_policy": (
            "No calibration. Every extended case preserves its baseline solver, "
            "geometry, roughness, grid, warm-up, and forcing rules."
        ),
        "cases": generated,
    }
    suite_path.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    return {"suite": str(suite_path), "cases": generated}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(generate_extended_cases(args.manifest, fetch=args.fetch), indent=2))


if __name__ == "__main__":
    main()
