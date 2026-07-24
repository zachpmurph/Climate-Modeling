"""Config-driven ingestion.

A curated reach definition (JSON) declares everything needed to build one
model-ready profile: identifiers, geometry/marker sources, provider windows,
and export options. :func:`ingest_reach` runs the full pipeline for one
definition; :func:`ingest_all` runs every definition in a directory, continuing
past failures and reporting successes and failures together.

Paths inside a definition are resolved relative to the definition file's own
directory. Network providers are injected through a ``requesters`` mapping
(keys ``elevation``/``flow``/``rainfall``), defaulting to the real HTTP
requester, so tests and dry-runs stay offline.

Definition schema (all sections except river/reach/markers/export optional)::

    {
      "river":    {"name": "...", "region": "...", "country": "...", "notes": "..."},
      "reach":    {"name": "...", "notes": "..."},
      "markers":  "markers.csv",                 # CSV/JSON/GeoJSON centerline
      "elevation":{"provider": "open-meteo-dem"},# presence => fetch DEM + slopes
      "roughness":{"file": "roughness.csv"},     # reviewed Manning n intervals
      "geometry": {"file": "geometry.csv"},      # optional width/depth samples
      "flow":     {"site": "12345678", "start": "...", "end": "...", "marker_order": 0},
      "rainfall": {"start_date": "...", "end_date": "...", "marker_order": 1},
      "export":   {"output": "reach.profile.csv", "minimum_slope": 1e-6,
                   "initial_depth_m": null, "rainfall_window": ["...","..."],
                   "flow_window": ["...","..."]}
    }
"""

import json
import warnings
from pathlib import Path

from .common import request_json
from .database import DEFAULT_DB_PATH
from .elevation import collect_elevations
from .export_profile import export_profile
from .markers import create_reach
from .parameters import import_geometry, import_roughness
from .rainfall import collect_rainfall
from .usgs_flow import collect_usgs_flow
from .validation import ProfileValidationError


def _resolve(base_dir, value):
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path)


def _requester(requesters, key):
    if requesters and key in requesters:
        return requesters[key]
    return request_json


def load_definition(definition_path):
    definition_path = Path(definition_path)
    data = json.loads(definition_path.read_text(encoding="utf-8"))
    for required in ("river", "reach", "markers", "export"):
        if required not in data:
            raise ValueError(f"{definition_path.name}: missing required section '{required}'")
    return data


def ingest_reach(definition_path, *, db_path=None, replace=False, requesters=None):
    """Run the full ingestion pipeline for one curated definition.

    Returns a result dict with ``status`` (``ok``/``error``), the steps run and
    their counts, retained warnings, and the export summary. Provider or
    validation failures are captured into the result rather than raised, so a
    batch can continue.
    """
    definition_path = Path(definition_path)
    result = {
        "definition": str(definition_path),
        "river": None,
        "reach": None,
        "reach_id": None,
        "status": "ok",
        "steps": {},
        "warnings": [],
        "export": None,
        "error": None,
    }
    try:
        definition = load_definition(definition_path)
        base_dir = definition_path.parent
        river = definition["river"]
        reach = definition["reach"]
        export_cfg = definition["export"]
        result["river"] = river.get("name")
        result["reach"] = reach.get("name")

        db = DEFAULT_DB_PATH if db_path is None else db_path

        reach_id = create_reach(
            river["name"], reach["name"], _resolve(base_dir, definition["markers"]),
            region=river.get("region"), country=river.get("country"),
            notes=reach.get("notes") or river.get("notes"),
            db_path=db, replace=replace,
        )
        result["reach_id"] = reach_id

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")

            if "elevation" in definition:
                res = collect_elevations(
                    reach_id, db_path=db, replace=replace,
                    requester=_requester(requesters, "elevation"),
                )
                result["steps"]["elevation"] = res

            if "roughness" in definition:
                res = import_roughness(
                    reach_id, _resolve(base_dir, definition["roughness"]["file"]),
                    db_path=db, replace=replace,
                )
                result["steps"]["roughness"] = res

            if "geometry" in definition:
                res = import_geometry(
                    reach_id, _resolve(base_dir, definition["geometry"]["file"]),
                    db_path=db, replace=replace,
                )
                result["steps"]["geometry"] = res

            if "flow" in definition:
                flow = definition["flow"]
                res = collect_usgs_flow(
                    reach_id, flow["site"], flow["start"], flow["end"],
                    marker_order=flow.get("marker_order", 0),
                    api_key=flow.get("api_key"), db_path=db, replace=replace,
                    requester=_requester(requesters, "flow"),
                )
                result["steps"]["flow"] = res

            rainfall = definition.get("rainfall")
            if rainfall:
                res = collect_rainfall(
                    reach_id, rainfall["start_date"], rainfall["end_date"],
                    marker_order=rainfall.get("marker_order"), db_path=db, replace=replace,
                    requester=_requester(requesters, "rainfall"),
                )
                result["steps"]["rainfall"] = res

            # Export windows: fall back to the rainfall/flow definition windows.
            # Rainfall observed_at values are full timestamps (e.g. 2020-01-01T00:00),
            # so a date-only window end would exclude that day lexically — expand
            # the date range to full-day bounds.
            rainfall_window = export_cfg.get("rainfall_window")
            if rainfall_window is None and rainfall:
                rainfall_window = [
                    f"{rainfall['start_date']}T00:00",
                    f"{rainfall['end_date']}T23:59:59",
                ]
            flow_window = export_cfg.get("flow_window")
            if flow_window is None and "flow" in definition:
                flow_window = [definition["flow"]["start"], definition["flow"]["end"]]

            output = _resolve(base_dir, export_cfg["output"])
            metadata = export_profile(
                reach_id, output, db_path=db,
                minimum_slope=export_cfg.get("minimum_slope", 1e-6),
                initial_depth_m=export_cfg.get("initial_depth_m"),
                rainfall_start=rainfall_window[0] if rainfall_window else None,
                rainfall_end=rainfall_window[1] if rainfall_window else None,
                flow_start=flow_window[0] if flow_window else None,
                flow_end=flow_window[1] if flow_window else None,
            )

        result["warnings"] = [str(w.message) for w in captured]
        # Fold validation warnings/infos into the reported warnings for the batch summary.
        for finding in metadata["validation"]["findings"]:
            if finding["severity"] == "warning":
                result["warnings"].append(f"{finding['code']}: {finding['message']}")
        result["export"] = {
            "path": str(output),
            "exists": Path(output).exists(),
            "validation": metadata["validation"],
            "adjustments": metadata["adjustments"],
        }
    except ProfileValidationError as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        result["validation_errors"] = [f.to_dict() for f in exc.report.errors]
    except Exception as exc:  # noqa: BLE001 - batch must survive one reach's failure
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def ingest_all(curated_dir, *, db_path=None, replace=False, requesters=None):
    """Ingest every ``*.json`` definition in ``curated_dir``.

    Continues past individual failures and aggregates the results. Overall
    success requires every reach to succeed AND every declared export file to
    exist on disk afterwards.
    """
    curated_dir = Path(curated_dir)
    definition_paths = sorted(curated_dir.glob("*.json"))
    reaches = [
        ingest_reach(path, db_path=db_path, replace=replace, requesters=requesters)
        for path in definition_paths
    ]
    succeeded = sum(1 for r in reaches if r["status"] == "ok")
    failed = len(reaches) - succeeded
    exports_present = all(
        r["export"] and r["export"]["exists"] for r in reaches if r["status"] == "ok"
    )
    return {
        "curated_dir": str(curated_dir),
        "total": len(reaches),
        "succeeded": succeeded,
        "failed": failed,
        "overall_success": failed == 0 and exports_present and len(reaches) > 0,
        "reaches": reaches,
    }
