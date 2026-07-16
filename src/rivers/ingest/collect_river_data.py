import argparse
import json
import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rivers.ingest.common import connect_database
from rivers.ingest.database import DEFAULT_DB_PATH, initialize_database
from rivers.ingest.elevation import collect_elevations
from rivers.ingest.export_profile import export_profile
from rivers.ingest.markers import create_reach
from rivers.ingest.parameters import import_geometry, import_roughness
from rivers.ingest.rainfall import collect_rainfall
from rivers.ingest.usgs_flow import collect_usgs_flow


def build_parser():
    parser = argparse.ArgumentParser(description="Collect and prepare real-river model data")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="SQLite database path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Initialize the SQLite database")
    subparsers.add_parser("list-reaches", help="List configured reaches")

    reach = subparsers.add_parser("create-reach", help="Import an upstream-to-downstream centerline")
    reach.add_argument("--river", required=True)
    reach.add_argument("--reach", required=True)
    reach.add_argument("--markers", type=Path, required=True, help="CSV, JSON, or GeoJSON LineString")
    reach.add_argument("--region")
    reach.add_argument("--country")
    reach.add_argument("--notes")
    reach.add_argument("--replace", action="store_true")

    elevation = subparsers.add_parser("fetch-elevation", help="Fetch DEM elevations and derive slopes")
    elevation.add_argument("--reach-id", type=int, required=True)
    elevation.add_argument("--replace", action="store_true")

    flow = subparsers.add_parser("fetch-flow", help="Fetch USGS continuous discharge")
    flow.add_argument("--reach-id", type=int, required=True)
    flow.add_argument("--site", required=True, help="USGS gauge number or monitoring location ID")
    flow.add_argument("--start", required=True, help="RFC 3339 timestamp")
    flow.add_argument("--end", required=True, help="RFC 3339 timestamp")
    flow.add_argument("--marker-order", type=int, default=0)
    flow.add_argument("--api-key")
    flow.add_argument("--replace", action="store_true")

    rainfall = subparsers.add_parser("fetch-rainfall", help="Fetch historical hourly precipitation")
    rainfall.add_argument("--reach-id", type=int, required=True)
    rainfall.add_argument("--start-date", required=True)
    rainfall.add_argument("--end-date", required=True)
    rainfall.add_argument("--marker-order", type=int)
    rainfall.add_argument("--replace", action="store_true")

    roughness = subparsers.add_parser("import-roughness", help="Import reviewed Manning n intervals")
    roughness.add_argument("--reach-id", type=int, required=True)
    roughness.add_argument("--file", type=Path, required=True)
    roughness.add_argument("--replace", action="store_true")

    geometry = subparsers.add_parser("import-geometry", help="Import channel width/depth samples")
    geometry.add_argument("--reach-id", type=int, required=True)
    geometry.add_argument("--file", type=Path, required=True)
    geometry.add_argument("--replace", action="store_true")

    export = subparsers.add_parser("export-profile", help="Export a model-ready CSV or JSON profile")
    export.add_argument("--reach-id", type=int, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--minimum-slope", type=float, default=1e-6)
    export.add_argument("--initial-depth", type=float)
    export.add_argument("--rainfall-start")
    export.add_argument("--rainfall-end")
    export.add_argument("--flow-start")
    export.add_argument("--flow-end")
    return parser


def _list_reaches(db_path):
    with connect_database(db_path) as conn:
        rows = conn.execute(
            """
            SELECT reaches.id, rivers.name AS river, reaches.name AS reach,
                   reaches.length_m, COUNT(reach_markers.id) AS markers
            FROM reaches
            JOIN rivers ON rivers.id = reaches.river_id
            LEFT JOIN reach_markers ON reach_markers.reach_id = reaches.id
            GROUP BY reaches.id ORDER BY rivers.name, reaches.name
            """
        ).fetchall()
    return [dict(row) for row in rows]


def main(argv=None):
    args = build_parser().parse_args(argv)
    db_path = args.db
    if args.command == "init":
        result = {"database": str(initialize_database(db_path))}
    elif args.command == "list-reaches":
        result = _list_reaches(db_path)
    elif args.command == "create-reach":
        reach_id = create_reach(
            args.river,
            args.reach,
            args.markers,
            region=args.region,
            country=args.country,
            notes=args.notes,
            db_path=db_path,
            replace=args.replace,
        )
        result = {"reach_id": reach_id}
    elif args.command == "fetch-elevation":
        result = collect_elevations(args.reach_id, db_path=db_path, replace=args.replace)
    elif args.command == "fetch-flow":
        result = collect_usgs_flow(
            args.reach_id,
            args.site,
            args.start,
            args.end,
            marker_order=args.marker_order,
            api_key=args.api_key,
            db_path=db_path,
            replace=args.replace,
        )
    elif args.command == "fetch-rainfall":
        result = collect_rainfall(
            args.reach_id,
            args.start_date,
            args.end_date,
            marker_order=args.marker_order,
            db_path=db_path,
            replace=args.replace,
        )
    elif args.command == "import-roughness":
        result = import_roughness(args.reach_id, args.file, db_path=db_path, replace=args.replace)
    elif args.command == "import-geometry":
        result = import_geometry(args.reach_id, args.file, db_path=db_path, replace=args.replace)
    else:
        result = export_profile(
            args.reach_id,
            args.output,
            db_path=db_path,
            minimum_slope=args.minimum_slope,
            initial_depth_m=args.initial_depth,
            rainfall_start=args.rainfall_start,
            rainfall_end=args.rainfall_end,
            flow_start=args.flow_start,
            flow_end=args.flow_end,
        )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
