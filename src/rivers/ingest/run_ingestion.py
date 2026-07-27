"""CLI entry point for config-driven ingestion.

Ingest one curated reach definition, or every definition in a directory:

    python -m rivers.ingest.run_ingestion real_world_rivers/curated/columbia_hanford.json
    python -m rivers.ingest.run_ingestion --all real_world_rivers/curated

Exit status is non-zero if any reach failed or any declared export is missing
from disk afterwards, so the command is safe to gate a pipeline on.
"""

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rivers.ingest.database import DEFAULT_DB_PATH
from rivers.ingest.orchestrator import ingest_all, ingest_reach


def build_parser():
    parser = argparse.ArgumentParser(description="Config-driven real-world river ingestion")
    parser.add_argument("target", type=Path,
                        help="A curated definition JSON, or a directory when --all is given")
    parser.add_argument("--all", action="store_true",
                        help="Treat TARGET as a directory and ingest every *.json definition")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="SQLite database path")
    parser.add_argument("--replace", action="store_true",
                        help="Replace existing reach data instead of erroring on conflict")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.all:
        summary = ingest_all(args.target, db_path=args.db, replace=args.replace, requesters=None)
        print(json.dumps(summary, indent=2))
        return 0 if summary["overall_success"] else 1

    result = ingest_reach(args.target, db_path=args.db, replace=args.replace, requesters=None)
    print(json.dumps(result, indent=2))
    export = result.get("export")
    ok = result["status"] == "ok" and bool(export) and export.get("exists")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
