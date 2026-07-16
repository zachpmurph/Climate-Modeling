from pathlib import Path
import sqlite3


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "real_world_rivers"
SCHEMA_PATH = DATA_DIR / "schema.sql"
DEFAULT_DB_PATH = DATA_DIR / "river_inputs.sqlite"


def initialize_database(db_path=DEFAULT_DB_PATH, schema_path=SCHEMA_PATH):
    """Create or update the SQLite database using the project schema."""
    db_path = Path(db_path)
    schema_path = Path(schema_path)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    schema = schema_path.read_text(encoding="utf-8")

    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema)
        conn.execute("PRAGMA foreign_keys = ON")

    return db_path


def main():
    db_path = initialize_database()
    print(f"Initialized river input database: {db_path}")


if __name__ == "__main__":
    main()

