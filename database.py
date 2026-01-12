from sqlite3 import Connection, connect
from pathlib import Path
from config import MIGRATION, Constants


def ensure_db(index_root: Path) -> Connection:
    """Create or connect to SQLite database and run migrations."""
    from indexing import ensure_root

    index_root = ensure_root(index_root)
    meta_file = index_root / Constants.META.value
    if not meta_file.exists():
        meta_file.touch()

    conn = connect(meta_file)
    conn.executescript(MIGRATION)
    return conn
