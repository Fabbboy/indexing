from sqlite3 import Connection, connect
from pathlib import Path
from config import MIGRATION, Constants


def ensure_indexed_column(conn: Connection) -> None:
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(chunks)")
    columns = {row[1] for row in cursor.fetchall()}
    if "indexed" not in columns:
        cursor.execute("ALTER TABLE chunks ADD COLUMN indexed INTEGER NOT NULL DEFAULT 0")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_indexed ON chunks(indexed)")
        conn.commit()


def ensure_db(index_root: Path) -> Connection:
    """Create or connect to SQLite database and run migrations."""
    from indexing import ensure_root

    index_root = ensure_root(index_root)
    meta_file = index_root / Constants.META.value
    if not meta_file.exists():
        meta_file.touch()

    conn = connect(meta_file)
    conn.executescript(MIGRATION)
    ensure_indexed_column(conn)
    return conn
