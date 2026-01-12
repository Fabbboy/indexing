from sqlite3 import Connection

from faiss import Index

from config import get_logger
from index_store import save_index

LOGGER = get_logger()


def reconcile_index_state(meta_db: Connection, faiss_index: Index, index_root) -> None:
    db_count = count_chunks(meta_db)
    index_count = faiss_index.ntotal
    if db_count == 0:
        return
    if index_count == 0:
        mark_all_unindexed(meta_db)
        return
    if index_count == db_count:
        mark_all_indexed(meta_db)
        return
    LOGGER.warning(
        "FAISS index count does not match metadata; resetting vectors to prevent mismatch."
    )
    reset_index(meta_db, faiss_index, index_root)


def count_chunks(meta_db: Connection) -> int:
    cursor = meta_db.execute("SELECT COUNT(*) FROM chunks")
    row = cursor.fetchone()
    return int(row[0]) if row else 0


def mark_all_unindexed(meta_db: Connection) -> None:
    meta_db.execute("UPDATE chunks SET indexed = 0")
    meta_db.commit()


def mark_all_indexed(meta_db: Connection) -> None:
    meta_db.execute("UPDATE chunks SET indexed = 1 WHERE indexed = 0")
    meta_db.commit()


def reset_index(meta_db: Connection, faiss_index: Index, index_root) -> None:
    faiss_index.reset()
    save_index(faiss_index, index_root)
    mark_all_unindexed(meta_db)
