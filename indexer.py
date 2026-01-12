from pathlib import Path
from sqlite3 import Connection
import time

from openai import OpenAI

from config import get_logger
from database import ensure_db
from index_batches import run_index_batches
from index_db import get_indexed_files
from index_paths import collect_paths
from index_state import reconcile_index_state
from index_store import ensure_index
from schemas import IndexConfig

LOGGER = get_logger()


def get_paths_to_index(
    source_root: Path,
    meta_db: Connection,
    erase: bool
) -> list[Path]:
    paths = collect_paths(source_root)
    if erase:
        return paths
    indexed_files = get_indexed_files(meta_db)
    if indexed_files:
        LOGGER.info(f"Found {len(indexed_files)} already indexed files, skipping them.")
    return [p for p in paths if str(p.relative_to(source_root)) not in indexed_files]


def log_index_summary(start_time: float, total_chunks: int, path_count: int) -> None:
    elapsed = time.time() - start_time
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    LOGGER.info(
        f"✓ Indexing complete! Indexed {total_chunks} chunks from {path_count} files in {elapsed_str}."
    )


def run_indexing(
    source_root: Path,
    index_root: Path,
    config: IndexConfig,
    client: OpenAI,
    erase: bool
) -> None:
    start_time = time.time()
    meta_db = ensure_db(index_root)
    faiss_index = ensure_index(index_root)
    reconcile_index_state(meta_db, faiss_index, index_root)
    LOGGER.info(f"Indexing files from {source_root} into index at {index_root}")
    paths = get_paths_to_index(source_root, meta_db, erase)
    LOGGER.info(f"Collected {len(paths)} files to index.")
    if not paths:
        LOGGER.warning("No new files to index.")
        meta_db.close()
        return
    total_chunks = run_index_batches(
        meta_db=meta_db,
        faiss_index=faiss_index,
        client=client,
        paths=paths,
        source_root=source_root,
        index_root=index_root,
        config=config
    )
    meta_db.close()
    log_index_summary(start_time, total_chunks, len(paths))
