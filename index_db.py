from pathlib import Path
from sqlite3 import Connection, Cursor
from os import stat_result
from typing import Any

from config import get_logger
from chunking import chunk_file

LOGGER = get_logger()


def process_file_batch(
    connection: Connection,
    paths: list[Path],
    source_root: Path,
    chunk_size: int | None = None
) -> tuple[list[tuple[str, int]], list[int]]:
    """Process a batch of files and return (prefixed_text, chunk_id) and ids to remove."""
    cursor = connection.cursor()
    embed_queue: list[tuple[str, int]] = []
    remove_ids: list[int] = []
    for path in paths:
        try:
            content, file_stat, relative_path = read_file_for_chunks(path, source_root)
            chunks = chunk_file(relative_path, content, chunk_size)
            new_queue, new_remove_ids = store_chunks(cursor, chunks, file_stat)
            embed_queue.extend(new_queue)
            remove_ids.extend(new_remove_ids)
        except Exception as e:
            LOGGER.warning(f"Failed to process {path}: {e}")
            continue
    connection.commit()
    return embed_queue, remove_ids


def read_file_for_chunks(
    path: Path,
    source_root: Path
) -> tuple[str, stat_result, Path]:
    content = path.read_text(encoding='utf-8', errors='ignore')
    file_stat = path.stat()
    relative_path = path.relative_to(source_root)
    return content, file_stat, relative_path


def store_chunks(
    cursor: Cursor,
    chunks: list[dict[str, Any]],
    file_stat: stat_result
) -> tuple[list[tuple[str, int]], list[int]]:
    embed_queue: list[tuple[str, int]] = []
    remove_ids: list[int] = []
    for chunk in chunks:
        chunk_id, was_indexed = upsert_chunk(cursor, chunk, file_stat)
        if was_indexed:
            remove_ids.append(chunk_id)
        prefixed_text = f"search_document: {chunk['content']}"
        embed_queue.append((prefixed_text, chunk_id))
    return embed_queue, remove_ids


def upsert_chunk(
    cursor: Cursor,
    chunk: dict[str, Any],
    file_stat: stat_result
) -> tuple[int, bool]:
    row = fetch_existing_chunk(cursor, chunk)
    if row:
        return update_chunk(cursor, row, chunk, file_stat)
    return insert_chunk(cursor, chunk, file_stat)


def fetch_existing_chunk(cursor: Cursor, chunk: dict[str, Any]) -> tuple[int, int] | None:
    cursor.execute("""
        SELECT id, indexed FROM chunks
        WHERE file_path = ? AND chunk_index = ?
    """, (str(chunk['file_path']), chunk['chunk_index']))
    row = cursor.fetchone()
    if not row:
        return None
    return int(row[0]), int(row[1])


def update_chunk(
    cursor: Cursor,
    row: tuple[int, int],
    chunk: dict[str, Any],
    file_stat: stat_result
) -> tuple[int, bool]:
    chunk_id, indexed = row
    cursor.execute("""
        UPDATE chunks
        SET file_size = ?, modified_time = ?, start_char = ?, end_char = ?, indexed = 0
        WHERE id = ?
    """, (
        file_stat.st_size,
        file_stat.st_mtime,
        chunk['start_char'],
        chunk['end_char'],
        chunk_id
    ))
    return chunk_id, indexed == 1


def insert_chunk(
    cursor: Cursor,
    chunk: dict[str, Any],
    file_stat: stat_result
) -> tuple[int, bool]:
    cursor.execute("""
        INSERT INTO chunks
        (file_path, chunk_index, file_size, modified_time, start_char, end_char, indexed)
        VALUES (?, ?, ?, ?, ?, ?, 0)
    """, (
        str(chunk['file_path']),
        chunk['chunk_index'],
        file_stat.st_size,
        file_stat.st_mtime,
        chunk['start_char'],
        chunk['end_char']
    ))
    return int(cursor.lastrowid), False


def read_chunk_content(file_path: Path, start_char: int, end_char: int) -> str:
    """Read the content of a specific chunk from a file."""
    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        return content[start_char:end_char]
    except Exception as e:
        LOGGER.warning(f"Failed to read chunk from {file_path}: {e}")
        return ""


def read_chunk_with_context(
    file_path: Path,
    start_char: int,
    end_char: int,
    context_chars: int
) -> str:
    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        start = max(0, start_char - context_chars)
        end = min(len(content), end_char + context_chars)
        return content[start:end]
    except Exception as e:
        LOGGER.warning(f"Failed to read context from {file_path}: {e}")
        return ""


def get_indexed_files(connection: Connection) -> set[str]:
    """Get set of file paths that are already indexed."""
    cursor = connection.cursor()
    cursor.execute("""
        SELECT file_path
        FROM chunks
        GROUP BY file_path
        HAVING MIN(indexed) = 1
    """)
    return {row[0] for row in cursor.fetchall()}
