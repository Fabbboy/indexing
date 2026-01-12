from pathlib import Path
from sqlite3 import Connection
from typing import Any

import numpy as np
from openai import OpenAI

from config import Constants
from database import ensure_db
from indexing import ensure_index, read_chunk_content, read_chunk_with_context


def make_query_embedding(
    client: OpenAI,
    model: str,
    query_str: str
) -> tuple[list[float] | None, str | None]:
    prefixed_query = f"search_query: {query_str}"
    try:
        response = client.embeddings.create(
            input=[prefixed_query],
            model=model,
            dimensions=Constants.DIMENSIONS.value
        )
        return response.data[0].embedding, None
    except Exception as exc:
        return None, f"Failed to generate query embedding: {exc}"


def fetch_search_results(
    meta_db: Connection,
    source_root: Path,
    indices: Any,
    distances: Any,
    include_content: bool,
    context_chars: int,
    include_metadata: bool
) -> list[dict[str, Any]]:
    cursor = meta_db.cursor()
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if int(idx) == -1:
            continue
        row = fetch_chunk_row(cursor, int(idx))
        if row:
            results.append(
                build_search_result(
                    row,
                    distance,
                    source_root,
                    include_content,
                    context_chars,
                    include_metadata
                )
            )
    return results


def fetch_chunk_row(cursor, chunk_id: int) -> tuple[Any, Any, Any, Any] | None:
    cursor.execute("""
        SELECT file_path, chunk_index, start_char, end_char
        FROM chunks
        WHERE id = ?
    """, (chunk_id,))
    row = cursor.fetchone()
    if not row:
        return None
    return row


def build_search_result(
    row: tuple[Any, Any, Any, Any],
    distance: Any,
    source_root: Path,
    include_content: bool,
    context_chars: int,
    include_metadata: bool
) -> dict[str, Any]:
    result = build_result_base(row, distance, include_metadata)
    if include_content:
        content, context = read_match_content(source_root, row, context_chars)
        result["content"] = content
        result["context"] = context
    return result


def build_result_base(
    row: tuple[Any, Any, Any, Any],
    distance: Any,
    include_metadata: bool
) -> dict[str, Any]:
    result: dict[str, Any] = {'file_path': row[0]}
    if include_metadata:
        result.update({
            'chunk_index': row[1],
            'start_char': row[2],
            'end_char': row[3],
            'distance': float(distance),
            'similarity': 1.0 / (1.0 + float(distance))
        })
    return result


def read_match_content(
    source_root: Path,
    row: tuple[Any, Any, Any, Any],
    context_chars: int
) -> tuple[str, str]:
    content = read_chunk_content(source_root / row[0], row[2], row[3])
    context = read_chunk_with_context(
        source_root / row[0],
        row[2],
        row[3],
        context_chars
    )
    return content, context


def run_faiss_search(
    faiss_index: Any,
    query_vector: list[float],
    limit: int
) -> tuple[Any, Any]:
    query_array = np.array([query_vector], dtype='float32')
    return faiss_index.search(query_array, limit)


def search_index(
    query_str: str,
    source_root: Path,
    index_root: Path,
    client: OpenAI,
    model: str,
    limit: int,
    include_content: bool = False,
    context_chars: int = 160,
    include_metadata: bool = True
) -> tuple[list[dict[str, Any]], str | None]:
    faiss_index = ensure_index(index_root)
    if faiss_index.ntotal == 0:
        return [], "Index is empty. Run 'index' command first."
    meta_db = ensure_db(index_root)
    try:
        return run_search(
            meta_db=meta_db,
            faiss_index=faiss_index,
            source_root=source_root,
            client=client,
            model=model,
            query_str=query_str,
            limit=limit,
            include_content=include_content,
            context_chars=context_chars,
            include_metadata=include_metadata
        )
    finally:
        meta_db.close()


def run_search(
    meta_db: Connection,
    faiss_index: Any,
    source_root: Path,
    client: OpenAI,
    model: str,
    query_str: str,
    limit: int,
    include_content: bool,
    context_chars: int,
    include_metadata: bool
) -> tuple[list[dict[str, Any]], str | None]:
    query_vector, error = make_query_embedding(client, model, query_str)
    if error:
        return [], error
    distances, indices = run_faiss_search(faiss_index, query_vector, limit)
    results = fetch_search_results(
        meta_db=meta_db,
        source_root=source_root,
        indices=indices,
        distances=distances,
        include_content=include_content,
        context_chars=context_chars,
        include_metadata=include_metadata
    )
    return results, None
