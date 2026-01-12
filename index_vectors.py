from sqlite3 import Connection

import numpy as np
from faiss import Index

from index_store import save_index


def add_vectors(
    faiss_index: Index,
    vectors: list[list[float]],
    vector_ids: list[int],
    index_root
) -> int:
    if not vectors:
        return 0
    vector_array = np.array(vectors, dtype='float32')
    id_array = np.array(vector_ids, dtype='int64')
    faiss_index.add_with_ids(vector_array, id_array)
    save_index(faiss_index, index_root)
    return len(vectors)


def remove_vectors(faiss_index: Index, vector_ids: list[int]) -> None:
    if not vector_ids:
        return
    id_array = np.array(vector_ids, dtype='int64')
    faiss_index.remove_ids(id_array)


def mark_chunks_indexed(meta_db: Connection, chunk_ids: list[int]) -> None:
    if not chunk_ids:
        return
    placeholders = ",".join("?" for _ in chunk_ids)
    meta_db.execute(
        f"UPDATE chunks SET indexed = 1 WHERE id IN ({placeholders})",
        chunk_ids
    )
    meta_db.commit()
