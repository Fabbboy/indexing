from pathlib import Path
from sqlite3 import Connection
import time

from faiss import Index
from openai import OpenAI
from tqdm import tqdm

from ai_utils import log_model_error
from config import get_logger
from embeddings import generate_embeddings_batch
from index_db import process_file_batch
from index_store import save_index
from index_vectors import add_vectors, mark_chunks_indexed, remove_vectors
from schemas import IndexConfig

LOGGER = get_logger()


def embed_text_batches(
    client: OpenAI,
    texts: list[str],
    model: str,
    batch_size: int,
    delay: float
) -> list[list[float]]:
    vectors = []
    with tqdm(total=len(texts), desc="  Embedding batch", unit="chunk", leave=False) as pbar:
        for i in range(0, len(texts), batch_size):
            text_batch = texts[i:i + batch_size]
            try:
                vectors.extend(generate_embeddings_batch(client, text_batch, model))
                pbar.update(len(text_batch))
                if delay > 0:
                    time.sleep(delay)
            except Exception as exc:
                LOGGER.error(f"Failed to generate embeddings: {exc}")
                log_model_error(client, str(exc))
                pbar.update(len(text_batch))
    return vectors


def split_embed_queue(
    embed_queue: list[tuple[str, int]]
) -> tuple[list[str], list[int]]:
    texts = [item[0] for item in embed_queue]
    ids = [item[1] for item in embed_queue]
    return texts, ids


def apply_vector_removals(
    faiss_index: Index,
    remove_ids: list[int],
    index_root: Path,
    embed_queue: list[tuple[str, int]]
) -> None:
    remove_vectors(faiss_index, remove_ids)
    if not embed_queue and remove_ids:
        save_index(faiss_index, index_root)


def handle_file_batch(
    meta_db: Connection,
    faiss_index: Index,
    client: OpenAI,
    file_batch: list[Path],
    source_root: Path,
    index_root: Path,
    config: IndexConfig
) -> int:
    embed_queue, remove_ids = process_file_batch(
        meta_db,
        file_batch,
        source_root,
        config.chunk_size
    )
    if not embed_queue and not remove_ids: return 0
    apply_vector_removals(faiss_index, remove_ids, index_root, embed_queue)
    if not embed_queue:
        return 0
    texts, vector_ids = split_embed_queue(embed_queue)
    vectors = embed_text_batches(
        client=client,
        texts=texts,
        model=config.model,
        batch_size=config.embed_batch_size,
        delay=config.embed_batch_delay
    )
    added = add_vectors(faiss_index, vectors, vector_ids, index_root)
    mark_chunks_indexed(meta_db, vector_ids)
    return added


def run_index_batches(
    meta_db: Connection,
    faiss_index: Index,
    client: OpenAI,
    paths: list[Path],
    source_root: Path,
    index_root: Path,
    config: IndexConfig
) -> int:
    total_chunks = 0
    with tqdm(total=len(paths), desc="Processing files", unit="file") as pbar:
        for i in range(0, len(paths), config.file_batch_size):
            file_batch = paths[i:i + config.file_batch_size]
            total_chunks += handle_file_batch(
                meta_db=meta_db,
                faiss_index=faiss_index,
                client=client,
                file_batch=file_batch,
                source_root=source_root,
                index_root=index_root,
                config=config
            )
            pbar.update(len(file_batch))
    return total_chunks
