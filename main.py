from typer import Typer
from pathlib import Path
from openai import OpenAI
import numpy as np
from tqdm import tqdm
from rich.console import Console
from rich.markdown import Markdown
import time

from config import Constants, get_logger
from database import ensure_db
from indexing import (
    is_excluded,
    erase_index,
    ensure_index,
    save_index,
    collect_paths,
    process_file_batch,
    generate_embeddings_batch,
    ensure_root,
    read_chunk_content,
    get_indexed_files,
)

CWD = Path.cwd()
APP = Typer()
LOGGER = get_logger()
CONSOLE = Console()


def connect_client(api: str, key: str) -> OpenAI:
    """Create OpenAI client."""
    return OpenAI(base_url=api, api_key=key)


def list_available_models(client: OpenAI) -> None:
    """List all available models from the API."""
    try:
        models = client.models.list()
        LOGGER.info("Available models:")
        for model in models.data:
            LOGGER.info(f"  - {model.id}")
    except Exception as e:
        LOGGER.error(f"Failed to list models: {e}")


@APP.command()
def index(
    source_root: Path = CWD,
    index_root: Path = CWD,
    api_base: str = "http://localhost:11434/v1",
    api_key: str = "not-needed",
    model: str = Constants.MODEL.value,
    chunk_size: int = Constants.CHUNK_SIZE.value,
    erase: bool = False,
) -> None:
    """Index source files into vector database with streaming to avoid RAM exhaustion."""
    start_time = time.time()

    if is_excluded(source_root):
        LOGGER.error(f"Source root {source_root} is in the excludes list.")
        return

    if erase:
        erase_index(index_root)

    meta_db = ensure_db(index_root)
    faiss_index = ensure_index(index_root)
    ai = connect_client(api_base, api_key)

    LOGGER.info(f"Indexing files from {source_root} into index at {index_root}")

    paths = collect_paths(source_root)

    if not erase:
        indexed_files = get_indexed_files(meta_db)
        paths = [p for p in paths if str(p.relative_to(source_root)) not in indexed_files]
        if len(indexed_files) > 0:
            LOGGER.info(f"Found {len(indexed_files)} already indexed files, skipping them.")

    LOGGER.info(f"Collected {len(paths)} files to index.")

    if len(paths) == 0:
        LOGGER.warning("No new files to index.")
        return

    file_batch_size = Constants.FILE_BATCH_SIZE.value
    embed_batch_size = Constants.EMBED_BATCH_SIZE.value
    total_chunks = 0

    with tqdm(total=len(paths), desc="Processing files", unit="file") as pbar:
        for i in range(0, len(paths), file_batch_size):
            file_batch = paths[i:i+file_batch_size]

            embed_queue = process_file_batch(meta_db, file_batch, source_root, chunk_size)

            if len(embed_queue) == 0:
                pbar.update(len(file_batch))
                continue

            vectors = []
            texts = [item[0] for item in embed_queue]

            with tqdm(total=len(texts), desc=f"  Embedding batch", unit="chunk", leave=False) as embed_pbar:
                for j in range(0, len(texts), embed_batch_size):
                    text_batch = texts[j:j+embed_batch_size]
                    try:
                        batch_vectors = generate_embeddings_batch(ai, text_batch, model)
                        vectors.extend(batch_vectors)
                        embed_pbar.update(len(text_batch))
                    except Exception as e:
                        LOGGER.error(f"Failed to generate embeddings: {e}")
                        if "not found" in str(e).lower():
                            list_available_models(ai)
                        embed_pbar.update(len(text_batch))
                        continue

            if len(vectors) > 0:
                vector_array = np.array(vectors, dtype='float32')
                faiss_index.add(vector_array)
                total_chunks += len(vectors)

                save_index(faiss_index, index_root)

            del embed_queue
            del vectors
            del texts

            pbar.update(len(file_batch))

    meta_db.close()

    elapsed = time.time() - start_time
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    LOGGER.info(f"✓ Indexing complete! Indexed {total_chunks} chunks from {len(paths)} files in {elapsed_str}.")


@APP.command()
def query(
    query_str: str,
    index_root: Path = CWD,
    api_base: str = "http://localhost:11434/v1",
    api_key: str = "not-needed",
    model: str = Constants.MODEL.value,
    limit: int = 5
) -> None:
    """Query the index and return similar code chunks."""
    faiss_index = ensure_index(index_root)

    if faiss_index.ntotal == 0:
        LOGGER.error("Index is empty. Run 'index' command first.")
        return

    meta_db = ensure_db(index_root)
    ai = connect_client(api_base, api_key)

    LOGGER.info(f"Querying '{query_str}' against index at {index_root}")

    prefixed_query = f"search_query: {query_str}"
    try:
        response = ai.embeddings.create(
            input=[prefixed_query],
            model=model,
            dimensions=Constants.DIMENSIONS.value
        )
        query_vector = response.data[0].embedding
    except Exception as e:
        LOGGER.error(f"Failed to generate query embedding: {e}")
        if "not found" in str(e).lower():
            list_available_models(ai)
        return

    query_array = np.array([query_vector], dtype='float32')
    distances, indices = faiss_index.search(query_array, limit)

    cursor = meta_db.cursor()
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        cursor.execute("""
            SELECT file_path, chunk_index, start_char, end_char
            FROM chunks
            WHERE id = ?
        """, (int(idx) + 1,))

        row = cursor.fetchone()
        if row:
            results.append({
                'file_path': row[0],
                'chunk_index': row[1],
                'start_char': row[2],
                'end_char': row[3],
                'distance': float(distance),
                'similarity': 1.0 / (1.0 + float(distance))
            })

    LOGGER.info(f"\nTop {len(results)} results:")
    for i, result in enumerate(results, 1):
        similarity_pct = result['similarity'] * 100
        LOGGER.info(
            f"{i}. {result['file_path']} "
            f"(chunk {result['chunk_index']}, chars {result['start_char']}-{result['end_char']}) "
            f"- {similarity_pct:.1f}% similar"
        )

    meta_db.close()


@APP.command()
def ask(
    question: str,
    source_root: Path = CWD,
    index_root: Path = CWD,
    api_base: str = "http://localhost:11434/v1",
    api_key: str = "not-needed",
    embed_model: str = Constants.MODEL.value,
    ai_model: str = "llama3.2",
    limit: int = 5
) -> None:
    """Ask a question and get an AI-generated answer based on code context."""
    faiss_index = ensure_index(index_root)

    if faiss_index.ntotal == 0:
        LOGGER.error("Index is empty. Run 'index' command first.")
        return

    meta_db = ensure_db(index_root)
    ai = connect_client(api_base, api_key)

    LOGGER.info(f"Searching for relevant code for: '{question}'")

    prefixed_query = f"search_query: {question}"
    try:
        response = ai.embeddings.create(
            input=[prefixed_query],
            model=embed_model,
            dimensions=Constants.DIMENSIONS.value
        )
        query_vector = response.data[0].embedding
    except Exception as e:
        LOGGER.error(f"Failed to generate query embedding: {e}")
        if "not found" in str(e).lower():
            list_available_models(ai)
        meta_db.close()
        return

    query_array = np.array([query_vector], dtype='float32')
    distances, indices = faiss_index.search(query_array, limit)

    cursor = meta_db.cursor()
    context_parts = []

    for idx, distance in zip(indices[0], distances[0]):
        cursor.execute("""
            SELECT file_path, chunk_index, start_char, end_char
            FROM chunks
            WHERE id = ?
        """, (int(idx) + 1,))

        row = cursor.fetchone()
        if row:
            file_path = source_root / row[0]
            chunk_content = read_chunk_content(file_path, row[2], row[3])
            if chunk_content:
                context_parts.append(f"# From {row[0]} (chunk {row[1]}):\n```\n{chunk_content}\n```")

    if not context_parts:
        LOGGER.error("No relevant code found.")
        meta_db.close()
        return

    context = "\n\n".join(context_parts)
    prompt = f"""You are a helpful programming assistant. Answer the question based on the provided code context.

Code Context:
{context}

Question: {question}

Answer:"""

    LOGGER.info(f"Generating answer using {ai_model}...")

    try:
        chat_response = ai.chat.completions.create(
            model=ai_model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        answer = chat_response.choices[0].message.content

        CONSOLE.print("\n" + "="*80)
        CONSOLE.print(f"[bold cyan]Question:[/bold cyan] {question}")
        CONSOLE.print("="*80 + "\n")
        CONSOLE.print(Markdown(answer))
        CONSOLE.print("\n" + "="*80)

    except Exception as e:
        LOGGER.error(f"Failed to generate answer: {e}")
        if "not found" in str(e).lower():
            list_available_models(ai)

    meta_db.close()


if __name__ == "__main__":
    APP()
