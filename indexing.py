from pathlib import Path
from sqlite3 import Connection
from os import walk
import shutil
from faiss import Index, read_index, IndexFlatL2, write_index
from openai import OpenAI
from openai.types import CreateEmbeddingResponse
from config import Constants, EXCLUDE_PATTERNS, get_logger

LOGGER = get_logger()


def is_excluded(path: Path) -> bool:
    """Check if a name matches any of the exclude patterns."""
    return any(pattern.match(path.name) for pattern in EXCLUDE_PATTERNS)


def ensure_root(index_root: Path) -> Path:
    """Create index directory structure."""
    index_root.mkdir(parents=True, exist_ok=True)

    index_dir = index_root / Constants.INDEX.value
    index_dir.mkdir(parents=True, exist_ok=True)

    return index_dir


def erase_index(index_root: Path) -> None:
    """Delete existing index directory and all contents."""
    index_dir = index_root / Constants.INDEX.value
    if index_dir.exists():
        shutil.rmtree(index_dir)
        LOGGER.info(f"Erased existing index at {index_dir}")


def ensure_index(index_root: Path) -> Index:
    """Create or load FAISS index."""
    index_root = ensure_root(index_root)
    index_file = index_root / Constants.VECTORS.value
    index: Index | None = None

    if index_file.exists():
        index = read_index(str(index_file))
    else:
        index = IndexFlatL2(Constants.DIMENSIONS.value)

    return index


def save_index(index: Index, index_root: Path) -> None:
    """Save FAISS index to disk."""
    index_root = ensure_root(index_root)
    index_file = index_root / Constants.VECTORS.value
    write_index(index, str(index_file))


def collect_paths(source_root: Path) -> list[Path]:
    """Recursively collect all non-excluded file paths."""
    paths = []
    for dirpath, dirnames, filenames in walk(source_root):
        dirpath = Path(dirpath)
        dirnames[:] = [d for d in dirnames if not is_excluded(Path(d))]
        paths.extend(
            dirpath / filename
            for filename in filenames
            if not is_excluded(dirpath / filename)
        )
    return paths


def chunk_file(file_path: Path, content: str) -> list[dict]:
    """Split file content into overlapping chunks with metadata."""
    chunks = []
    chunk_size = Constants.CHUNK_SIZE.value
    overlap = Constants.CHUNK_OVERLAP.value

    if len(content) <= chunk_size:
        return [{
            'file_path': file_path,
            'chunk_index': 0,
            'content': content,
            'start_char': 0,
            'end_char': len(content)
        }]

    start = 0
    chunk_index = 0
    while start < len(content):
        end = min(start + chunk_size, len(content))
        chunks.append({
            'file_path': file_path,
            'chunk_index': chunk_index,
            'content': content[start:end],
            'start_char': start,
            'end_char': end
        })
        chunk_index += 1
        start += (chunk_size - overlap)
        if end == len(content):
            break

    return chunks


def process_file_batch(
    connection: Connection,
    paths: list[Path],
    source_root: Path
) -> list[tuple[str, int]]:
    """Process a batch of files: read, chunk, store metadata. Returns (prefixed_text, chunk_id) pairs."""
    cursor = connection.cursor()
    embed_queue = []

    for path in paths:
        try:
            content = path.read_text(encoding='utf-8', errors='ignore')
            file_stat = path.stat()
            relative_path = path.relative_to(source_root)
            chunks = chunk_file(relative_path, content)

            for chunk in chunks:
                cursor.execute("""
                    INSERT OR REPLACE INTO chunks
                    (file_path, chunk_index, file_size, modified_time, start_char, end_char)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    str(chunk['file_path']),
                    chunk['chunk_index'],
                    file_stat.st_size,
                    file_stat.st_mtime,
                    chunk['start_char'],
                    chunk['end_char']
                ))
                chunk_id = cursor.lastrowid
                prefixed_text = f"search_document: {chunk['content']}"
                embed_queue.append((prefixed_text, chunk_id))

        except Exception as e:
            LOGGER.warning(f"Failed to process {path}: {e}")
            continue

    connection.commit()
    return embed_queue


def generate_embeddings_batch(
    client: OpenAI,
    texts: list[str],
    model: str
) -> list[list[float]]:
    """Generate embeddings for a batch of texts."""
    response: CreateEmbeddingResponse = client.embeddings.create(
        input=texts,
        model=model,
        dimensions=Constants.DIMENSIONS.value
    )
    return [embedding.embedding for embedding in response.data]


def read_chunk_content(file_path: Path, start_char: int, end_char: int) -> str:
    """Read the content of a specific chunk from a file."""
    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        return content[start_char:end_char]
    except Exception as e:
        LOGGER.warning(f"Failed to read chunk from {file_path}: {e}")
        return ""
