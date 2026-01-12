from pathlib import Path
from typing import Any

from config import Constants


def build_chunk(
    file_path: Path,
    content: str,
    chunk_index: int,
    start: int,
    end: int
) -> dict[str, Any]:
    return {
        'file_path': file_path,
        'chunk_index': chunk_index,
        'content': content[start:end],
        'start_char': start,
        'end_char': end
    }


def chunk_file(
    file_path: Path,
    content: str,
    chunk_size: int | None = None
) -> list[dict[str, Any]]:
    """Split file content into overlapping chunks with metadata."""
    if chunk_size is None:
        chunk_size = Constants.CHUNK_SIZE.value
    overlap = Constants.CHUNK_OVERLAP.value

    if len(content) <= chunk_size:
        return [build_chunk(file_path, content, 0, 0, len(content))]

    start = 0
    chunk_index = 0
    chunks = []
    while start < len(content):
        end = min(start + chunk_size, len(content))
        chunks.append(build_chunk(file_path, content, chunk_index, start, end))
        chunk_index += 1
        start += (chunk_size - overlap)
        if end == len(content):
            break

    return chunks
