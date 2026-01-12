from pathlib import Path
import shutil

import numpy as np
from faiss import Index, IndexFlatL2, IndexIDMap2, read_index, write_index

from config import Constants, get_logger

LOGGER = get_logger()


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
    if index_file.exists():
        index = read_index(str(index_file))
    else:
        index = IndexIDMap2(IndexFlatL2(Constants.DIMENSIONS.value))
    if not hasattr(index, "add_with_ids"):
        index = upgrade_index_to_id_map(index)
    return index


def save_index(index: Index, index_root: Path) -> None:
    """Save FAISS index to disk."""
    index_root = ensure_root(index_root)
    index_file = index_root / Constants.VECTORS.value
    write_index(index, str(index_file))


def upgrade_index_to_id_map(index: Index) -> Index:
    if not hasattr(index, "reconstruct_n"):
        raise ValueError("Existing FAISS index cannot be upgraded to ID map.")
    vectors = index.reconstruct_n(0, index.ntotal)
    id_map = IndexIDMap2(IndexFlatL2(Constants.DIMENSIONS.value))
    ids = np.arange(1, index.ntotal + 1, dtype="int64")
    id_map.add_with_ids(vectors, ids)
    return id_map
