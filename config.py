from enum import Enum
from logging import getLogger, INFO, Logger
from rich.logging import RichHandler
import re

MIGRATION = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    file_size INTEGER NOT NULL,
    modified_time REAL NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    UNIQUE(file_path, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_file_path ON chunks(file_path);
"""


class Constants(Enum):
    INDEX = ".index"
    META = "metadata.db"
    DIMENSIONS = 256
    VECTORS = "index.faiss"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MODEL = "nomic-embed-text"
    FILE_BATCH_SIZE = 50
    EMBED_BATCH_SIZE = 100


EXCLUDES = [
    r"^\.index$",
    r"^\.git$",
    r"^__pycache__$",
    r".*build.*",
    r".*dist.*",
    r".*(^|\.|_|-)out(put)?(s)?(\.|_|-|$).*",
    r".*cache.*",
    r"^\.venv$",
    r"^venv$",
    r"^target$",
    r"^\.idea$",
    r"^\.vscode$",
    r"^node_modules$",
    r"^\.DS_Store$",
    r".*\.lock$",
    r".*-lock\..*$",
]
EXCLUDE_PATTERNS = [re.compile(pattern) for pattern in EXCLUDES]


def get_logger() -> Logger:
    """Get configured logger with RichHandler."""
    logger = getLogger(__package__)
    logger.setLevel(INFO)
    logger.addHandler(
        RichHandler(
            show_time=False,
        )
    )
    return logger
