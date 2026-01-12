from pathlib import Path
from os import walk

from config import EXCLUDE_PATTERNS


def is_excluded(path: Path) -> bool:
    """Check if a name matches any of the exclude patterns."""
    return any(pattern.match(path.name) for pattern in EXCLUDE_PATTERNS)


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
