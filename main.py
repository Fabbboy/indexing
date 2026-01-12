from typer import Typer
from pathlib import Path

from config import Constants
from cli_handlers import handle_ask, handle_index, handle_query

CWD = Path.cwd()
APP = Typer()


@APP.command()
def index(
    source_root: Path = CWD,
    index_root: Path = CWD,
    api_base: str = "http://localhost:11434/v1",
    api_key: str = "not-needed",
    model: str = Constants.MODEL.value,
    chunk_size: int = Constants.CHUNK_SIZE.value,
    file_batch_size: int = Constants.FILE_BATCH_SIZE.value,
    embed_batch_size: int = Constants.EMBED_BATCH_SIZE.value,
    embed_batch_delay: float = 0.0,
    erase: bool = False,
) -> None:
    handle_index(
        source_root=source_root,
        index_root=index_root,
        api_base=api_base,
        api_key=api_key,
        model=model,
        chunk_size=chunk_size,
        file_batch_size=file_batch_size,
        embed_batch_size=embed_batch_size,
        embed_batch_delay=embed_batch_delay,
        erase=erase,
    )


@APP.command()
def query(
    query_str: str,
    index_root: Path = CWD,
    api_base: str = "http://localhost:11434/v1",
    api_key: str = "not-needed",
    model: str = Constants.MODEL.value,
    limit: int = 5
) -> None:
    handle_query(
        query_str=query_str,
        index_root=index_root,
        api_base=api_base,
        api_key=api_key,
        model=model,
        limit=limit,
    )


@APP.command()
def ask(
    question: str,
    source_root: Path = CWD,
    index_root: Path = CWD,
    api_base: str = "http://localhost:11434/v1",
    api_key: str = "not-needed",
    embed_model: str = Constants.MODEL.value,
    ai_model: str = "llama3.2",
    limit: int = 5,
    tool_max_retries: int = 3
) -> None:
    handle_ask(
        question=question,
        source_root=source_root,
        index_root=index_root,
        api_base=api_base,
        api_key=api_key,
        embed_model=embed_model,
        ai_model=ai_model,
        limit=limit,
        tool_max_retries=tool_max_retries,
    )


if __name__ == "__main__":
    APP()
