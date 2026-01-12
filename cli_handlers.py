from pathlib import Path
from typing import Any

from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown

from ai_utils import connect_client, log_model_error
from assistant_loop import run_assistant_loop
from config import get_logger
from indexer import run_indexing
from indexing import erase_index, is_excluded
from schemas import (
    AskConfig,
    QueryConfig,
    build_ask_config,
    build_index_config,
    build_query_config,
)
from searching import search_index

LOGGER = get_logger()
CONSOLE = Console()


def log_query_results(results: list[dict[str, Any]]) -> None:
    LOGGER.info(f"\nTop {len(results)} results:")
    for i, result in enumerate(results, 1):
        similarity_pct = result['similarity'] * 100
        LOGGER.info(
            f"{i}. {result['file_path']} "
            f"(chunk {result['chunk_index']}, chars {result['start_char']}-{result['end_char']}) "
            f"- {similarity_pct:.1f}% similar"
        )


def print_answer(question: str, answer: str) -> None:
    CONSOLE.print("\n" + "=" * 80)
    CONSOLE.print(f"[bold cyan]Question:[/bold cyan] {question}")
    CONSOLE.print("=" * 80 + "\n")
    CONSOLE.print(Markdown(answer))
    CONSOLE.print("\n" + "=" * 80)


def handle_index(
    source_root: Path,
    index_root: Path,
    api_base: str,
    api_key: str,
    model: str,
    chunk_size: int,
    file_batch_size: int,
    embed_batch_size: int,
    embed_batch_delay: float,
    erase: bool
) -> None:
    if is_excluded(source_root):
        LOGGER.error(f"Source root {source_root} is in the excludes list.")
        return
    config = build_index_config(
        file_batch_size=file_batch_size,
        embed_batch_size=embed_batch_size,
        embed_batch_delay=embed_batch_delay,
        chunk_size=chunk_size,
        api_base=api_base,
        api_key=api_key,
        model=model,
    )
    if not config:
        return
    if erase:
        erase_index(index_root)
    client = connect_client(config.api_base, config.api_key)
    run_indexing(source_root, index_root, config, client, erase)


def build_query_context(
    query_str: str,
    api_base: str,
    api_key: str,
    model: str,
    limit: int
) -> tuple[QueryConfig | None, OpenAI | None]:
    config = build_query_config(
        api_base=api_base,
        api_key=api_key,
        model=model,
        limit=limit,
        query_str=query_str,
    )
    if not config:
        return None, None
    client = connect_client(config.api_base, config.api_key)
    return config, client


def run_query_search(
    config: QueryConfig,
    client: OpenAI,
    index_root: Path
) -> tuple[list[dict[str, Any]], str | None]:
    LOGGER.info(f"Querying '{config.query_str}' against index at {index_root}")
    return search_index(
        query_str=config.query_str,
        source_root=Path.cwd(),
        index_root=index_root,
        client=client,
        model=config.model,
        limit=config.limit,
        include_content=False,
        include_metadata=True
    )


def handle_query(
    query_str: str,
    index_root: Path,
    api_base: str,
    api_key: str,
    model: str,
    limit: int
) -> None:
    config, client = build_query_context(query_str, api_base, api_key, model, limit)
    if not config or not client:
        return
    results, error = run_query_search(config, client, index_root)
    if error:
        LOGGER.error(error)
        log_model_error(client, error)
        return
    log_query_results(results)


def build_ask_context(
    question: str,
    api_base: str,
    api_key: str,
    embed_model: str,
    ai_model: str,
    limit: int,
    tool_max_retries: int
) -> tuple[AskConfig | None, OpenAI | None]:
    config = build_ask_config(
        api_base=api_base,
        api_key=api_key,
        embed_model=embed_model,
        ai_model=ai_model,
        limit=limit,
        tool_max_retries=tool_max_retries,
        question=question,
    )
    if not config:
        return None, None
    client = connect_client(config.api_base, config.api_key)
    return config, client


def execute_ask(
    client: OpenAI,
    config: AskConfig,
    source_root: Path,
    index_root: Path
) -> tuple[str | None, str | None]:
    LOGGER.info(f"Generating answer using {config.ai_model}...")
    return run_assistant_loop(
        client=client,
        config=config,
        source_root=source_root,
        index_root=index_root
    )


def handle_ask(
    question: str,
    source_root: Path,
    index_root: Path,
    api_base: str,
    api_key: str,
    embed_model: str,
    ai_model: str,
    limit: int,
    tool_max_retries: int
) -> None:
    config, client = build_ask_context(
        question,
        api_base,
        api_key,
        embed_model,
        ai_model,
        limit,
        tool_max_retries
    )
    if not config or not client:
        return
    answer, error = execute_ask(client, config, source_root, index_root)
    if error:
        LOGGER.error(error)
        return
    print_answer(question, answer)
