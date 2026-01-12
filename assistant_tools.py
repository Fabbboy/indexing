import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from ai_utils import log_model_error
from config import get_logger
from searching import search_index
from schemas import AskConfig

LOGGER = get_logger()


def get_tool_calls(message: Any) -> list[Any]:
    return message.tool_calls or []


def parse_tool_arguments(tool_call: Any) -> dict[str, Any]:
    args = getattr(tool_call.function, "arguments", None)
    if isinstance(args, dict):
        return args
    parsed = getattr(tool_call.function, "parsed_arguments", None)
    if isinstance(parsed, dict):
        return parsed
    if isinstance(args, str):
        return parse_json_arguments(args)
    LOGGER.error("Tool arguments are not structured; update the SDK to use built-in parsing.")
    return {}


def log_tool_call(tool_name: str, tool_args: dict[str, Any]) -> None:
    LOGGER.info(f"Tool call: {tool_name} args={tool_args}")


def log_query_result_count(query: str, count: int) -> None:
    LOGGER.info(f"Tool result: query='{query}' matches={count}")


def format_tool_message(tool_call_id: str, content: str) -> dict[str, str]:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def parse_json_arguments(arguments: str) -> dict[str, Any]:
    try:
        return json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        LOGGER.error("Tool arguments are not valid JSON.")
        return {}


def read_query_args(tool_args: dict[str, Any], default_limit: int) -> tuple[str, int, int]:
    query_text = tool_args.get("query", "")
    query_limit = int(tool_args.get("limit", default_limit))
    context_chars = int(tool_args.get("context_chars", 160))
    return query_text, query_limit, context_chars


def build_query_payload(
    query_text: str,
    results: list[dict[str, Any]],
    context_chars: int
) -> dict[str, Any]:
    return {
        "query": query_text,
        "results": results,
        "match_count": len(results),
        "context_chars": context_chars
    }


def run_query_tool(
    tool_args: dict[str, Any],
    client: OpenAI,
    config: AskConfig,
    source_root: Path,
    index_root: Path
) -> tuple[str, bool]:
    query_text, query_limit, context_chars = read_query_args(tool_args, config.limit)
    results, error = search_index(
        query_str=query_text,
        source_root=source_root,
        index_root=index_root,
        client=client,
        model=config.embed_model,
        limit=query_limit,
        include_content=True,
        context_chars=context_chars,
        include_metadata=False
    )
    log_query_result_count(query_text, len(results))
    payload = build_query_payload(query_text, results, context_chars)
    if error:
        payload["error"] = error
        log_model_error(client, error)
        return json.dumps(payload), True
    return json.dumps(payload), False


def run_think_tool(tool_args: dict[str, Any]) -> tuple[str, bool]:
    thought = tool_args.get("thought", "")
    return json.dumps({"thought": thought}), False


def build_tool_output(
    tool_name: str,
    tool_args: dict[str, Any],
    client: OpenAI,
    config: AskConfig,
    source_root: Path,
    index_root: Path
) -> tuple[str, bool]:
    if tool_name == "query":
        return run_query_tool(tool_args, client, config, source_root, index_root)
    if tool_name == "think":
        return run_think_tool(tool_args)
    return json.dumps({"error": f"Unknown tool: {tool_name}"}), True


def should_retry_tool(failed: bool, attempt: int, max_retries: int) -> bool:
    return failed and attempt < max_retries


def run_tool_call(
    tool_call: Any,
    client: OpenAI,
    config: AskConfig,
    source_root: Path,
    index_root: Path,
    max_retries: int
) -> dict[str, str]:
    tool_args = parse_tool_arguments(tool_call)
    tool_name = tool_call.function.name
    attempt = 0
    while True:
        log_tool_call(tool_name, tool_args)
        output, failed = build_tool_output(
            tool_name=tool_name,
            tool_args=tool_args,
            client=client,
            config=config,
            source_root=source_root,
            index_root=index_root
        )
        if not should_retry_tool(failed, attempt, max_retries):
            return format_tool_message(tool_call.id, output)
        attempt += 1
        LOGGER.warning(f"Retrying tool call {tool_name}, attempt {attempt + 1}/{max_retries + 1}")
