from pathlib import Path
from typing import Any

from openai import OpenAI

from ai_utils import log_model_error
from indexing import ensure_index
from schemas import AskConfig
from assistant_prompt import build_messages, build_system_prompt, build_tools
from assistant_tools import get_tool_calls, run_tool_call


def index_ready(index_root: Path) -> bool:
    return ensure_index(index_root).ntotal > 0


def build_assistant_state(config: AskConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    system_prompt = build_system_prompt()
    tools = build_tools(config.limit)
    messages = build_messages(system_prompt, config.question)
    return tools, messages


def call_model(
    client: OpenAI,
    ai_model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]]
) -> Any:
    return client.chat.completions.create(
        model=ai_model,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )


def request_response(
    client: OpenAI,
    config: AskConfig,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]]
) -> tuple[Any | None, str | None]:
    try:
        return call_model(client, config.ai_model, messages, tools), None
    except Exception as exc:
        error = f"Failed to generate answer: {exc}"
        log_model_error(client, str(exc))
        return None, error


def append_tool_calls(
    messages: list[dict[str, Any]],
    assistant_message: Any,
    tool_calls: list[Any],
    client: OpenAI,
    config: AskConfig,
    source_root: Path,
    index_root: Path,
    max_retries: int
) -> None:
    messages.append(assistant_message.model_dump(exclude_none=True))
    for tool_call in tool_calls:
        messages.append(run_tool_call(
            tool_call=tool_call,
            client=client,
            config=config,
            source_root=source_root,
            index_root=index_root,
            max_retries=max_retries
        ))


def run_assistant_loop(
    client: OpenAI,
    config: AskConfig,
    source_root: Path,
    index_root: Path,
    max_turns: int = 8
) -> tuple[str | None, str | None]:
    if not index_ready(index_root):
        return None, "Index is empty. Run 'index' command first."
    tools, messages = build_assistant_state(config)
    for _ in range(max_turns):
        response, error = request_response(client, config, messages, tools)
        if error:
            return None, error
        assistant_message = response.choices[0].message
        tool_calls = get_tool_calls(assistant_message)
        if not tool_calls:
            return assistant_message.content or "", None
        append_tool_calls(
            messages=messages,
            assistant_message=assistant_message,
            tool_calls=tool_calls,
            client=client,
            config=config,
            source_root=source_root,
            index_root=index_root,
            max_retries=config.tool_max_retries
        )
    return None, "Failed to produce a final answer after tool calls."
