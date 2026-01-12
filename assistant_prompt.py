from typing import Any


THINK_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "think",
        "description": "Use this to think step-by-step or outline a plan before answering.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Your internal reasoning or plan."
                }
            },
            "required": ["thought"]
        }
    }
}


def build_system_prompt() -> str:
    return (
        "You are a direct, precise coding assistant. "
        "Use the available tools to gather evidence before answering. "
        "Use `query` to search the indexed codebase and `think` to plan or refine your approach. "
        "You may call one or more tools in a tight loop until you have enough evidence. "
        "Do not answer with minimal information if more tool calls would help. "
        "Only answer using evidence from tool results; if evidence is missing, say you cannot answer. "
        "The `query` tool returns authoritative project truth; rely on it over assumptions."
    )


def build_query_tool(limit: int) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "query",
            "description": "Search the indexed codebase for relevant chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query describing what to find in the code."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                        "default": limit
                    },
                    "context_chars": {
                        "type": "integer",
                        "description": "Extra characters to include before and after each match.",
                        "default": 160
                    }
                },
                "required": ["query"]
            }
        }
    }


def build_tools(limit: int) -> list[dict[str, Any]]:
    return [THINK_TOOL, build_query_tool(limit)]


def build_messages(system_prompt: str, question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
