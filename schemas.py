from pydantic import BaseModel, PositiveInt, NonNegativeFloat, ValidationError, ConfigDict

from config import get_logger

LOGGER = get_logger()


class IndexConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_batch_size: PositiveInt
    embed_batch_size: PositiveInt
    embed_batch_delay: NonNegativeFloat
    chunk_size: PositiveInt
    api_base: str
    api_key: str
    model: str


class QueryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_base: str
    api_key: str
    model: str
    limit: PositiveInt
    query_str: str


class AskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_base: str
    api_key: str
    embed_model: str
    ai_model: str
    limit: PositiveInt
    tool_max_retries: PositiveInt
    question: str


def build_index_config(
    file_batch_size: int,
    embed_batch_size: int,
    embed_batch_delay: float,
    chunk_size: int,
    api_base: str,
    api_key: str,
    model: str
) -> IndexConfig | None:
    try:
        return IndexConfig(
            file_batch_size=file_batch_size,
            embed_batch_size=embed_batch_size,
            embed_batch_delay=embed_batch_delay,
            chunk_size=chunk_size,
            api_base=api_base,
            api_key=api_key,
            model=model,
        )
    except ValidationError as exc:
        LOGGER.error(f"Invalid index options: {exc}")
        return None


def build_query_config(
    api_base: str,
    api_key: str,
    model: str,
    limit: int,
    query_str: str
) -> QueryConfig | None:
    try:
        return QueryConfig(
            api_base=api_base,
            api_key=api_key,
            model=model,
            limit=limit,
            query_str=query_str,
        )
    except ValidationError as exc:
        LOGGER.error(f"Invalid query options: {exc}")
        return None


def build_ask_config(
    api_base: str,
    api_key: str,
    embed_model: str,
    ai_model: str,
    limit: int,
    tool_max_retries: int,
    question: str
) -> AskConfig | None:
    try:
        return AskConfig(
            api_base=api_base,
            api_key=api_key,
            embed_model=embed_model,
            ai_model=ai_model,
            limit=limit,
            tool_max_retries=tool_max_retries,
            question=question,
        )
    except ValidationError as exc:
        LOGGER.error(f"Invalid ask options: {exc}")
        return None
