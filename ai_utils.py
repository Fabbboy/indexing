from openai import OpenAI

from config import get_logger

LOGGER = get_logger()


def connect_client(api: str, key: str) -> OpenAI:
    """Create OpenAI client."""
    return OpenAI(base_url=api, api_key=key)


def list_available_models(client: OpenAI) -> None:
    """List all available models from the API."""
    try:
        models = client.models.list()
        LOGGER.info("Available models:")
        for model in models.data:
            LOGGER.info(f"  - {model.id}")
    except Exception as exc:
        LOGGER.error(f"Failed to list models: {exc}")


def log_model_error(client: OpenAI, error: str) -> None:
    if "not found" in error.lower():
        list_available_models(client)
