from openai import OpenAI
from openai.types import CreateEmbeddingResponse

from config import Constants


def generate_embeddings_batch(
    client: OpenAI,
    texts: list[str],
    model: str
) -> list[list[float]]:
    """Generate embeddings for a batch of texts."""
    response: CreateEmbeddingResponse = client.embeddings.create(
        input=texts,
        model=model,
        dimensions=Constants.DIMENSIONS.value
    )
    return [embedding.embedding for embedding in response.data]
