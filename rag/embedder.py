"""Embedding client — calls the local embedding server at :8081."""

from openai import OpenAI
from config import EMBED_BASE_URL, EMBED_MODEL

_client = OpenAI(base_url=EMBED_BASE_URL, api_key="not-needed")


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, return a list of vectors."""
    response = _client.embeddings.create(input=texts, model=EMBED_MODEL)
    return [item.embedding for item in response.data]


def embed_one(text: str) -> list[float]:
    """Embed a single text, return one vector."""
    return embed([text])[0]
