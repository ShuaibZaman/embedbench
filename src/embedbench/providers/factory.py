"""Map YAML ``provider:`` values to embedder classes."""

from __future__ import annotations

from typing import Any

from embedbench.providers.base import Embedder
from embedbench.providers.cohere import CohereEmbedder
from embedbench.providers.local_st import SentenceTransformerEmbedder
from embedbench.providers.openai import OpenAIEmbedder
from embedbench.providers.voyage import VoyageEmbedder

PROVIDERS: dict[str, type[Embedder]] = {
    "openai": OpenAIEmbedder,
    "voyage": VoyageEmbedder,
    "cohere": CohereEmbedder,
    "sentence_transformers": SentenceTransformerEmbedder,
}


def create_embedder(spec: dict[str, Any]) -> Embedder:
    try:
        provider = spec["provider"]
        model_id = spec["id"]
        model = spec["model"]
    except KeyError as exc:
        raise ValueError(f"model spec missing required field: {exc}") from exc

    cls = PROVIDERS.get(provider)
    if cls is None:
        known = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"unknown provider {provider!r}; known: {known}")

    return cls(
        id=model_id,
        model=model,
        price_per_1m_tokens=float(spec.get("price_per_1m_tokens") or 0),
        query_prefix=spec.get("query_prefix"),
        document_prefix=spec.get("document_prefix"),
        normalize=bool(spec.get("normalize", True)),
    )
