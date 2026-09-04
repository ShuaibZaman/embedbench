"""Cohere embeddings provider. Client is created only when embed is called."""

from __future__ import annotations

import os
from typing import Any, Literal

import numpy as np

from embedbench.providers.base import Embedder, approx_tokens, batched

# Cohere embed endpoints cap texts per request at 96.
COHERE_MAX_BATCH = 96


def _cohere_key_present() -> bool:
    return bool(
        os.environ.get("CO_API_KEY", "").strip()
        or os.environ.get("COHERE_API_KEY", "").strip()
    )


def _vectors_from_response(response: Any) -> list[list[float]]:
    embeddings = response.embeddings
    if hasattr(embeddings, "float_"):
        floats = embeddings.float_
        if floats is None:
            raise RuntimeError("Cohere response missing float embeddings")
        return list(floats)
    return list(embeddings)


def _tokens_from_response(response: Any) -> int | None:
    meta = getattr(response, "meta", None)
    if meta is None:
        return None
    billed = getattr(meta, "billed_units", None)
    if billed is not None and getattr(billed, "input_tokens", None):
        return int(billed.input_tokens)
    tokens = getattr(meta, "tokens", None)
    if tokens is not None and getattr(tokens, "input_tokens", None):
        return int(tokens.input_tokens)
    return None


class CohereEmbedder(Embedder):
    def __init__(
        self,
        *,
        id: str,
        model: str,
        price_per_1m_tokens: float = 0.0,
        query_prefix: str | None = None,
        normalize: bool = True,
        api_key_env: str = "COHERE_API_KEY",
        client: Any | None = None,
    ) -> None:
        super().__init__(
            id=id,
            model=model,
            price_per_1m_tokens=price_per_1m_tokens,
            api_key_env=api_key_env,
            query_prefix=query_prefix,
            normalize=normalize,
        )
        self._client = client

    def available(self) -> bool:
        return _cohere_key_present()

    def unavailable_reason(self) -> str | None:
        if self.available():
            return None
        return "missing environment variable COHERE_API_KEY or CO_API_KEY"

    def _get_client(self) -> Any:
        if self._client is None:
            import cohere

            self._client = cohere.Client()
        return self._client

    def embed_documents(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        return self._embed(texts, batch_size=batch_size, input_type="search_document")

    def embed_queries(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        return self._embed(
            self._prefixed(texts),
            batch_size=batch_size,
            input_type="search_query",
        )

    def _embed(
        self,
        texts: list[str],
        *,
        batch_size: int,
        input_type: Literal["search_document", "search_query"],
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        client = self._get_client()
        size = min(batch_size, COHERE_MAX_BATCH)
        rows: list[list[float]] = []
        for batch in batched(texts, size):
            sanitized = [text if text.strip() else " " for text in batch]
            response = client.embed(
                texts=sanitized,
                model=self.model,
                input_type=input_type,
                embedding_types=["float"],
            )
            rows.extend(_vectors_from_response(response))
            counted = _tokens_from_response(response)
            if counted:
                self._tokens_used += counted
            else:
                self._tokens_used += approx_tokens(sanitized)
        return self._maybe_normalize(np.asarray(rows, dtype=np.float32))
