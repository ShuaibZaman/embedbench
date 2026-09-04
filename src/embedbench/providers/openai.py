"""OpenAI embeddings provider. Never constructed client unless embed is called."""

from __future__ import annotations

from typing import Any

import numpy as np

from embedbench.providers.base import Embedder, approx_tokens, batched


class OpenAIEmbedder(Embedder):
    def __init__(
        self,
        *,
        id: str,
        model: str,
        price_per_1m_tokens: float = 0.0,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
        normalize: bool = True,
        api_key_env: str = "OPENAI_API_KEY",
        client: Any | None = None,
    ) -> None:
        super().__init__(
            id=id,
            model=model,
            price_per_1m_tokens=price_per_1m_tokens,
            api_key_env=api_key_env,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            normalize=normalize,
        )
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def embed_documents(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        return self._embed(texts, batch_size=batch_size)

    def embed_queries(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        return self._embed(self._prefixed(texts), batch_size=batch_size)

    def _embed(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        client = self._get_client()
        rows: list[list[float]] = []
        for batch in batched(texts, batch_size):
            sanitized = [text if text.strip() else " " for text in batch]
            response = client.embeddings.create(
                model=self.model,
                input=sanitized,
                encoding_format="float",
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            rows.extend(item.embedding for item in ordered)
            usage = getattr(response, "usage", None)
            total = getattr(usage, "total_tokens", None) if usage is not None else None
            if total:
                self._tokens_used += int(total)
            else:
                self._tokens_used += approx_tokens(sanitized)
        return self._maybe_normalize(np.asarray(rows, dtype=np.float32))
