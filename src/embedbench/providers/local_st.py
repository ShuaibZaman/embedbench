"""Local sentence-transformers provider. Model weights load on first embed."""

from __future__ import annotations

from typing import Any

import numpy as np

from embedbench.providers.base import Embedder, whitespace_tokens


class SentenceTransformerEmbedder(Embedder):
    def __init__(
        self,
        *,
        id: str,
        model: str,
        price_per_1m_tokens: float = 0.0,
        query_prefix: str | None = None,
        normalize: bool = True,
        model_obj: Any | None = None,
    ) -> None:
        super().__init__(
            id=id,
            model=model,
            price_per_1m_tokens=price_per_1m_tokens,
            api_key_env=None,
            query_prefix=query_prefix,
            normalize=normalize,
        )
        self._model = model_obj

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model)
        return self._model

    def embed_documents(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        return self._embed(texts, batch_size=batch_size)

    def embed_queries(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        return self._embed(self._prefixed(texts), batch_size=batch_size)

    def _embed(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        model = self._get_model()
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        self._tokens_used += whitespace_tokens(texts)
        return np.asarray(vectors, dtype=np.float32)
