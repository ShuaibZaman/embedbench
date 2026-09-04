"""Abstract embedding provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
import os

import numpy as np

EPS = 1e-12


def l2_normalize(vectors: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Row-wise L2 normalize. Empty input is returned unchanged."""
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def batched(items: list[str], size: int) -> list[list[str]]:
    if size < 1:
        raise ValueError("batch_size must be >= 1")
    return [items[i : i + size] for i in range(0, len(items), size)]


def approx_tokens(texts: list[str]) -> int:
    """Fallback token estimate used when a provider does not return usage."""
    return sum(max(1, len(text) // 4) for text in texts)


def whitespace_tokens(texts: list[str]) -> int:
    """Whitespace token count for local models (reporting only, not billed)."""
    return sum(max(1, len(text.split())) for text in texts)


class Embedder(ABC):
    """One embedding model. Subclasses must not call the network in ``__init__``."""

    def __init__(
        self,
        *,
        id: str,
        model: str,
        price_per_1m_tokens: float = 0.0,
        api_key_env: str | None = None,
        query_prefix: str | None = None,
        normalize: bool = True,
    ) -> None:
        self.id = id
        self.model = model
        self.price_per_1m_tokens = price_per_1m_tokens
        self.api_key_env = api_key_env
        self.query_prefix = query_prefix
        self.normalize = normalize
        self._tokens_used = 0

    @property
    def requires_api_key(self) -> bool:
        return self.api_key_env is not None

    def available(self) -> bool:
        if not self.requires_api_key:
            return True
        key = os.environ.get(self.api_key_env or "", "")
        return bool(key.strip())

    def unavailable_reason(self) -> str | None:
        if self.available():
            return None
        return f"missing environment variable {self.api_key_env}"

    @property
    def tokens_used(self) -> int:
        return self._tokens_used

    def reset_usage(self) -> None:
        self._tokens_used = 0

    def _prefixed(self, texts: list[str]) -> list[str]:
        if not self.query_prefix:
            return texts
        prefix = self.query_prefix
        return [f"{prefix}{text}" for text in texts]

    def _maybe_normalize(self, vectors: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return np.asarray(vectors, dtype=np.float32)
        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    @abstractmethod
    def embed_documents(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        """Return (n, d) document embeddings."""

    @abstractmethod
    def embed_queries(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        """Return (n, d) query embeddings (may apply a query prefix)."""
