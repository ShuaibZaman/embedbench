"""Provider contract tests. No network, no model downloads."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from embedbench.providers.base import l2_normalize
from embedbench.providers.cohere import CohereEmbedder
from embedbench.providers.factory import create_embedder
from embedbench.providers.local_st import SentenceTransformerEmbedder
from embedbench.providers.openai import OpenAIEmbedder
from embedbench.providers.voyage import VoyageEmbedder


def test_l2_normalize_unit_rows() -> None:
    raw = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    out = l2_normalize(raw)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)


def test_openai_unavailable_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    embedder = OpenAIEmbedder(id="oai", model="text-embedding-3-small")
    assert embedder.requires_api_key
    assert not embedder.available()
    assert embedder.unavailable_reason() == "missing environment variable OPENAI_API_KEY"


def test_openai_available_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    embedder = OpenAIEmbedder(id="oai", model="text-embedding-3-small")
    assert embedder.available()
    assert embedder.unavailable_reason() is None


class _FakeEmbeddings:
    def create(self, *, model: str, input: list[str], encoding_format: str) -> SimpleNamespace:
        assert encoding_format == "float"
        data = [
            SimpleNamespace(index=i, embedding=[1.0, 0.0] if i == 0 else [0.0, 1.0])
            for i in range(len(input))
        ]
        return SimpleNamespace(
            data=data,
            usage=SimpleNamespace(total_tokens=12),
        )


def test_openai_embed_uses_mock_client_and_counts_tokens() -> None:
    embedder = OpenAIEmbedder(
        id="oai",
        model="text-embedding-3-small",
        client=SimpleNamespace(embeddings=_FakeEmbeddings()),
    )
    docs = embedder.embed_documents(["alpha", "beta"], batch_size=8)
    assert docs.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(docs, axis=1), 1.0, rtol=1e-5)
    assert embedder.tokens_used == 12


def test_sentence_transformers_always_available() -> None:
    embedder = SentenceTransformerEmbedder(id="minilm", model="stub")
    assert not embedder.requires_api_key
    assert embedder.available()


class _FakeST:
    def encode(self, texts, **kwargs):
        return np.eye(len(texts), 3, dtype=np.float32)


def test_sentence_transformers_query_prefix_and_usage() -> None:
    seen: list[list[str]] = []

    class Capture(_FakeST):
        def encode(self, texts, **kwargs):
            seen.append(list(texts))
            return super().encode(texts, **kwargs)

    embedder = SentenceTransformerEmbedder(
        id="bge",
        model="stub",
        query_prefix="query: ",
        model_obj=Capture(),
    )
    queries = embedder.embed_queries(["hello"], batch_size=4)
    assert seen == [["query: hello"]]
    assert queries.shape == (1, 3)
    assert embedder.tokens_used == 2  # whitespace tokens of the prefixed query


def test_factory_openai_and_st() -> None:
    oai = create_embedder(
        {
            "id": "openai-3-small",
            "provider": "openai",
            "model": "text-embedding-3-small",
            "price_per_1m_tokens": 0.02,
        }
    )
    assert isinstance(oai, OpenAIEmbedder)
    st = create_embedder(
        {
            "id": "minilm",
            "provider": "sentence_transformers",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
        }
    )
    assert isinstance(st, SentenceTransformerEmbedder)


def test_factory_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        create_embedder({"id": "x", "provider": "nope", "model": "x"})


def test_voyage_unavailable_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    embedder = VoyageEmbedder(id="voyage", model="voyage-3-lite")
    assert embedder.requires_api_key
    assert not embedder.available()
    assert "VOYAGE_API_KEY" in (embedder.unavailable_reason() or "")


def test_cohere_unavailable_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("CO_API_KEY", raising=False)
    embedder = CohereEmbedder(id="cohere", model="embed-english-v3.0")
    assert not embedder.available()


def test_cohere_available_with_co_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setenv("CO_API_KEY", "co-test")
    embedder = CohereEmbedder(id="cohere", model="embed-english-v3.0")
    assert embedder.available()


def test_voyage_embed_uses_query_input_type() -> None:
    seen: list[dict] = []

    class FakeVoyage:
        def embed(self, texts, model, input_type):
            seen.append({"texts": list(texts), "model": model, "input_type": input_type})
            return SimpleNamespace(embeddings=[[0.0, 1.0] for _ in texts], total_tokens=9)

    embedder = VoyageEmbedder(id="voyage", model="voyage-3-lite", client=FakeVoyage())
    out = embedder.embed_queries(["q1"], batch_size=4)
    assert seen[0]["input_type"] == "query"
    assert out.shape == (1, 2)
    assert embedder.tokens_used == 9

    embedder.reset_usage()
    embedder.embed_documents(["d1"], batch_size=4)
    assert seen[1]["input_type"] == "document"


def test_cohere_embed_uses_search_input_types() -> None:
    seen: list[dict] = []

    class FakeCohere:
        def embed(self, *, texts, model, input_type, embedding_types):
            seen.append(
                {
                    "texts": list(texts),
                    "model": model,
                    "input_type": input_type,
                    "embedding_types": embedding_types,
                }
            )
            return SimpleNamespace(
                embeddings=SimpleNamespace(float_=[[1.0, 0.0] for _ in texts]),
                meta=SimpleNamespace(
                    billed_units=SimpleNamespace(input_tokens=7),
                    tokens=None,
                ),
            )

    embedder = CohereEmbedder(id="cohere", model="embed-english-v3.0", client=FakeCohere())
    out = embedder.embed_queries(["q1"], batch_size=4)
    assert seen[0]["input_type"] == "search_query"
    assert seen[0]["embedding_types"] == ["float"]
    assert out.shape == (1, 2)
    assert embedder.tokens_used == 7

    embedder.embed_documents(["d1"], batch_size=4)
    assert seen[1]["input_type"] == "search_document"


def test_factory_voyage_and_cohere() -> None:
    voyage = create_embedder(
        {"id": "voyage-lite", "provider": "voyage", "model": "voyage-3-lite"}
    )
    cohere = create_embedder(
        {"id": "cohere-v3", "provider": "cohere", "model": "embed-english-v3.0"}
    )
    assert isinstance(voyage, VoyageEmbedder)
    assert isinstance(cohere, CohereEmbedder)

