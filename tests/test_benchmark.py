"""Benchmark runner tests with a fake embedder (no model downloads)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from embedbench.benchmark import run_benchmark
from embedbench.config import load_model_specs, select_specs
from embedbench.dataset import load_beir_folder
from embedbench.providers.base import Embedder

FIXTURE = Path(__file__).parent / "fixtures" / "scifact_mini"


class FakeEmbedder(Embedder):
    def __init__(self, *, available: bool = True, dim: int = 4) -> None:
        super().__init__(id="fake", model="fake-model", price_per_1m_tokens=1.0)
        self._is_available = available
        self.dim = dim
        self.doc_calls = 0
        self.query_calls = 0

    def available(self) -> bool:
        return self._is_available

    def unavailable_reason(self) -> str | None:
        return None if self._is_available else "forced skip"

    def _encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vec = rng.normal(size=self.dim).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-12
            rows.append(vec)
        self._tokens_used += len(texts)
        return np.stack(rows, axis=0)

    def embed_documents(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        self.doc_calls += 1
        return self._encode(texts)

    def embed_queries(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        self.query_calls += 1
        return self._encode(texts)


def test_select_specs_default_order(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        """
models:
  - id: bge-small
    provider: sentence_transformers
    model: BAAI/bge-small-en-v1.5
  - id: minilm
    provider: sentence_transformers
    model: sentence-transformers/all-MiniLM-L6-v2
  - id: openai-3-small
    provider: openai
    model: text-embedding-3-small
""",
        encoding="utf-8",
    )
    specs = load_model_specs(path)
    selected = select_specs(specs, ["minilm", "bge-small"])
    assert [s["id"] for s in selected] == ["minilm", "bge-small"]


def test_run_benchmark_skips_unavailable(tmp_path: Path) -> None:
    dataset = load_beir_folder(FIXTURE, split="test", name="mini")
    ready = FakeEmbedder(available=True)
    skipped_model = FakeEmbedder(available=False)
    skipped_model.id = "skipped"
    rows, skipped = run_benchmark(
        [],
        dataset,
        k=2,
        batch_size=2,
        cache_dir=tmp_path / "cache",
        embedders=[skipped_model, ready],
    )
    assert skipped == ["skipped: forced skip"]
    assert len(rows) == 1
    row = rows[0]
    assert row["model_id"] == "fake"
    assert row["n_queries"] == 3
    assert "mrr" in row
    assert row["embed_dim"] == 4
    assert ready.doc_calls == 1


def test_real_models_yaml_has_local_defaults() -> None:
    specs = load_model_specs(Path("configs/models.yaml"))
    ids = [spec["id"] for spec in specs]
    assert "bge-small" in ids
    assert "minilm" in ids
    assert "e5-small" in ids
    assert "bge-base" in ids
    assert "voyage-3" not in ids


def test_corpus_cache_avoids_second_embed(tmp_path: Path) -> None:
    dataset = load_beir_folder(FIXTURE, split="test", name="mini")
    first = FakeEmbedder()
    run_benchmark([], dataset, k=2, batch_size=2, cache_dir=tmp_path / "cache", embedders=[first])
    second = FakeEmbedder()
    run_benchmark([], dataset, k=2, batch_size=2, cache_dir=tmp_path / "cache", embedders=[second])
    assert first.doc_calls == 1
    assert second.doc_calls == 0
    assert second.tokens_used >= first.tokens_used - 20
