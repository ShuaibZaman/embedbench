"""Golden tests for MRR, Recall@k (Success@k), index, and SciFact parsing."""

from __future__ import annotations

from pathlib import Path
import io
import zipfile

import numpy as np
import pytest

from embedbench.dataset import load_beir_folder, load_scifact
from embedbench.index import ranked_doc_ids, top_k
from embedbench.metrics import (
    embedding_cost,
    mean_reciprocal_rank,
    percentile,
    recall_at_k,
    reciprocal_rank,
    retrieval_metrics,
    success_at_k,
)

FIXTURE = Path(__file__).parent / "fixtures" / "scifact_mini"

# Hand-ranked lists for a 4-doc / 3-query toy set:
# q0 relevant {d0} ranked [d0, d1, d2, d3] -> RR=1, R@1=1
# q1 relevant {d2} ranked [d1, d2, d0, d3] -> RR=0.5, R@1=0, R@5=1
# q2 relevant {d3} ranked [d0, d1, d2, d1] -> RR=0 (d3 never appears)
RANKINGS = [
    ["d0", "d1", "d2", "d3"],
    ["d1", "d2", "d0", "d3"],
    ["d0", "d1", "d2", "d1"],
]
QRELS = [{"d0"}, {"d2"}, {"d3"}]


def test_reciprocal_rank_first_hit() -> None:
    assert reciprocal_rank(["d0", "d1"], {"d0"}) == 1.0
    assert reciprocal_rank(["d1", "d2"], {"d2"}) == 0.5
    assert reciprocal_rank(["d0", "d1"], {"d9"}) == 0.0


def test_success_at_k() -> None:
    assert success_at_k(["d0", "d1"], {"d0"}, k=1) == 1.0
    assert success_at_k(["d1", "d2"], {"d2"}, k=1) == 0.0
    assert success_at_k(["d1", "d2"], {"d2"}, k=2) == 1.0


def test_mrr_and_recall_toy_set() -> None:
    mrr = mean_reciprocal_rank(RANKINGS, QRELS)
    # (1.0 + 0.5 + 0.0) / 3
    assert mrr == pytest.approx(0.5)
    assert recall_at_k(RANKINGS, QRELS, k=1) == pytest.approx(1 / 3)
    assert recall_at_k(RANKINGS, QRELS, k=5) == pytest.approx(2 / 3)
    metrics = retrieval_metrics(RANKINGS, QRELS, ks=(1, 5, 10))
    assert metrics["mrr"] == pytest.approx(0.5)
    assert metrics["recall@1"] == pytest.approx(1 / 3)
    assert metrics["recall@10"] == pytest.approx(2 / 3)


def test_cost_and_percentile() -> None:
    assert embedding_cost(500_000, 0.02) == pytest.approx(0.01)
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)


def test_index_returns_nearest_doc() -> None:
    corpus = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    queries = np.array([[0.1, 0.0, 0.9]], dtype=np.float32)
    ids = ranked_doc_ids(queries, corpus, ["a", "b", "c"], k=2)
    assert ids[0][0] == "c"
    indices, scores = top_k(queries, corpus, k=2)
    assert indices.shape == (1, 2)
    assert scores[0, 0] >= scores[0, 1]


def test_load_beir_mini_fixture() -> None:
    dataset = load_beir_folder(FIXTURE, split="test", name="mini")
    assert dataset.corpus_ids == ["d0", "d1", "d2", "d3"]
    assert dataset.corpus_texts[0].startswith("Alpha paper")
    assert dataset.query_ids == ["q0", "q1", "q2"]
    assert dataset.qrels["q0"] == {"d0"}
    assert dataset.qrels["q1"] == {"d2"}
    subset = dataset.with_max_queries(1)
    assert subset.query_ids == ["q0"]
    assert "q1" not in subset.qrels


def test_load_scifact_uses_extracted_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import embedbench.dataset as dataset_mod

    buffer = io.BytesIO()
    root = Path("scifact")
    with zipfile.ZipFile(buffer, "w") as archive:
        for rel in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
            archive.writestr((root / rel).as_posix(), (FIXTURE / rel).read_text(encoding="utf-8"))
    payload = buffer.getvalue()

    class FakeResponse:
        content = payload

        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url: str):
            assert "scifact.zip" in url
            return FakeResponse()

    monkeypatch.setattr(dataset_mod.httpx, "Client", FakeClient)
    loaded = load_scifact(tmp_path, max_queries=2)
    assert loaded.query_ids == ["q0", "q1"]
    assert len(loaded.corpus_ids) == 4
