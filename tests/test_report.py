"""Report writer tests — no model downloads."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from embedbench.report import results_frame, write_report, write_spotcheck

ROWS = [
    {
        "model_id": "minilm",
        "provider_model": "sentence-transformers/all-MiniLM-L6-v2",
        "n_corpus": 4,
        "n_queries": 3,
        "k": 10,
        "batch_size": 32,
        "mrr": 0.5,
        "recall@1": 0.33,
        "recall@5": 0.66,
        "recall@10": 0.66,
        "latency_p50_ms": 12.0,
        "latency_p95_ms": 20.0,
        "latency_batch_ms_per_query": 3.0,
        "tokens_used": 100,
        "cost_usd": 0.0,
        "price_per_1m_tokens": 0.0,
        "embed_dim": 384,
        "device": "cpu",
        "processor": "test-cpu",
        "platform": "test",
    },
    {
        "model_id": "bge-small",
        "provider_model": "BAAI/bge-small-en-v1.5",
        "n_corpus": 4,
        "n_queries": 3,
        "k": 10,
        "batch_size": 32,
        "mrr": 0.7,
        "recall@1": 0.5,
        "recall@5": 0.8,
        "recall@10": 0.9,
        "latency_p50_ms": 30.0,
        "latency_p95_ms": 40.0,
        "latency_batch_ms_per_query": 8.0,
        "tokens_used": 120,
        "cost_usd": 0.0,
        "price_per_1m_tokens": 0.0,
        "embed_dim": 384,
        "device": "cpu",
        "processor": "test-cpu",
        "platform": "test",
    },
]


def test_write_report_creates_csv_and_pngs(tmp_path: Path) -> None:
    paths = write_report(ROWS, tmp_path, day=date(2026, 9, 4))
    csv_path = paths["csv"]
    assert csv_path.name == "benchmark_20260904.csv"
    assert csv_path.exists()
    assert (tmp_path / "benchmark_latest.csv").exists()
    assert paths["mrr_vs_cost"].exists()
    assert paths["latency_vs_mrr"].exists()
    assert paths["metrics_comparison"].exists()
    frame = results_frame(ROWS)
    assert list(frame["model_id"]) == ["minilm", "bge-small"]
    text = csv_path.read_text(encoding="utf-8")
    assert "mrr" in text
    assert "minilm" in text
    assert "embed_dim" in text


def test_write_report_merges_by_model_id(tmp_path: Path) -> None:
    write_report([ROWS[0]], tmp_path, day=date(2026, 9, 4))
    write_report([ROWS[1]], tmp_path, day=date(2026, 9, 4))
    latest = (tmp_path / "benchmark_latest.csv").read_text(encoding="utf-8")
    assert "minilm" in latest
    assert "bge-small" in latest


def test_write_spotcheck(tmp_path: Path) -> None:
    path = tmp_path / "spotcheck.md"
    write_spotcheck(
        path,
        model_id="minilm",
        query_id="q0",
        query_text="query about alpha",
        gold={"d0"},
        top_k=["d0", "d1"],
    )
    body = path.read_text(encoding="utf-8")
    assert "minilm" in body
    assert "d0" in body
