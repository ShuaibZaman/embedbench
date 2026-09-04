"""Benchmark runner: embed corpus/queries, cosine top-k, metrics."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from embedbench.config import load_model_specs, select_specs
from embedbench.dataset import RetrievalDataset, load_scifact
from embedbench.index import ranked_doc_ids
from embedbench.metrics import embedding_cost, percentile, retrieval_metrics
from embedbench.providers.base import Embedder
from embedbench.providers.factory import create_embedder
from embedbench.report import write_report, write_spotcheck

LOGGER = logging.getLogger("embedbench")
DEFAULT_MODELS = ("bge-small", "minilm", "e5-small", "bge-base")


def hardware_info() -> dict[str, str]:
    device = "cpu"
    processor = platform.processor() or platform.machine()
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            processor = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return {
        "device": device,
        "processor": processor,
        "platform": platform.platform(),
    }


def _cache_paths(cache_dir: Path, model_id: str) -> tuple[Path, Path]:
    safe = model_id.replace("/", "_")
    return cache_dir / f"{safe}_corpus.npy", cache_dir / f"{safe}_corpus.meta.json"


def embed_corpus_cached(
    embedder: Embedder,
    texts: list[str],
    *,
    batch_size: int,
    cache_dir: Path,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy_path, meta_path = _cache_paths(cache_dir, embedder.id)
    if npy_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("n") == len(texts) and meta.get("model") == embedder.model:
            LOGGER.info("corpus cache hit for %s", embedder.id)
            embedder._tokens_used += int(meta.get("tokens_used") or 0)
            return np.load(npy_path)
    LOGGER.info("embedding %s corpus (%s docs)", embedder.id, len(texts))
    before = embedder.tokens_used
    vectors = embedder.embed_documents(texts, batch_size=batch_size)
    np.save(npy_path, vectors)
    meta_path.write_text(
        json.dumps(
            {
                "n": len(texts),
                "model": embedder.model,
                "dim": int(vectors.shape[1]),
                "tokens_used": embedder.tokens_used - before,
            }
        ),
        encoding="utf-8",
    )
    return vectors


def _restore_tokens(embedder: Embedder, tokens: int) -> None:
    embedder._tokens_used = tokens


def benchmark_model(
    embedder: Embedder,
    dataset: RetrievalDataset,
    *,
    k: int,
    batch_size: int,
    cache_dir: Path,
) -> dict[str, Any]:
    hw = hardware_info()
    corpus_vectors = embed_corpus_cached(
        embedder,
        dataset.corpus_texts,
        batch_size=batch_size,
        cache_dir=cache_dir,
    )

    latencies_ms: list[float] = []
    query_rows: list[np.ndarray] = []
    for text in dataset.query_texts:
        started = time.perf_counter()
        vector = embedder.embed_queries([text], batch_size=1)
        latencies_ms.append((time.perf_counter() - started) * 1000)
        query_rows.append(vector[0])
    query_vectors = np.stack(query_rows, axis=0)

    tokens_after_official = embedder.tokens_used
    started = time.perf_counter()
    embedder.embed_queries(dataset.query_texts, batch_size=batch_size)
    batch_elapsed_s = time.perf_counter() - started
    _restore_tokens(embedder, tokens_after_official)
    n_queries = max(len(dataset.query_texts), 1)
    batch_ms_per_query = (batch_elapsed_s * 1000) / n_queries

    rankings = ranked_doc_ids(query_vectors, corpus_vectors, dataset.corpus_ids, k)
    qrels = [dataset.qrels.get(qid, set()) for qid in dataset.query_ids]
    metrics = retrieval_metrics(rankings, qrels, ks=(1, 5, 10))
    cost = embedding_cost(embedder.tokens_used, embedder.price_per_1m_tokens)
    return {
        "model_id": embedder.id,
        "provider_model": embedder.model,
        "n_corpus": len(dataset.corpus_ids),
        "n_queries": len(dataset.query_ids),
        "k": k,
        "batch_size": batch_size,
        "mrr": metrics["mrr"],
        "recall@1": metrics["recall@1"],
        "recall@5": metrics["recall@5"],
        "recall@10": metrics["recall@10"],
        "latency_p50_ms": percentile(latencies_ms, 50),
        "latency_p95_ms": percentile(latencies_ms, 95),
        "latency_batch_ms_per_query": batch_ms_per_query,
        "tokens_used": embedder.tokens_used,
        "cost_usd": cost,
        "price_per_1m_tokens": embedder.price_per_1m_tokens,
        "embed_dim": int(corpus_vectors.shape[1]) if corpus_vectors.size else 0,
        **hw,
        "rankings": rankings,
    }


def run_benchmark(
    specs: list[dict[str, Any]],
    dataset: RetrievalDataset,
    *,
    k: int,
    batch_size: int,
    cache_dir: Path,
    embedders: list[Embedder] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    instances = embedders if embedders is not None else [create_embedder(spec) for spec in specs]
    for embedder in instances:
        if not embedder.available():
            reason = embedder.unavailable_reason() or "unavailable"
            LOGGER.warning("skipping %s (%s)", embedder.id, reason)
            skipped.append(f"{embedder.id}: {reason}")
            continue
        LOGGER.info("benchmarking %s", embedder.id)
        rows.append(
            benchmark_model(
                embedder,
                dataset,
                k=k,
                batch_size=batch_size,
                cache_dir=cache_dir,
            )
        )
    return rows, skipped


def _parse_models(raw: str | None) -> list[str]:
    if raw is None:
        return list(DEFAULT_MODELS)
    return [part.strip() for part in raw.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the embedbench retrieval benchmark.")
    parser.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model ids. Default: bge-small,minilm,e5-small,bge-base (local only).",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--spot-check", type=int, default=0, help="Print top-k for the first N queries.")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    cache_dir = args.cache_dir or (args.data_dir / "cache")
    specs = select_specs(load_model_specs(args.config), _parse_models(args.models))
    dataset = load_scifact(args.data_dir, max_queries=args.max_queries)
    rows, skipped = run_benchmark(
        specs,
        dataset,
        k=args.k,
        batch_size=args.batch_size,
        cache_dir=cache_dir,
    )
    public_rows = [{k: v for k, v in row.items() if k != "rankings"} for row in rows]
    print(json.dumps({"results": public_rows, "skipped": skipped}, indent=2))
    if rows:
        paths = write_report(rows, args.out_dir)
        LOGGER.info("wrote %s", paths["csv"])
        LOGGER.info("wrote %s", paths["mrr_vs_cost"])
        LOGGER.info("wrote %s", paths["latency_vs_mrr"])
        LOGGER.info("wrote %s", paths["metrics_comparison"])
    if args.spot_check and rows:
        _print_spot_check(dataset, rows[0], n=args.spot_check)
        first = rows[0]
        write_spotcheck(
            args.out_dir / "spotcheck.md",
            model_id=first["model_id"],
            query_id=dataset.query_ids[0],
            query_text=dataset.query_texts[0],
            gold=dataset.qrels.get(dataset.query_ids[0], set()),
            top_k=first["rankings"][0],
        )
    return 0 if rows else 1


def _print_spot_check(dataset: RetrievalDataset, row: dict[str, Any], *, n: int) -> None:
    print("\n# spot-check")
    for i in range(min(n, len(dataset.query_ids))):
        qid = dataset.query_ids[i]
        print(f"query {qid}: {dataset.query_texts[i][:200]}")
        print(f"  gold: {sorted(dataset.qrels.get(qid, set()))}")
        print(f"  top-k: {row['rankings'][i]}")


if __name__ == "__main__":
    raise SystemExit(main())
