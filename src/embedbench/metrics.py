"""Retrieval metrics. Recall@k is Success@k: any relevant doc in the top-k."""

from __future__ import annotations

import numpy as np


def reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def success_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return 1.0 if relevant.intersection(ranked_ids[:k]) else 0.0


def mean_reciprocal_rank(
    rankings: list[list[str]],
    qrels: list[set[str]],
) -> float:
    if not rankings:
        return 0.0
    return float(
        np.mean([reciprocal_rank(ranked, rel) for ranked, rel in zip(rankings, qrels, strict=True)])
    )


def recall_at_k(
    rankings: list[list[str]],
    qrels: list[set[str]],
    k: int,
) -> float:
    """Spec Recall@k = fraction of queries with at least one relevant doc in top-k."""
    if not rankings:
        return 0.0
    return float(
        np.mean([success_at_k(ranked, rel, k) for ranked, rel in zip(rankings, qrels, strict=True)])
    )


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def embedding_cost(tokens_used: int, price_per_1m_tokens: float) -> float:
    return (tokens_used / 1_000_000) * price_per_1m_tokens


def retrieval_metrics(
    rankings: list[list[str]],
    qrels: list[set[str]],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    out: dict[str, float] = {"mrr": mean_reciprocal_rank(rankings, qrels)}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(rankings, qrels, k)
    return out
