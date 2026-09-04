"""Brute-force cosine search over L2-normalized embeddings."""

from __future__ import annotations

import numpy as np


def top_k(
    query_vectors: np.ndarray,
    corpus_vectors: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, scores) of shape (n_queries, k), highest cosine first."""
    if query_vectors.size == 0 or corpus_vectors.size == 0:
        n_q = query_vectors.shape[0]
        return (
            np.zeros((n_q, 0), dtype=np.int64),
            np.zeros((n_q, 0), dtype=np.float32),
        )
    k = min(k, corpus_vectors.shape[0])
    scores = query_vectors @ corpus_vectors.T
    if k == corpus_vectors.shape[0]:
        order = np.argsort(-scores, axis=1)
        row = np.arange(scores.shape[0])[:, None]
        return order, scores[row, order]
    partitioned = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    row = np.arange(scores.shape[0])[:, None]
    part_scores = scores[row, partitioned]
    order = np.argsort(-part_scores, axis=1)
    indices = partitioned[row, order]
    return indices, part_scores[row, order]


def ranked_doc_ids(
    query_vectors: np.ndarray,
    corpus_vectors: np.ndarray,
    corpus_ids: list[str],
    k: int,
) -> list[list[str]]:
    indices, _scores = top_k(query_vectors, corpus_vectors, k)
    return [[corpus_ids[j] for j in row] for row in indices]
