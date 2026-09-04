"""Write benchmark CSV and matplotlib charts."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CSV_COLUMNS = [
    "model_id",
    "provider_model",
    "n_corpus",
    "n_queries",
    "k",
    "batch_size",
    "mrr",
    "recall@1",
    "recall@5",
    "recall@10",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_batch_ms_per_query",
    "tokens_used",
    "cost_usd",
    "price_per_1m_tokens",
    "embed_dim",
    "device",
    "processor",
    "platform",
]

_MARKERS = ["o", "s", "D", "^", "v", "P"]
_METRIC_BARS = ("mrr", "recall@1", "recall@5", "recall@10")


def results_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{col: row.get(col) for col in CSV_COLUMNS} for row in rows])


def merge_result_frames(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Replace existing rows that share ``model_id``; keep the rest."""
    existing = _align_columns(existing)
    incoming = _align_columns(incoming)
    if existing.empty:
        return incoming
    incoming_ids = set(incoming["model_id"].astype(str))
    kept = existing[~existing["model_id"].astype(str).isin(incoming_ids)]
    return pd.concat([kept, incoming], ignore_index=True)


def _align_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aligned = frame.copy()
    for col in CSV_COLUMNS:
        if col not in aligned.columns:
            aligned[col] = pd.NA
    return aligned[CSV_COLUMNS]


def _load_latest(out_dir: Path) -> pd.DataFrame | None:
    latest = out_dir / "benchmark_latest.csv"
    if not latest.exists():
        return None
    return pd.read_csv(latest)


def merged_frame(rows: list[dict[str, Any]], out_dir: Path) -> pd.DataFrame:
    incoming = results_frame(rows)
    existing = _load_latest(out_dir)
    if existing is None or existing.empty:
        return incoming
    return merge_result_frames(existing, incoming)


def write_csv(rows: list[dict[str, Any]], out_dir: Path, *, day: date | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = (day or date.today()).strftime("%Y%m%d")
    path = out_dir / f"benchmark_{stamp}.csv"
    frame = merged_frame(rows, out_dir)
    frame.to_csv(path, index=False)
    latest = out_dir / "benchmark_latest.csv"
    frame.to_csv(latest, index=False)
    return path


def _axis_limits(values: list[float], *, lower: float | None = None, min_span: float = 0.05) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    span = max(hi - lo, min_span)
    pad = span * 0.2
    start = lo - pad if lower is None else lower
    return start, hi + pad


def _scatter_points(ax, xs, ys, labels) -> None:
    cmap = plt.get_cmap("tab10")
    for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
        ax.scatter(
            x,
            y,
            color=cmap(i % 10),
            marker=_MARKERS[i % len(_MARKERS)],
            s=80,
            zorder=3,
            label=labels[i],
        )
    ax.legend(frameon=False, loc="best")


def write_charts(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = results_frame(rows)
    labels = frame["model_id"].tolist()

    cost_path = out_dir / "mrr_vs_cost.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    costs = frame["cost_usd"].tolist()
    mrrs = frame["mrr"].tolist()
    _scatter_points(ax, costs, mrrs, labels)
    ax.set_title("SciFact MRR vs embedding cost")
    ax.set_xlabel("Cost (USD, corpus + queries)")
    ax.set_ylabel("MRR")
    ax.set_xlim(*_axis_limits(costs, min_span=0.04))
    ax.set_ylim(*_axis_limits(mrrs, min_span=0.08))
    fig.tight_layout()
    fig.savefig(cost_path, dpi=140)
    plt.close(fig)

    latency_path = out_dir / "latency_vs_mrr.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    lats = frame["latency_p50_ms"].tolist()
    _scatter_points(ax, mrrs, lats, labels)
    ax.set_title("Query embed latency (p50) vs SciFact MRR")
    ax.set_xlabel("MRR")
    ax.set_ylabel("Latency p50 (ms / query, batch=1)")
    ax.set_xlim(*_axis_limits(mrrs, min_span=0.08))
    ax.set_ylim(*_axis_limits(lats, lower=0.0, min_span=1.0))
    fig.tight_layout()
    fig.savefig(latency_path, dpi=140)
    plt.close(fig)

    bars_path = out_dir / "metrics_comparison.png"
    ranked = frame.sort_values("mrr", ascending=False)
    bar_labels = ranked["model_id"].tolist()
    x = range(len(bar_labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    for i, metric in enumerate(_METRIC_BARS):
        xs = [pos + (i - 1.5) * width for pos in x]
        ax.bar(xs, ranked[metric].tolist(), width=width, label=metric, color=cmap(i))
    ax.set_xticks(list(x))
    ax.set_xticklabels(bar_labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("SciFact retrieval metrics")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(bars_path, dpi=140)
    plt.close(fig)

    return {
        "mrr_vs_cost": cost_path,
        "latency_vs_mrr": latency_path,
        "metrics_comparison": bars_path,
    }


def write_report(rows: list[dict[str, Any]], out_dir: Path, *, day: date | None = None) -> dict[str, Path]:
    csv_path = write_csv(rows, out_dir, day=day)
    merged_rows = pd.read_csv(csv_path).to_dict(orient="records")
    charts = write_charts(merged_rows, out_dir)
    return {"csv": csv_path, **charts}


def write_spotcheck(
    path: Path,
    *,
    model_id: str,
    query_id: str,
    gold: set[str],
    top_k: list[str],
    query_text: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gold_txt = ", ".join(sorted(gold)) or "(none)"
    ranked = "\n".join(f"- {doc_id}" for doc_id in top_k)
    path.write_text(
        f"# Spot-check ({model_id})\n\n"
        f"**Query `{query_id}`:** {query_text}\n\n"
        f"**Gold docs:** {gold_txt}\n\n"
        f"**Top-k:**\n{ranked}\n",
        encoding="utf-8",
    )
