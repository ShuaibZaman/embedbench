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
    "device",
    "processor",
    "platform",
]


def results_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{col: row.get(col) for col in CSV_COLUMNS} for row in rows])


def write_csv(rows: list[dict[str, Any]], out_dir: Path, *, day: date | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = (day or date.today()).strftime("%Y%m%d")
    path = out_dir / f"benchmark_{stamp}.csv"
    frame = results_frame(rows)
    frame.to_csv(path, index=False)
    latest = out_dir / "benchmark_latest.csv"
    frame.to_csv(latest, index=False)
    return path


def _annotate(ax, xs, ys, labels) -> None:
    for x, y, label in zip(xs, ys, labels, strict=True):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)


def write_charts(rows: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = results_frame(rows)
    labels = frame["model_id"].tolist()

    cost_path = out_dir / "mrr_vs_cost.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(frame["cost_usd"], frame["mrr"])
    _annotate(ax, frame["cost_usd"], frame["mrr"], labels)
    ax.set_title("SciFact MRR vs embedding cost")
    ax.set_xlabel("Cost (USD, corpus + queries)")
    ax.set_ylabel("MRR")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(cost_path, dpi=140)
    plt.close(fig)

    latency_path = out_dir / "latency_vs_mrr.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(frame["mrr"], frame["latency_p50_ms"])
    _annotate(ax, frame["mrr"], frame["latency_p50_ms"], labels)
    ax.set_title("Query embed latency (p50) vs SciFact MRR")
    ax.set_xlabel("MRR")
    ax.set_ylabel("Latency p50 (ms / query, batch=1)")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(latency_path, dpi=140)
    plt.close(fig)
    return cost_path, latency_path


def write_report(rows: list[dict[str, Any]], out_dir: Path, *, day: date | None = None) -> dict[str, Path]:
    csv_path = write_csv(rows, out_dir, day=day)
    cost_path, latency_path = write_charts(rows, out_dir)
    return {"csv": csv_path, "mrr_vs_cost": cost_path, "latency_vs_mrr": latency_path}


def write_spotcheck(
    path: Path,
    *,
    model_id: str,
    query_id: str,
    query_text: str,
    gold: set[str],
    top_k: list[str],
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
