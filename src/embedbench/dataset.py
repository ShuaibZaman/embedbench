"""BEIR SciFact loader. Parses official zip files; does not depend on the beir package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import zipfile

import httpx

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


@dataclass(frozen=True)
class RetrievalDataset:
    name: str
    corpus_ids: list[str]
    corpus_texts: list[str]
    query_ids: list[str]
    query_texts: list[str]
    qrels: dict[str, set[str]]

    def with_max_queries(self, max_queries: int | None) -> RetrievalDataset:
        if max_queries is None or max_queries >= len(self.query_ids):
            return self
        query_ids = self.query_ids[:max_queries]
        keep = set(query_ids)
        qrels = {qid: docs for qid, docs in self.qrels.items() if qid in keep}
        query_texts = self.query_texts[:max_queries]
        return RetrievalDataset(
            name=self.name,
            corpus_ids=self.corpus_ids,
            corpus_texts=self.corpus_texts,
            query_ids=query_ids,
            query_texts=query_texts,
            qrels=qrels,
        )


def _corpus_text(row: dict) -> str:
    title = (row.get("title") or "").strip()
    body = (row.get("text") or "").strip()
    if title and body:
        return f"{title} {body}"
    return title or body


def load_beir_folder(folder: Path, *, split: str = "test", name: str = "scifact") -> RetrievalDataset:
    folder = folder.resolve()
    corpus_ids: list[str] = []
    corpus_texts: list[str] = []
    with (folder / "corpus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            corpus_ids.append(str(row["_id"]))
            corpus_texts.append(_corpus_text(row))

    query_ids: list[str] = []
    query_texts: list[str] = []
    with (folder / "queries.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query_ids.append(str(row["_id"]))
            query_texts.append(str(row.get("text") or ""))

    qrels: dict[str, set[str]] = {qid: set() for qid in query_ids}
    qrels_path = folder / "qrels" / f"{split}.tsv"
    with qrels_path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            if parts[0] in {"query-id", "qid"}:
                continue
            if len(parts) == 4:
                qid, _iter, doc_id, score = parts
            elif len(parts) == 3:
                qid, doc_id, score = parts
            else:
                raise ValueError(f"unexpected qrels row: {line!r}")
            if float(score) <= 0:
                continue
            qrels.setdefault(str(qid), set()).add(str(doc_id))

    kept_ids: list[str] = []
    kept_texts: list[str] = []
    kept_qrels: dict[str, set[str]] = {}
    for qid, text in zip(query_ids, query_texts, strict=True):
        rel = qrels.get(qid, set())
        if not rel:
            continue
        kept_ids.append(qid)
        kept_texts.append(text)
        kept_qrels[qid] = rel

    return RetrievalDataset(
        name=name,
        corpus_ids=corpus_ids,
        corpus_texts=corpus_texts,
        query_ids=kept_ids,
        query_texts=kept_texts,
        qrels=kept_qrels,
    )


def download_scifact(data_dir: Path, *, url: str = SCIFACT_URL, timeout: float = 120.0) -> Path:
    """Download and unzip SciFact into ``data_dir/scifact`` if missing."""
    dest = data_dir / "scifact"
    if (dest / "corpus.jsonl").exists() and (dest / "queries.jsonl").exists():
        return dest
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "scifact.zip"
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        response = client.get(url)
        response.raise_for_status()
        zip_path.write_bytes(response.content)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(data_dir)
    zip_path.unlink(missing_ok=True)
    if (dest / "corpus.jsonl").exists():
        return dest
    nested = data_dir / "scifact"
    if nested != dest and (nested / "corpus.jsonl").exists():
        return nested
    raise FileNotFoundError(f"extracted SciFact zip but corpus.jsonl not found under {data_dir}")


def load_scifact(data_dir: Path, *, max_queries: int | None = None) -> RetrievalDataset:
    folder = download_scifact(data_dir)
    dataset = load_beir_folder(folder, split="test", name="scifact")
    return dataset.with_max_queries(max_queries)
