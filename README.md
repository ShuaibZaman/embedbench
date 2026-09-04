# embedbench

Reproducible side-by-side benchmark of embedding models on a fixed retrieval task — latency, cost, and accuracy.

## Leaderboard (BEIR SciFact test)

Local models below. API rows are implemented in-repo but **not run** in this pass (no keys configured).

| Model | Type | MRR | Recall@1 | Recall@5 | Recall@10 | p50 ms | p95 ms | Cost (USD) |
|-------|------|-----|----------|----------|-----------|--------|--------|------------|
| `bge-small` (`BAAI/bge-small-en-v1.5`) | local | *pending run* | | | | | | $0 |
| `minilm` (`all-MiniLM-L6-v2`) | local | *pending run* | | | | | | $0 |
| `openai-3-small` | API | not run — key not configured | | | | | | |
| `openai-3-large` | API | not run — key not configured | | | | | | |
| `voyage-3-lite` | API | not run — key not configured | | | | | | |
| `cohere-embed-v3` | API | not run — key not configured | | | | | | |

Charts after a local run: [`results/mrr_vs_cost.png`](results/mrr_vs_cost.png), [`results/latency_vs_mrr.png`](results/latency_vs_mrr.png).

## Problem

Teams pick embedding models from blog posts. The same retrieval task, same k, and the same metrics make cost, latency, and quality comparable — and the run is one command.

## Dataset

[BEIR SciFact](https://github.com/allenai/scifact) test split from the official zip (not the `beir` Python package):

- ~5,183 scientific abstracts (corpus)
- ~300 fact-checking queries
- qrels in `qrels/test.tsv`

Each corpus row is `title + text`. Use `--max-queries N` to subsample queries for a cheaper iteration loop; published numbers should use the full test split.

## Methodology

- **Task:** embed the corpus once, embed each query, brute-force cosine top-10.
- **k:** 10 for every model.
- **Recall@k:** fraction of queries with **at least one** gold document in the top-k (Success@k, not classic multi-relevant recall).
- **MRR:** mean reciprocal rank of the first relevant document (0 if none in the retrieved list).
- **Latency:** per-query embed time at **batch=1** (p50 / p95). A second pass at **batch=32** is recorded as `latency_batch_ms_per_query`.
- **Cost:** `(tokens_used / 1e6) * price_per_1m_tokens`. Local models report a whitespace token count for scale only; billed cost is $0.
- **Normalization:** embeddings are L2-normalized before cosine. BGE queries get the official instruction prefix from `configs/models.yaml`.
- **Hardware:** recorded per run (`device`, `processor`, `platform` columns in the CSV). Fill the note below after you reproduce.

**Hardware (this machine):** *pending first local run.*

**Statistical note:** 300 SciFact test queries is enough to rank models for a portfolio write-up; treat small MRR gaps as noisy.

## Results

See the table at the top and `results/benchmark_YYYYMMDD.csv` after `make reproduce`.

## Insights

Filled after the local MiniLM vs bge-small run (both $0, so the comparison is quality vs CPU latency). API cost/quality triangles need keys.

1. *pending local numbers*
2. *pending local numbers*
3. *pending local numbers*

## Limitations

- Single English dataset (SciFact). Rankings may not transfer to code, chat, or multilingual corpora.
- No fine-tuning, rerankers, or ANN indexes — brute-force cosine only.
- API providers are implemented but skipped without keys, so the public table is local-only until someone re-runs with `OPENAI_API_KEY` / `VOYAGE_API_KEY` / `COHERE_API_KEY`.
- Query latency includes Python/SDK overhead, not a production embedding service.
- Local token counts are approximations; they are not billed.

## Reproduce

Python 3.11+ (this repo pins 3.12 via `.python-version`). [uv](https://docs.astral.sh/uv/) is used for the lockfile.

```powershell
python -m uv sync
python -m uv run python -m embedbench.benchmark --config configs/models.yaml --models bge-small,minilm --spot-check 1
```

Unix:

```bash
./scripts/run_benchmark.sh --models bge-small,minilm
```

Windows PowerShell:

```powershell
./scripts/run_benchmark.ps1 --models bge-small,minilm
```

`make reproduce` is the same local-only command if you have `make`.

API models (spends money):

```powershell
python -m uv run python -m embedbench.benchmark --models openai-3-small,openai-3-large,voyage-3-lite,cohere-embed-v3
```

Copy `.env.example` to `.env` first.

## Adding a model

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: add a YAML row; implement a provider only if the vendor is new.
