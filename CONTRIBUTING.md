# Contributing

## Add a model in 3 steps

1. **YAML row.** Append to `configs/models.yaml`:

   ```yaml
   - id: my-model
     provider: openai          # or voyage, cohere, sentence_transformers
     model: the-vendor-id
     price_per_1m_tokens: 0.02
     # query_prefix: "optional instruction prefix "
   ```

2. **Provider (only if the vendor is new).** Implement `Embedder` in `src/embedbench/providers/`, register it in `PROVIDERS` inside `src/embedbench/providers/factory.py`, and add a mocked unit test in `tests/test_providers.py`. `available()` must be false when the API key is missing so the runner never calls the SDK.

3. **Run just that row.**

   ```powershell
   python -m uv run python -m embedbench.benchmark --models my-model
   ```

   Check `results/benchmark_latest.csv` and, if quality looks off, `--spot-check 1` to print one query’s top-k against the SciFact gold set.

## Tests

```powershell
python -m uv run pytest
```

Please keep metric tests in `tests/test_eval.py` hand-checkable (small ranked lists, known MRR).
