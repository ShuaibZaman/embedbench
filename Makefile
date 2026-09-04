.PHONY: reproduce test

reproduce:
	python -m uv run python -m embedbench.benchmark --config configs/models.yaml --models bge-small,minilm

test:
	python -m uv run pytest
