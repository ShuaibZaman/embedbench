.PHONY: reproduce test

reproduce:
	python -m uv run python -m embedbench.benchmark --config configs/models.yaml --models bge-small,minilm,e5-small,bge-base

test:
	python -m uv run pytest
