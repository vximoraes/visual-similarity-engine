.PHONY: check lint format typecheck test migrate api worker clusterizar imagenet importar avaliar

# Pasta do conjunto de avaliação (uma subpasta por categoria)
PASTA ?= assets

# Tem que passar antes de cada commit
check: lint typecheck test

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy app tests scripts

test:
	uv run pytest

migrate:
	uv run alembic upgrade head

api:
	uv run fastapi dev --port 8000

worker:
	uv run python -m app.worker

clusterizar:
	uv run python -m app.jobs.clusterizar

# Extrai a base do professor (zip "Imagenet Dataset Mini") em $(PASTA)/<categoria>/
imagenet:
	uv run python scripts/preparar_imagenet.py "$(ZIP)" $(PASTA)

importar:
	uv run python -m app.jobs.importar $(PASTA)

avaliar:
	uv run python -m app.jobs.avaliar $(PASTA)
