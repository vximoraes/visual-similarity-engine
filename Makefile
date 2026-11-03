.PHONY: check lint format typecheck test migrate api worker clusterizar

# Tem que passar antes de cada commit
check: lint typecheck test

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy app tests

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
