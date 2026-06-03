.PHONY: install fmt lint type test test-int demo db-up db-down migrate check hooks

install:  ## Create the venv and install all dependencies
	uv sync

fmt:  ## Auto-format and auto-fix
	uv run black src tests scripts
	uv run ruff check --fix src tests scripts

lint:  ## Lint and format-check (no changes)
	uv run ruff check src tests scripts
	uv run black --check src tests scripts

type:  ## Static type check
	uv run pyright

test:  ## Fast tests (no database required)
	uv run pytest -m "not integration"

test-int:  ## All tests, including Postgres integration (needs MT_TEST_DATABASE_URL)
	uv run pytest

demo:  ## Run the Phase 0 end-to-end demo
	uv run python scripts/phase0_demo.py

db-up:  ## Start Postgres (pgvector) via docker compose
	docker compose up -d db

db-down:  ## Stop the database
	docker compose down

migrate:  ## Apply Alembic migrations to MT_DATABASE_URL
	uv run alembic upgrade head

hooks:  ## Install pre-commit hooks
	uv run pre-commit install
	uv run pre-commit install --hook-type pre-push

check: lint type test  ## Lint + types + fast tests (what CI runs, minus integration)
