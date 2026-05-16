.PHONY: install dev-install test lint format type-check docker-up docker-down clean train-local train-spark tf-init tf-validate tf-fmt tf-lint

install:
	uv sync

dev-install:
	uv sync --all-extras

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

type-check:
	uv run mypy libs app

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov/ dist/ build/

train-local:
	uv run python -m app.cli.train --backend sklearn

train-spark:
	uv run python -m app.cli.train --backend spark

# terraform validation (no cloud connection required)
tf-init:
	cd terraform && terraform init -backend=false

tf-validate:
	cd terraform && terraform validate

tf-fmt:
	cd terraform && terraform fmt -check -recursive

tf-lint:
	cd terraform && tflint --recursive
