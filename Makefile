.PHONY: help setup sync data lint format check typecheck test test-cov \
        notebooks run lab clean reset ci dev

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## First-time setup (uv sync + ipykernel)
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	uv sync --all-extras
	uv run python -m ipykernel install --user --name anomaly-arena
	@echo "\n✅ Ready. Run 'make data' then 'make test'."

sync: ## Sync dependencies
	uv sync --all-extras

data: ## Download and cache credit card dataset
	uv run python -m src.datasets.credit_card
	@echo "✅ Data cached in data/"

lint: format check typecheck ## Run all linters

format: ## ruff format
	uv run ruff format src/ tests/ app/

check: ## ruff check
	uv run ruff check --fix src/ tests/ app/

typecheck: ## ty type check
	uv run ty check src/

test: ## pytest with coverage report (fails if coverage < 80%)
	uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=80

test-cov: ## pytest with coverage + HTML report in htmlcov/
	uv run pytest --cov=src --cov-report=term-missing --cov-report=html --cov-fail-under=80

notebooks: data ## Execute all notebooks sequentially
	@for nb in notebooks/0*.ipynb; do \
		echo "▶ Executing $$nb ..."; \
		uv run jupyter nbconvert --to notebook --execute --inplace \
			--ExecutePreprocessor.timeout=300 "$$nb" || exit 1; \
	done
	@echo "\n✅ All notebooks executed."

run: ## Start Streamlit app on :8501
	uv run streamlit run app/streamlit_app.py --server.port 8501

lab: ## Start JupyterLab on :8888
	uv run jupyter lab --no-browser --port 8888

clean: ## Remove cache and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache htmlcov .coverage .coverage.*
	@echo "🧹 Cleaned."

reset: clean ## Full reset (removes venv and cached data)
	rm -rf .venv data/creditcard.csv
	@echo "🔄 Reset complete."

ci: sync lint test ## Full CI pipeline

dev: lint test ## Fast dev loop
