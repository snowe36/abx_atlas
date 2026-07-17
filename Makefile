.PHONY: setup lint test reproduce clean

setup:
	python3.11 -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -e ".[dev]"

lint:
	ruff check .

test:
	MPLBACKEND=Agg pytest -q

reproduce:
	bash scripts/reproduce.sh

clean:
	find . -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
