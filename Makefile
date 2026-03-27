.PHONY: install install-dev test lint format clean train sweep compare

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,notebooks]"

test:
	pytest tests/

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info/ __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

train:
	medre-bench train $(ARGS)

sweep:
	medre-bench sweep $(ARGS)

compare:
	medre-bench compare $(ARGS)
