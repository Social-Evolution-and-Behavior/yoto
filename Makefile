.PHONY: install test test-fast lint typecheck dead-code format quality \
       docs-build docs-serve clean

install:
	pip install -e ".[dev,docs,lint]"

test:
	pytest

test-fast:
	pytest -m "not slow"

lint:
	black --check src/ tests/
	flake8 src/ tests/

typecheck:
	mypy src/

dead-code:
	vulture src/yoto

format:
	black src/ tests/

quality: lint typecheck dead-code

docs-build:
	mkdocs build

docs-serve:
	mkdocs serve

clean:
	rm -rf build/ dist/ .eggs/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache .pytest_cache .coverage htmlcov
	rm -rf docs/site
