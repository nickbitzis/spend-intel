PORT ?= 8849
LINES ?= 120000

.PHONY: install test lint format analyze gate run docker-build docker-run screenshot clean

install:
	uv sync --python 3.12

test:
	uv run --group dev pytest -q

lint:
	uvx ruff check .
	uvx ruff format --check .

format:
	uvx ruff format .
	uvx ruff check . --fix

analyze:  ## full pipeline on 120,000 lines, writes data/report.json (about 90 seconds)
	uv run spendintel analyze --lines $(LINES)

gate:  ## what CI runs: fail if clustering precision or L1 accuracy drops below the floor
	uv run spendintel analyze --lines 40000 --gate --min-supplier-precision 0.97 --min-l1-accuracy 0.97

run:
	PORT=$(PORT) SPENDINTEL_LINES=$(LINES) uv run spendintel serve

docker-build:
	docker build -t spend-intel .

docker-run:
	docker run --rm -p $(PORT):8849 spend-intel

screenshot:  ## needs the server running on $(PORT) and Google Chrome
	uv run python scripts/screenshot.py --port $(PORT)

clean:
	rm -rf .pytest_cache .ruff_cache data
