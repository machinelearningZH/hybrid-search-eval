.DEFAULT_GOAL := help

PROFILE ?= embeddings
CONFIG ?= _configs/config.yaml
INPUT ?=
DATASET ?=
ARGS ?=

export CONFIG INPUT DATASET

ifeq ($(PROFILE),embeddings)
UV_PROFILE :=
else ifeq ($(PROFILE),colbert)
UV_PROFILE := --no-group embeddings --group colbert
else
$(error PROFILE must be embeddings or colbert)
endif

UV_RUN := uv run --locked $(UV_PROFILE)

.PHONY: help sync eval eval-force queries download datasets test lint format format-check check

help:
	@printf '%s\n' \
	  'Commands:' \
	  '  make sync          Sync the locked environment' \
	  '  make eval          Run evaluations with CONFIG (default: _configs/config.yaml)' \
	  '  make eval-force    Run evaluations, recomputing cached embeddings and results' \
	  '  make queries       Generate queries; requires INPUT=path/to/documents.csv' \
	  '  make download      Download an MTEB dataset; requires DATASET=mteb/scifact' \
	  '  make datasets      List available retrieval datasets' \
	  '  make test          Run pytest' \
	  '  make lint          Run Ruff lint checks' \
	  '  make format        Apply safe Ruff lint fixes, then format Python files' \
	  '  make format-check  Check Python formatting without changing files' \
	  '  make check         Run formatting, lint, and tests in sequence' \
	  '' \
	  'Options:' \
	  '  PROFILE=colbert    Use PyLate with Sentence Transformers 5.3.x' \
	  '  PROFILE=embeddings Use Sentence Transformers 6.x (default)' \
	  '  CONFIG=path.yaml   Override evaluation/query configuration' \
	  '  ARGS="..."        Append shell command-line arguments (except sync/check/help)'

sync:
	uv sync --locked $(UV_PROFILE)

eval:
	$(UV_RUN) generate_evals.py --config "$$CONFIG" $(ARGS)

eval-force:
	$(UV_RUN) generate_evals.py --config "$$CONFIG" --force-recompute $(ARGS)

queries:
	@test -n "$$INPUT" || { printf '%s\n' 'Set INPUT=path/to/documents.csv'; exit 2; }
	$(UV_RUN) generate_queries.py "$$INPUT" --config "$$CONFIG" $(ARGS)

download:
	@test -n "$$DATASET" || { printf '%s\n' 'Set DATASET=mteb/scifact'; exit 2; }
	$(UV_RUN) download_mteb_datasets.py "$$DATASET" $(ARGS)

datasets:
	$(UV_RUN) list_retrieval_datasets.py $(ARGS)

test:
	$(UV_RUN) pytest $(ARGS)

lint:
	$(UV_RUN) ruff check . $(ARGS)

format:
	$(UV_RUN) ruff check --fix --exit-zero . $(ARGS)
	$(UV_RUN) ruff format . $(ARGS)

format-check:
	$(UV_RUN) ruff format --check . $(ARGS)

check:
	$(MAKE) format-check
	$(MAKE) lint
	$(MAKE) test
