.PHONY: install train-tokenizer prepare-data pretrain sft eval test lint format export clean help

# ──────────────────────────────────────────────────────────────────────
# ACE — Autonomous Coding Engine
# ──────────────────────────────────────────────────────────────────────

PYTHON     ?= python
PIP        ?= pip
CONFIG     ?= configs/ace_micro.yaml
WANDB_PROJ ?= ace

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Setup ─────────────────────────────────────────────────────────────

install: ## Install package + dev deps
	$(PIP) install -e ".[dev]"
	pre-commit install

# ── Data pipeline ─────────────────────────────────────────────────────

train-tokenizer: ## Train the BPE tokenizer
	$(PYTHON) scripts/train_tokenizer.py --config $(CONFIG)

prepare-data: ## Preprocess & shard training data
	$(PYTHON) scripts/prepare_data.py --config $(CONFIG)

# ── Training ──────────────────────────────────────────────────────────

pretrain: ## Launch pre-training run
	$(PYTHON) scripts/pretrain.py --config $(CONFIG)

sft: ## Supervised fine-tuning
	$(PYTHON) scripts/sft.py --config $(CONFIG)

# ── Evaluation ────────────────────────────────────────────────────────

eval: ## Run evaluation suite
	$(PYTHON) scripts/evaluate.py --config $(CONFIG)

# ── Export ────────────────────────────────────────────────────────────

export: ## Export model to GGUF format
	$(PYTHON) scripts/export_gguf.py --config $(CONFIG)

# ── Quality ───────────────────────────────────────────────────────────

test: ## Run test suite with coverage
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=ace --cov-report=term-missing

lint: ## Run ruff + mypy
	$(PYTHON) -m ruff check ace/ tests/ scripts/
	$(PYTHON) -m mypy ace/ --ignore-missing-imports

format: ## Auto-format with black + ruff
	$(PYTHON) -m black ace/ tests/ scripts/
	$(PYTHON) -m ruff check --fix ace/ tests/ scripts/

# ── Cleanup ───────────────────────────────────────────────────────────

clean: ## Remove caches & build artefacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/
