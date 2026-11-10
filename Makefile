.PHONY: help install prepare-data pretrain eval generate test lint format

PYTHON ?= python
CONFIG ?= configs/ace_smoke.yaml
DATA ?= data/demo
RUN ?= checkpoints/demo
MODEL ?= $(RUN)/bundle-00000020

help:
	@echo "See Readme.md for portable setup, training, resumption and merge commands."

install:
	$(PYTHON) -m pip install -e ".[dev]"

prepare-data:
	$(PYTHON) -m scripts.prepare_data --input examples/tiny_code.jsonl --output $(DATA) --config $(CONFIG) --validation-fraction 0.2 --source-license project-authored-workflow-examples

pretrain:
	$(PYTHON) -m scripts.pretrain --data $(DATA) --output $(RUN) --max-steps 20 --warmup-steps 2 --seq-len 64 --gradient-accumulation-steps 1

eval:
	$(PYTHON) -m scripts.evaluate --model $(MODEL) --data $(DATA) --seq-len 64

generate:
	$(PYTHON) -m ace.inference.generate --model $(MODEL) --prompt "def add(a, b):" --max-new-tokens 20

test:
	$(PYTHON) -m pytest tests -q

lint:
	$(PYTHON) -m ruff check ace scripts tests
	$(PYTHON) -m mypy ace scripts --ignore-missing-imports

format:
	$(PYTHON) -m black ace scripts tests
	$(PYTHON) -m ruff check --fix ace scripts tests
