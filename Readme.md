# ACE — Autonomous Coding Engine

ACE is an experimental decoder-only language model with configurable attention,
Mamba and sparse expert blocks. This repository now supports an offline path from
local documents to a small trained checkpoint, evaluation and CPU generation.
It does **not** yet contain a generally capable coding assistant or the completed
multimodal/world-model system. The bundled corpus is for workflow verification.

Start with [the development plan](docs/DEVELOPMENT_PLAN.md),
[architecture and file formats](docs/TECHNICAL_GUIDE.md), and
[the low-compute strategy](docs/COMPUTE_AND_MERGING.md).
Recorded checks and the small CPU run are in [VALIDATION.md](docs/VALIDATION.md).

## Install

Use Python 3.11 or 3.12. From the repository root:

```text
python -m venv .venv
```

Activate it using `.venv\Scripts\Activate.ps1` in Windows PowerShell, or
`source .venv/bin/activate` on Linux/macOS. Then:

```text
python -m pip install -e ".[dev]"
```

The base package no longer requires CUDA compilation, FlashAttention, Mamba,
Triton, or bitsandbytes. Install the appropriate PyTorch build for your device
using [PyTorch's installer](https://pytorch.org/get-started/locally/).
The old `requirements.txt` is a historical Windows environment snapshot, not
the portable installation contract. Use `pyproject.toml`.

## Run the complete small workflow

These commands use project-authored examples and require no external datasets,
accounts, model downloads or API keys. Choose new output directories for each
new experiment. Commands intentionally fit Windows and Linux shells without
shell-specific line continuations.

```text
python -m scripts.prepare_data --input examples/tiny_code.jsonl --output data/demo --config configs/ace_smoke.yaml --validation-fraction 0.2 --source-license project-authored-workflow-examples
python -m scripts.pretrain --data data/demo --output checkpoints/demo --max-steps 20 --warmup-steps 2 --seq-len 64 --gradient-accumulation-steps 1 --checkpoint-every 5 --eval-every 10 --device cpu
python -m scripts.evaluate --model checkpoints/demo/bundle-00000020 --data data/demo --seq-len 64 --output checkpoints/demo/evaluation.json
python -m ace.inference.generate --model checkpoints/demo/bundle-00000020 --prompt "def add(a, b):" --max-new-tokens 20
```

Twenty steps establish that the pipeline works; expect poor generated code.
The evaluator reports token-weighted cross-entropy, perplexity and next-token
accuracy. These are language metrics, not executable coding benchmarks.

Run verification with:

```text
python -m pytest tests -q
python -m ruff check ace scripts tests
```

## Train on your data

Pass explicit UTF-8 files or JSONL records containing `{"text": "..."}`. Each
plain file is one document; each JSONL row is one document. Include enough
independent documents to populate both splits. Provide the corpus's actual
source/license description; the program records it but does not establish rights.

```text
python -m scripts.prepare_data --input my_corpus.jsonl --output data/code-v1 --config configs/ace_laptop.yaml --source-license YOUR_SOURCE_AND_LICENSE
python -m scripts.pretrain --data data/code-v1 --output checkpoints/code-v1 --max-steps 200 --max-seconds 900 --seq-len 256
```

`ace_laptop.yaml` has 5,573,888 parameters at its configured 4,096-token vocabulary.
It is a small dense experimental baseline, not a pretrained
replacement for a commercial coding model. Data preparation writes its actual
tokenizer vocabulary into `data/code-v1/config.yaml`; training uses that file.
The default training guard rejects models over 50 million parameters **before**
allocating weights. An explicit `--max-parameters` override is needed for larger
experiments. Parameter count alone does not bound activation or optimizer memory.

## Resume interrupted sessions

Run the same training command with `--resume`. Keep the original data, seed,
batch size, context and learning-rate schedule. `--max-seconds` may change;
`--max-steps` remains the original total schedule horizon.

```text
python -m scripts.pretrain --data data/code-v1 --output checkpoints/code-v1 --max-steps 200 --max-seconds 900 --seq-len 256 --resume
```

The time limit is checked between optimizer steps; a long step, evaluation,
or checkpoint write can exceed it. Abrupt notebook termination loses work
since the last complete checkpoint. CPU tests verify exact resumption with
the supplied sampler, zero workers and unchanged settings. Cross-device or
cross-version bitwise reproducibility is not promised.

Exports are named `bundle-STEP`, using the final completed optimizer step.
Copy the whole bundle to another machine with ACE installed and use the same
generation command. No optimizer state is required for inference.

## Continue a base model and merge compatible runs

Use `--init-from BASE_BUNDLE` to start a new training run from saved ACE weights.
Use the base bundle's config and tokenizer when preparing new data:

```text
python -m scripts.prepare_data --input specialist.jsonl --output data/specialist --config BASE_BUNDLE/config.yaml --tokenizer BASE_BUNDLE/tokenizer --source-license YOUR_SOURCE_AND_LICENSE
python -m scripts.pretrain --data data/specialist --output checkpoints/specialist --init-from BASE_BUNDLE
```

Train two specializations from the same base to retain common lineage. Then:

```text
python -m ace.merging --left LEFT_BUNDLE --right RIGHT_BUNDLE --output checkpoints/merged --alpha 0.5
```

The implementation processes safetensors shards on CPU and checks architecture,
tokenizer fingerprint, backend, lineage, tensor names, shapes and dtypes. It
interpolates both weights and biases. It does not import arbitrary Hugging Face
models into ACE and does not increase parameter count. Evaluate each parent and
the merged model on the same held-out tasks before adopting the result.

## Increase stored capacity without a training pass

Starting with a **dense** ACE bundle, duplicate selected FFNs into experts:

```text
python -m scripts.upcycle --model DENSE_BUNDLE --output checkpoints/expanded --layers 4 5 --experts 4 --top-k 2
```

Layer indices start at zero. Initial evaluation-mode language outputs are
preserved within floating-point tolerance because all experts start identical.
This adds capacity and storage requirements, **not newly learned skills**.
Specialization requires continued training; more total parameters also consume
more laptop memory. The expert router is initialized anew and needs evaluation.

## Colab, Kaggle and other compute

Use the same module commands from this repository in a notebook terminal or
with a notebook `!` prefix. See the platform checklist and limitations in
[COMPUTE_AND_MERGING.md](docs/COMPUTE_AND_MERGING.md). No paid resources are started
by these commands. The old launcher notebook is superseded by these commands.

## Current boundaries

- Text input and language output only; batch size one for interactive generation.
- Native Mamba training is optional. The GRU fallback is explicitly a different
  backend, requires training opt-in, and cannot substitute for native weights.
- Hybrid generation recomputes the full prefix; recurrent-state caching is pending.
- RetNet, Transformer² and world-model flags fail explicitly until implemented.
- Agent execution, instruction tuning, external model conversion, quantization,
  GGUF export, image/audio/video processing and production serving remain planned.
- Existing legacy tokenizers without all 256 byte symbols must be retrained.
  Never replace a trained model's tokenizer with a different token-ID mapping.
