# ACE technical guide

## Implemented model

`AceModel` embeds token IDs, applies configurable pre-norm residual blocks,
normalizes the result and projects to vocabulary logits. A block uses either
grouped-query causal attention with rotary position embeddings, or the selected
Mamba implementation. Its feed-forward path is dense SwiGLU or top-k routed
SwiGLU experts. Embedding/output weights are tied by default.

The trained objective is shifted next-token cross-entropy plus the MoE router's
load-balancing loss. Chat, code, thinking and tool special tokens are vocabulary
markers, not trained reasoning, instruction-following or tool-use capabilities.

The laptop baseline is dense to establish correctness and a cheap reference.
The smoke preset includes an MoE layer to exercise conditional computation.
Native Mamba and its GRU fallback are different architectures; bundles record
which backend produced the weights. Silent backend substitution is rejected.
RetNet, Transformer² and world-model settings raise until those modules exist.

Attention retains causality when a custom padding/additive mask is supplied,
including cached multi-token chunks. Right-padding masks use shape `(B,1,1,L)`.
Generation uses KV caching for attention-only models. Hybrid generation uses
full prefixes because recurrent-state caching is not yet implemented. No
million-token context or trained 32K-context capability is claimed.

## Data preparation

Inputs are explicit UTF-8 files or JSONL rows with a `text` string. The pipeline:

1. Hashes document bytes and removes exact duplicates.
2. Assigns documents to train/validation using SHA-256 of seed and content hash.
3. Trains byte-level BPE only on training documents, or reuses a supplied base
   tokenizer. All 256 byte symbols are included. Unicode normalization is disabled
   for newly trained tokenizers to preserve source text.
4. Writes BOS/document/EOS token streams to little-endian uint32 `.bin` files.
5. Writes actual vocabulary size into `config.yaml` and records source/token
   counts, source descriptions and SHA-256 fingerprints in `manifest.json`.

`PackedTokenDataset` memory-maps the binary stream. Adjacent windows overlap
one token so all next-token targets are covered. `CausalCollator` right-pads
the final short window and sets padded labels to `-100`.

The packed stream can contain more than one document. Current training permits
attention across document boundaries inside a window and learns EOS-to-BOS
transitions. Repository-level isolation, near-duplicate detection, richer data
quality checks and isolated-document attention are not implemented. Hash-based
splitting prevents exact document overlap, not broader benchmark contamination.

Input processing is streaming by document, but a single plain-text document is
read into RAM and deduplication holds one hash per unique document. Use JSONL
documents of manageable size. Preparation is not yet a distributed corpus job.

Prepared data layout:

```text
data/experiment/
  manifest.json             # corpus and tokenizer identity
  config.yaml               # exact vocabulary-adjusted model config
  tokenizer/tokenizer.json
  tokenizer/special_tokens_map.json
  tokenizer_corpus.txt      # training-only tokenizer input
  train.jsonl
  validation.jsonl
  train.bin                 # uint32 IDs
  validation.bin
```

## Training and resumption

The CLI validates vocabulary/fingerprints, checks actual parameter count on the
metadata-only device, then creates a float32 model. AdamW, cosine decay, warmup,
gradient clipping and gradient accumulation are supplied by `AceTrainer`.
BF16 autocast is enabled only on supported CUDA devices. It is not FP16 training.

The resumable sampler stores seed, epoch and next sample position. A checkpoint
contains model, optimizer, scheduler, config, optimizer step, CPU/CUDA RNG state
and sampler state. Writes use a temporary file followed by replacement; the last
three complete checkpoints are retained. Checkpoints use restricted
`torch.load(weights_only=True)` loading.

For exact CPU resumption, retain the same dataset, model, seed, batch and context,
schedule and library/device environment. Keep `num_workers=0` to avoid prefetching
the sampler beyond the committed batch. The CLI stores these identities in
`run.json` and rejects changed schedules. Use `--init-from` for a new training
schedule starting from a bundle; its optimizer starts afresh.

Gradient accumulation averages micro-batch mean losses. A short final window
therefore has the same micro-batch weight; standalone evaluation instead sums
loss over all non-padding target tokens and divides by the actual token count.

## Portable model format

```text
bundle-STEP/
  manifest.json
  config.yaml
  tokenizer/tokenizer.json
  tokenizer/special_tokens_map.json
  weights-00000.safetensors
  weights-00001.safetensors   # additional shards only when needed
```

Manifest version 1 records:

- full model config, implementation/backend identity and parameter count;
- tokenizer SHA-256, tensor-to-shard mapping and each shard's SHA-256;
- lineage ID identifying a common base for compatible fine-tunes;
- operation provenance and an explicit unverified task-quality marker.

Tied output weights are stored once as the embedding; loading restores the tie.
Inference creates the model on CPU, loads its state strictly, then moves it to
the requested device. This is intended for laptop-sized artifacts; loading has
additional transient memory beyond the final model footprint. Cross-version
compatibility requires tests and is not guaranteed solely by manifest version.

CPU merging reads mapped safetensors and writes shards, with float32 arithmetic
for floating parameters. Config, tokenizer identity, backend, lineage and tensor
layouts must agree. A lineage string records provenance, not mathematical proof
that a merge will preserve either parent's capability.

Dense-to-MoE expansion copies each selected FFN into independent experts, adds
a small seeded router and records a new lineage. Identical experts preserve the
evaluation-mode FFN output under normalized routing weights; dropout during
training and subsequent optimization can change that equivalence.

## Verification and remaining risks

Regression coverage includes unseen Unicode/bytes, causal masks and cached
chunks, exact deduplication/split separation, packed target accounting, bundle
reload, deterministic generation, merge arithmetic/rejection, upcycling output
equivalence, optimizer learning and exact CPU training resumption.

The command-line demo uses only 30 authored documents. Its learned checkpoint
proves the mechanics, not general coding ability. Native CUDA Mamba, GPU mixed
precision, long context, external model conversion, useful SFT, quantized serving
and agent execution remain unvalidated or unimplemented. See the milestone plan.
