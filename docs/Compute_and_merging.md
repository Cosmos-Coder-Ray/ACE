# Compute budget and model reuse

## What can and cannot be free

No method can reliably train arbitrary new capabilities without computation.
Downloading, reading, copying, averaging and running weights all use resources.
The practical goal is **no backpropagation for selected reuse operations**, and
small, measured training jobs for actual learning. Bigger parameter counts
increase storage and usually the RAM needed on the target laptop.

| Method | What changes | Compute requirement | ACE status |
|---|---|---|---|
| Compatible checkpoint interpolation | Blends corresponding weights and biases; same parameter count | CPU arithmetic and disk I/O, no training pass | Implemented for ACE bundles |
| Dense-to-MoE upcycling | Copies learned dense FFNs into more experts | CPU copying, larger output/storage, no initial training | Implemented for dense ACE bundles |
| External pretrained model reuse | Starts with learned capabilities in that model's native architecture | Download/storage and inference; optional later training | Separate backend planned |
| LoRA/QLoRA | Trains a small adapter over a base model | Forward/backward passes and activation memory still required | Planned |
| Distillation | Teaches ACE from another model's outputs | Teacher inference plus student training | Planned; budget must be measured |
| Quantization | Reduces weight precision/storage | Conversion and quality testing; possibly calibration | Planned |
| Retrieval and tool use | Supplies relevant context and executable feedback | Indexing, search, inference and tool execution | Planned |

MergeKit supports CPU and out-of-core model merging, but that is not a guarantee
of cross-architecture compatibility or improved task quality. Its methods are
useful for supported model families; the ACE CLI currently consumes ACE bundles
only. See [MergeKit's implementation and supported methods](https://github.com/arcee-ai/mergekit).

Upcycling reuses existing dense weights as expert initialization. Expert
specialization requires training and evaluation. See the original
[Sparse Upcycling paper](https://arxiv.org/abs/2212.05055).
Low-bit adapter training still trains parameters and retains activation costs;
see [PEFT's quantization guide](https://huggingface.co/docs/peft/main/en/developer_guides/quantization).

## Proposed local targets

The current default is an 8–16 GB RAM laptop, CPU inference, with no assumption
of an NVIDIA GPU. The native ACE baseline is intentionally only a few million
parameters so iteration is possible. That size is a research starter, not an
assurance of useful general coding quality.

For a later pretrained backend, evaluate a roughly 0.5–3B parameter candidate
with a supported 4-bit runtime. Raw weight storage at 4 bits is P/2 bytes, but
scales, metadata, embeddings, runtime allocations and context caches add memory.
Measure the full process. Do not promise that all nominal 3B models fit or run
fast on all laptops. Native ACE's current portable bundles contain float weights;
the project has not implemented a 4-bit ACE runtime.

Approximate planning arithmetic for P float32 parameters:

- Weights alone: 4P bytes.
- Gradients plus Adam first/second moments: about another 12P bytes.
- Activations, temporary tensors, attention, MoE dispatch and runtime: additional.

This is a rough floor for the current AdamW training path, not a peak-memory
estimator. The CLI's parameter guard protects against accidental huge models,
but does not guarantee that a chosen batch/context fits memory.

## Platform workflow

**Local CPU:** prepare datasets, train/tokenizer-test the smoke model, run unit
tests, merge compatible bundles, test upcycling and measure inference. No cloud
account is needed. Default to two PyTorch CPU threads for small experiments.

**Colab:** clone/upload the repository, select an available GPU runtime if needed,
install the appropriate PyTorch build and `pip install -e .`, then use the README
module commands. Put datasets and the complete run directory on storage that
survives runtime loss; copy them back before resuming. Save every 25–100 steps
after measuring checkpoint cost. Start with batch size 1, context 128/256, and
gradient accumulation 4. The trainer uses BF16 only when the GPU reports support;
otherwise it keeps float32 rather than assuming all free GPUs support BF16.

Google states that free resources, limits and available hardware vary. Plan
interruptible sessions, not a guaranteed multi-hour allocation. See the
[Colab FAQ](https://research.google.com/colaboratory/faq.html).

**Kaggle:** run the same code in a notebook environment, using the platform's
currently available accelerator and persistent output/version mechanism. Verify
the current account quota before a run. Do not rely on a fixed free-GPU-hour
number. Resume requires all run metadata and the latest complete checkpoint.

**AWS/paid tiers:** use only an explicitly chosen instance, region and spending
cap after estimating a small pilot. Check current price and storage charges;
the repository does not create instances or start billed jobs.

**TPU:** unsupported by the current trainer. TPU use requires a deliberate
PyTorch/XLA training path, compatible operations and checkpoint tests. Availability
in a notebook does not make CUDA Mamba or this eager-PyTorch trainer TPU-compatible.

## A realistic larger-capability strategy

1. Select and license-check a small pretrained coding model whose native runtime
   runs on the target laptop. Freeze its model and tokenizer revisions.
2. Establish coding accuracy, RAM and speed before modification.
3. Adapt that same base for separate skills with small LoRA/QLoRA runs. Track
   training data, budget, validation metrics and exact base identity.
4. Merge compatible fine-tunes or adapters within the supported model family.
   Compare the result against each parent; revert if one skill regresses too far.
5. Quantize and re-evaluate on the target device. Use retrieval/tool execution
   for project-specific context and verification instead of increasing weights
   whenever those methods solve the task more cheaply.
6. To transfer behavior into native ACE, use an explicitly budgeted distillation
   experiment. There is no implemented universal weight/bias translation between
   unrelated tokenizers, hidden dimensions, layer types and expert layouts.

ACE's own upcycling is an additional research branch. It grows dense ACE
checkpoints into a sparse model with initially equivalent language outputs.
Increasing the number of copied experts is not evidence of acquired skills.

## Merge and expansion acceptance checks

- Use models descended from the same initialization/base, with identical tensor
  meanings and tokenizer IDs. Matching shapes alone are insufficient.
- Preserve parent model/data provenance and confirm distribution permissions.
- For interpolation, evaluate alpha endpoints and a small fixed grid; record
  the chosen alpha and every result rather than cherry-picking a single task.
- For upcycling, verify initial output equivalence, parameter/storage growth,
  expert utilization after training and actual improvement per unit of compute.
- Keep the previous working bundle and never overwrite it in place.
- Sharded CPU operations bound tensor buffering to a shard plus the largest
  tensors; memory mapping and the OS file cache still consume resources.
