# ACE development plan

Status date: 2026-09-12. This plan preserves the vision in `architecture.txt`.
The implementation order separates a useful local coding release from later
architecture experiments. A stage is complete only when its acceptance checks
pass; an empty file or configuration switch is not an implemented capability.

## Constraints and decisions

- Initial hardware assumption: CPU inference on an 8–16 GB RAM laptop. Validate
  real latency and peak process memory on the target machine before release.
- Spend no money automatically. Training defaults to small models and bounded
  sessions. Free-tier availability is not guaranteed and is not a compute budget.
- Preserve the custom ACE research architecture. For useful near-term coding
  capability, also investigate a separately labeled pretrained-model adapter
  track; its weights must not be presented as native ACE checkpoints.
- Track exact parameter counts, measured RAM, task quality and tokens/second.
  Larger parameter count is not a success metric by itself.
- Weight merging is a CPU/data-movement operation, not compute-free training.
  Expert duplication can grow capacity but cannot create knowledge.

## Milestones

| Stage | Deliverable | Acceptance gate | State |
|---|---|---|---|
| M0: correctness | Lossless tokenizer, causal attention, truthful feature flags and counts | Unicode/formatting regressions, cached-chunk equivalence, unsupported settings rejected | Implemented; CPU regression tests |
| M1: offline foundation | Corpus preparation, train/resume, portable bundles, generation, evaluation | Complete command-line workflow; optimizer learns a simple pattern; exact CPU resume | Implemented; workflow verification |
| M2: economical reuse | Compatible ACE merging and dense-to-MoE initialization | Reject incompatible parents; arithmetic correctness; initial upcycled outputs preserved | CPU primitives implemented; task quality unproven |
| M3: useful coding model | Real licensed corpus, coding/SFT pipeline, reproducible external-base candidate | Held-out execution results improve over baseline, without unacceptable regression | Pending |
| M4: laptop release | Quantized runtime, resource profiler, stable packaging and local interface | Proposed target: peak RAM <=4 GB on 8 GB host; latency measured on named device | Pending |
| M5: coding agent | Repository retrieval, patch application, isolated tests and iterative repair | Solve versioned repository tasks with reviewable patches and bounded actions | Pending |
| M6: hybrid research | Native Mamba state cache and measured architecture ablations | Full/cached equivalence, GPU tests, quality and efficiency comparisons | Partial core only |
| M7: multimodal input | Image, then audio/video encoders and token fusion | Paired-data tasks improve; text quality and laptop footprint remain acceptable | Pending |
| M8: planning/adaptation | World model, persistent memory, task adaptation | Better actual task outcomes than retrieval/tool-loop baselines | Pending |
| M9: generative media | Image/video decoders and output interfaces | Valid media output, measured quality, deployable resource footprint | Pending |

M4's RAM target is a proposed acceptance budget, not a measured guarantee.
Multimodal generation may require separate optional models or remote execution;
requiring every module simultaneously on a standard laptop conflicts with the
current budget. Decide the exact release feature set after M3/M4 measurements.

## Next work, in dependency order

1. Choose a target device, RAM allowance, desired coding languages and initial
   use cases (completion, explanation, patching). Record a measurable baseline.
2. Replace the toy corpus with an auditable dataset. Record repository/source
   identity, revisions, licenses, language mix and split assignments. Implement
   near-duplicate detection and repository-level holdouts before reporting scores.
3. Freeze a tokenizer version. Add source-code fill-in-the-middle transforms,
   chat formatting, assistant-only label masking and a real SFT entry point.
4. Build a held-out task runner with isolated code execution, time/memory limits,
   pass/fail summaries and no evaluation-data leakage. Keep language loss and
   execution success as distinct metrics.
5. Select one small pretrained coding baseline after checking its model card,
   license and actual laptop footprint. Preserve its native tokenizer/runtime.
   Add LoRA/QLoRA adaptation in an explicitly separate backend.
6. Compare base, each fine-tune and merged checkpoints. Retain a merge only if it
   improves the agreed task mix. Save parent revisions and merge recipes.
7. Add local quantization with reload and quality tests. A custom ACE checkpoint
   is not automatically supported by llama.cpp/GGUF. Implement a compatible
   converter/runtime or retain the supported PyTorch path.
8. Add retrieval and a controlled edit/test loop before a learned world model.
   Keep execution isolation and explicit file/tool permissions in the agent design.

## Research tracks that retain the original vision

**Mamba:** benchmark native kernels separately from the GRU substitute. Implement
convolution and SSM state caches, backend-specific checkpoint migration, padded
sequence semantics and multi-step cache equivalence. Avoid general initializers
overwriting native SSM-specific parameters.

**MoE:** measure expert load, routing entropy, dropped tokens (if capacity limits
are introduced), peak RAM and effective throughput. Compare dense, duplicated
experts, trained experts and optimized dispatch. Duplicate experts are an
initialization, not specialists. Expert offloading trades RAM against I/O latency.

**RetNet/RWKV/Hyena/BigBird:** prototype one mechanism at a time with identical
data, token budgets and evaluation. Keep it only when it solves a measured
problem. Do not assume their theoretical sequence costs describe the whole
hybrid model or guarantee million-token usable context.

**Transformer²:** define the adaptation objective and trainable components,
then test task adaptation, regressions and runtime overhead. A boolean cannot
enable adaptation without its method, training data and evaluation.

**World model:** start with predicting test/build outcomes of code edits, using
recorded executions as supervision. Compare planning against an ordinary
test-feedback loop. Measure calibration and action-selection improvements.

**Multimodal:** start with screenshot understanding for coding tasks, keeping
the vision encoder optional. Audio and video need separate encoders, paired data,
temporal processing and explicit memory budgets. Media generation is a separate
training and deployment problem, not a language-head extension.

## Definition of a finished first release

- Install and run on a clean, specified laptop without cloud credentials.
- Include a versioned model artifact, tokenizer, provenance, model card and
  reproducible evaluation report.
- Support the chosen coding tasks with measured correctness and resource use.
- Resume supported training runs and load artifacts without silent backend drift.
- Run automated correctness checks and a separately reported coding benchmark.
- Document remaining limitations; do not claim general autonomy, multimodality,
  adaptation, or long-context performance based on scaffolding.

Finishing M1 establishes an engineering foundation. It does not imply that the
full vision, or the training of a useful coding model, is complete.
