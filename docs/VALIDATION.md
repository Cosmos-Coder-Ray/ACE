# Foundation validation — 2026-09-12

Environment: Windows, Python 3.12.10, PyTorch 2.11.0+cpu. Checks used the existing
local Python environment against this worktree; no GPU or paid service was used.

| Check | Result |
|---|---|
| Full automated test suite | 169 passed |
| Ruff | Passed |
| Strict mypy across `ace` and `scripts` | Passed, 46 source files |
| Wheel build without dependency downloads | Passed: `ace-0.1.0-py3-none-any.whl` |
| Git whitespace check | Passed |
| Prepare/train/export/evaluate/generate CLI workflow | Completed on CPU |

The tests include regression coverage for cached chunks, masking, unknown bytes,
document splitting, token packing, portable weight loading, deterministic
generation, compatible interpolation, incompatible lineage rejection and
dense-to-MoE output equivalence. Training tests verify that a repeated pattern
can be learned and that an interrupted CPU run resumes to the exact same weights,
including dropout and sample order. A completed resume retains its reported loss.

## Small command-line run

The project-authored example corpus produced 25 training documents and five
validation documents. Its tokenizer contains 506 entries. The smoke model has
131,136 parameters and completed 20 optimizer steps with context length 64.

| Measurement | Value |
|---|---:|
| Training stream tokens | 971 |
| Validation stream tokens | 284 |
| Evaluated next-token targets | 283 |
| Final training total loss, including MoE auxiliary loss | 6.104212 |
| Held-out next-token cross-entropy | 6.133972 |
| Held-out perplexity | 461.264574 |
| Held-out next-token accuracy | 1.7668% |

Generation ran successfully but produced repetitive, poor output. These results
validate mechanics, not useful coding ability. This tiny corpus is not an
independent coding benchmark and no broader generalization claim follows.

Local artifacts are under `data/foundation-demo`,
`checkpoints/foundation-demo/bundle-00000020`, and `dist/`. They are intentionally
ignored by Git. The README commands reproduce the workflow in fresh directories;
small numerical or tokenizer-ID differences across environments remain possible.

## Not validated yet

Native Mamba/CUDA, GPU mixed precision, long context, TPU training, laptop peak
memory/latency targets, external pretrained-model conversion, merged task quality,
quantized deployment, real coding benchmarks, agent actions and multimodal
capabilities. These remain explicit milestones in the development plan.
