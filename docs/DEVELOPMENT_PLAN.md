# ACE development plan

Status date: 2026-11-10. ACE maintains two coding tracks: a native model trained from scratch and a separate LoRA adapter over Qwen2.5-Coder-0.5B-Instruct. Neither track is considered useful until it has held-out compile/test results and laptop resource measurements.

## Constraints and decisions

- Target CPU inference on the current Windows 11 laptop with about 16 GB RAM. Measure process memory and actual tokens per second; the initial process-memory goal is 4 GB.
- Both tracks address completion and bug fixing in Python, JavaScript, TypeScript, Java, Rust, C and C++.
- The native ACE tokenizer is versioned and retains compatibility with legacy token IDs. The pretrained model uses and records its own tokenizer and exact base revision.
- Repository inputs are checked against MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause and ISC. Commit, license and file hashes are recorded, with repository-level splits and license snapshots.
- Colab and Kaggle jobs remain manually started, interruptible and resumable. Paid compute is a user-selected provider, instance and spending cap; this repository does not provision or bill resources.
- Mamba caching, MoE efficiency, quantization and multimodal work remain separate experiments with their own quality and resource comparisons.

## Delivery status

| Area | Implemented | Remaining acceptance work |
|---|---|---|
| Repository entry points | Smoke config, bundle import, README case and documentation links repaired | Fresh install workflow still needs full run |
| Licensed corpus path | Git source manifest, license allowlist, commit/file provenance, exact deduplication, whole-repository splits and snapshots | Review source revisions and expand/refresh corpus deliberately |
| Native SFT | Versioned FIM tokens, assistant-only labels, prepared data fingerprints, SFT and resume CLI | Train a baseline and compare held-out results |
| Pretrained SFT | Qwen default, exact revision resolution, model tokenizer, LoRA adapter, resume metadata and separate artifacts | Download/run with current Colab/Kaggle GPU; verify reload after export |
| Coding evaluation | Ten completion and ten bug-fix fixtures per language; compile/test reports, language loss, token speed and process RSS | Run on Docker-enabled host; expand synthetic unit tasks into held-out repository tasks |
| Local assistant | CLI, loopback-only service, bounded workspace retrieval, inline completions and reviewable diffs | Install VS Code extension and verify on target laptop |
| Test isolation | Scratch copy, network denial, CPU/memory/time limits, path checks and unavailable dependency reporting | Verify Docker behavior and actual configured test dependencies on each target OS |
| Architecture research | Existing merge/upcycle utilities | Mamba cache equivalence, MoE efficiency, quantization and multimodal comparisons are unstarted |

## Next work

1. Review the configured public repositories and pin their resolved commits in `configs/code_sources.json` for reproducible repeats.
2. Run a bounded baseline on Colab or Kaggle for both tracks. Keep `run.json`, manifests, tokenizer identity, checkpoints, validation loss and training settings.
3. Export the native bundle and Qwen adapter separately. Load both on the Windows CPU laptop and measure peak RSS, generation speed and failure modes.
4. Run all seven language fixture runners in the Docker sandbox. Add project-level held-out completion and bug-fix tasks with repository isolation before treating the small starter suite as a broad score.
5. Build/install the VS Code VSIX locally. Test loopback authentication, ignored-file and secret exclusion, diff preview, explicit apply, container limits and clear missing-dependency reports.
6. Compare every change to the same task revisions and run manifests. Reject changes that improve loss while reducing task pass rates or exceeding the agreed laptop footprint.

## Research tracks

**Mamba:** benchmark native kernels separately from the GRU substitute. Implement convolution and SSM state caches, backend-specific checkpoint migration, padded-sequence semantics and multi-step cache equivalence.

**MoE:** measure expert load, routing entropy, peak memory and throughput. Compare dense, duplicated experts and trained experts. Duplication only initializes experts; specialization requires optimization on training data.

**Quantization:** test one runtime and precision at a time. Keep conversion, reload, pass-rate and RSS results. Do not fold quantization into the first adapter baseline.

**Multimodal:** begin with a separate coding screenshot task and optional vision model. Audio, video and media generation need distinct encoders, datasets and resource budgets.

**World model/adaptation:** compare predicted test/build outcomes against the ordinary patch-and-test loop. Add these components only if held-out task outcomes improve.

## Definition of a finished first release

- Fresh install and local CPU service work without cloud credentials.
- Native and pretrained artifacts remain separate, pinned and reloadable.
- Both tracks have reproducible results on the same held-out compile/test suite, split by language and task type.
- Peak CPU memory and generation speed are measured on the named laptop.
- Workspace retrieval is bounded and patch changes require explicit diff review.
- Tests run in a network-disabled, resource-bounded scratch environment; unavailable runtimes and dependencies are reported.
- Documentation covers local setup, remote kernels, checkpoint recovery and user-controlled paid compute without claiming unmeasured quality.
