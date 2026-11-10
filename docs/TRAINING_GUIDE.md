# ACE coding model training guide

This guide covers two separate, equally evaluated tracks:

1. Native ACE: train the small ACE architecture from scratch, then instruction/FIM tune it.
2. Pretrained: LoRA fine-tune Qwen2.5-Coder-0.5B-Instruct with its own tokenizer and FIM tokens.

The Qwen model card lists 0.49B parameters, Apache-2.0, and Transformers support. The adapter workflow resolves and records an exact model commit and tokenizer hash in each run. See the [model card](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) and the [Qwen FIM token definitions](https://github.com/huggingface/Qwen2.5-Coder/blob/main/README.md).

The initial target is CPU inference on the Windows 11 laptop with about 16 GB RAM. Measure peak process memory and tokens per second on that device; do not infer performance from parameter count. Training commands are interruptible. This repository does not start cloud jobs or provision paid compute.

## 1. Install locally

From Windows PowerShell in the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,finetune]"
```

Use Python 3.11 or 3.12. For CPU inference, use the CPU PyTorch build installed for this machine. The `finetune` extra contains Transformers, PEFT and Accelerate. The optional `finetune-cuda` extra adds bitsandbytes for separate CUDA experiments; it is not required by these commands.

## 2. Prepare a lawful repository corpus

Review [`configs/code_sources.json`](../configs/code_sources.json) before running. It has train and validation projects for Python, JavaScript, TypeScript, Java, Rust, C and C++. The preparer accepts only MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause and ISC SPDX expressions. It verifies each declared license file, excludes generated/vendor/build directories, caps file size, deduplicates exact file content, and keeps entire repositories in one split. It writes resolved commit IDs, file hashes, license hashes and license snapshots to the corpus manifest.

The source manifest starts from named public repositories. After the first preparation, copy the recorded `resolved_commit` into each source's `revision` field to make later retrievals use the same commits. Keep the generated `manifest.json` and `licenses/` directory with every run. Unclear or incompatible licenses should be excluded rather than guessed.

```powershell
ace-prepare-repositories --sources configs/code_sources.json --output data/coding-corpus --config configs/ace_laptop.yaml
```

The repository preparer needs Git and Internet access. `--output` must be a new directory. The tokenizer is trained on the training repositories only; validation projects are never tokenizer training input.

## 3. Prepare instruction and FIM data

FIM examples are generated deterministically from source files in their existing repository split. The example instruction file contains ACE-authored records under the repository's Apache-2.0 license. Each instruction record identifies its source, project, license and split; projects cannot occur in both splits. Add only records whose source/license are documented.

```powershell
ace-prepare-sft --code-data data/coding-corpus --instruction-data examples/coding_instructions.jsonl --output data/coding-sft --seed 42
```

The output is a distinct supervised dataset with verified train/validation fingerprints, source commit/license metadata and copied license snapshots. Native ACE uses its own versioned tokenizer. The pretrained track uses the Qwen tokenizer saved under its adapter run. Both mask loss outside the assistant response or FIM middle target.

Instruction JSONL rows use this shape:

```json
{"kind":"instruction","split":"train","repository":"project-id","source":"dataset or repository URL","license":"MIT","system":"You are a coding assistant.","user":"Fix the bug described here.","assistant":"Corrected code goes here."}
```

Use different project IDs for training and validation. The command rejects a mixed split for one project or a license outside the allowlist.

## 4. Train native ACE from scratch

Run pretraining on the prepared code stream first. The laptop config is intentionally small; a short run validates mechanics and is not expected to produce a strong model.

```powershell
ace-train --data data/coding-corpus --output artifacts/native-pretrain --max-steps 200 --max-seconds 900 --warmup-steps 10 --seq-len 256 --batch-size 1 --gradient-accumulation-steps 4 --checkpoint-every 25 --eval-every 50 --device cuda
```

For CPU, replace `--device cuda` with `--device cpu`, lower `--seq-len` if memory is tight, and keep `--threads 2`. The CLI reports the final bundle directory; use that actual path below. The current config is dense and under the pretraining CLI's default 50-million-parameter guard.

To resume an interrupted run, use the exact same data, output directory, seed, batch size, context, learning rate and total `--max-steps`. `--max-seconds` may be changed between sessions:

```powershell
ace-train --data data/coding-corpus --output artifacts/native-pretrain --max-steps 200 --max-seconds 900 --warmup-steps 10 --seq-len 256 --batch-size 1 --gradient-accumulation-steps 4 --checkpoint-every 25 --eval-every 50 --device cuda --resume
```

Start supervised tuning from the completed native bundle. Replace `BASE_BUNDLE` with the path printed by pretraining:

```powershell
ace-sft --data data/coding-sft --init-from BASE_BUNDLE --output artifacts/native-sft --max-steps 200 --max-seconds 900 --warmup-steps 10 --seq-len 256 --batch-size 1 --gradient-accumulation-steps 4 --checkpoint-every 25 --eval-every 50 --device cuda
```

Resume SFT with the same command and `--resume`. Keep the `checkpoints/`, `run.json`, dataset manifest and bundle directory together. Each native bundle includes the exact ACE tokenizer and source/training provenance.

## 5. Fine-tune the separate Qwen adapter

The model weights, adapter, saved tokenizer and manifests are kept under a separate directory. The base model is not converted into native ACE weights.

```powershell
ace-finetune-pretrained --data data/coding-sft --output artifacts/qwen-coder-lora --model-id Qwen/Qwen2.5-Coder-0.5B-Instruct --revision main --max-steps 200 --max-seconds 900 --warmup-steps 10 --seq-len 1024 --batch-size 1 --gradient-accumulation-steps 8 --checkpoint-every 25 --eval-every 50 --device cuda
```

The requested `main` revision is resolved to an exact 40-character commit and recorded with the tokenizer SHA-256. For future reproducibility, use that recorded commit as `--revision`. To resume, retain the entire `artifacts/qwen-coder-lora` directory and add `--resume` to the same command. The runner rejects changed data, base revision, tokenizer or schedule.

## 6. Compare both tracks on coding tasks

The checked-in starter suite contains ten completion and ten bug-fix fixtures for each of the seven languages. Each task runs compilation/tests in a disposable Docker container with networking disabled, 2 CPUs, 4 GB RAM and a 5-minute timeout by default. The test runner does not mount the user's project, environment or host credentials. It copies only workspace files that pass ignore/secret filters, with a 256 MiB scratch-copy cap.

Generate the suite once:

```powershell
ace-create-eval-suite --output evaluation/coding_tasks
```

Evaluate one native bundle and the pretrained adapter on the same suite:

```powershell
ace-evaluate-coding --model artifacts/native-sft/bundle-00000200 --suite evaluation/coding_tasks --output reports/native-coding.json --threads 2 --cpus 2 --memory 4g --timeout-seconds 300
ace-evaluate-coding --model artifacts/qwen-coder-lora --suite evaluation/coding_tasks --output reports/qwen-coding.json --threads 2 --cpus 2 --memory 4g --timeout-seconds 300
```

Replace the native bundle path with the actual `bundle-STEP` name. The report records pass rates by language and task type, per-language reference loss, model/tokenizer/data identity, generation speed and peak ACE process memory. Docker must be installed and running. Missing Docker, compilers or test dependencies are reported as unavailable. A failed test is a measured failure, not an unavailable dependency.

The starter suite is unit-level and synthetic. Add realistic held-out repository tasks before using its score as a broad measure of coding ability. Evaluate the base model and every adapter change on the same task revisions.

## 7. Run the local service and VS Code extension

Install the pretrained inference dependencies, then start a loopback-only service. The model is loaded once and the service prints a random bearer token:

```powershell
ace-serve --model artifacts/qwen-coder-lora --workspace .
```

For patch test execution, put an `ace-tests.json` in the workspace root:

```json
{
  "language": "python",
  "test_command": ["python", "-m", "pytest", "-q"],
  "cpus": 2,
  "memory": "4g",
  "timeout_seconds": 300
}
```

The command and workspace are copied into a scratch container; the test command must not depend on host-only environments. Build an image containing project dependencies before disabling network access, or the service reports the missing dependencies. The service never edits workspace files. Patch proposals are returned as diffs and the VS Code extension applies a proposal only after the user reviews and accepts it.

To package and install the extension locally:

```powershell
cd vscode-extension
npm install
npm run package
code --install-extension ace-coding-assistant-0.1.0.vsix
```

In VS Code, run **ACE: Set Service Token** and paste the token printed by `ace-serve`. Inline completions remain suggestions until accepted. Patch actions open a diff; workspace changes require the explicit **Apply Reviewed Patch** action.

The CLI can also call the service. In a separate PowerShell window, set `ACE_SERVICE_TOKEN` to the printed token, then use `ace-agent completion --prompt "..."` or `ace-agent patch --file src/file.py --instructions "..."`.

## 8. Colab, Kaggle and recovery

### Colab from VS Code

The Colab extension selects a remote Colab kernel for the local VS Code notebook. It does not synchronize the local repository into the VM. Clone ACE in the remote terminal or upload the repository, then set the notebook working directory to its root. The [Colab extension quick start](https://marketplace.visualstudio.com/items?itemName=google.colab) documents selecting **Kernel → Colab → Auto Connect**. Save the full run directory to persistent storage or download it before disconnecting.

### Kaggle

Clone/upload ACE into the Kaggle runtime and write runs under `/kaggle/working/ace-run`. Keep `run.json`, `checkpoints/`, prepared data and manifests. Use **Save Version → Quick Save** to snapshot the current work and output files; select output saving in version settings. Use **Save & Run All** only when replaying the whole notebook from a clean session is intended and fits the session allowance. After the version completes, add that notebook's output as an input to a new notebook. Copy the saved run directory from read-only `/kaggle/input` back into `/kaggle/working` and rerun the original CLI with `--resume`. The official [Kaggle Notebooks guide](https://www.kaggle.com/docs/notebooks) describes notebook versions and output files.

### Other paid GPU platforms

Choose a provider, instance and spending cap manually. Check current GPU, RAM, persistent storage, session time and price before starting a pilot. Copy the run directory and data manifests to persistent storage, then use the same commands and `--resume`. No provider is provisioned and no billing is initiated by this project.

## 9. Separate architecture experiments

Keep Mamba caching, MoE dispatch/efficiency, quantization and multimodal input/output in separate runs. Each needs its own quality, memory and speed comparison against the coding baseline. Weight merging and expert expansion can reuse or rearrange compatible weights; they do not create new learned skills without training.
