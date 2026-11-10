"""LoRA fine-tuning for a pinned external Transformers coding model."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ace.data.preprocessing import file_sha256
from ace.data.samplers import ResumableRandomSampler
from ace.data.sft_manifest import read_sft_manifest
from ace.data.transformer_sft import TransformersSFTCollator, TransformersSFTDataset

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared SFT directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default="main", help="Branch, tag, or exact model commit")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def _tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def _resolved_revision(model: Any, tokenizer: Any) -> str:
    candidates = [
        getattr(model.config, "_commit_hash", None),
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
    ]
    pinned = [
        revision for revision in candidates
        if isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision)
    ]
    if len(set(pinned)) > 1:
        raise ValueError("Model and tokenizer resolved to different repository revisions")
    if pinned:
        return pinned[0]
    raise RuntimeError("The model download did not expose an exact 40-character revision")


def _lora_config() -> Any:
    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise ImportError("Install the optional training dependencies with pip install -e '.[finetune]'") from exc
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM",
        bias="none",
    )


def _evaluate(model: Any, loader: DataLoader[Any], device: torch.device, amp_dtype: torch.dtype | None) -> float:
    model.eval()
    loss_sum = 0.0
    token_count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            autocast = torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.float32,
                enabled=amp_dtype is not None,
            )
            with autocast:
                result = model(**batch)
            targets = int((batch["labels"][:, 1:] != -100).sum())
            loss_sum += float(result.loss) * targets
            token_count += targets
    model.train()
    if token_count == 0:
        raise ValueError("Validation data has no target tokens")
    return loss_sum / token_count


def _save_checkpoint(
    model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    sampler: ResumableRandomSampler,
    checkpoint_dir: Path,
    step: int,
    loss: float,
    device: torch.device,
) -> Path:
    target = checkpoint_dir / f"step_{step:08d}"
    if target.exists():
        return target
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"step-{step:08d}-", dir=checkpoint_dir) as temp:
        temp_dir = Path(temp)
        model.save_pretrained(temp_dir / "adapter")
        extra: dict[str, Any] = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "sampler": sampler.state_dict(),
            "rng_cpu": torch.get_rng_state(),
            "step": step,
            "loss": loss,
        }
        if device.type == "cuda":
            extra["rng_cuda"] = torch.cuda.get_rng_state_all()
        torch.save(extra, temp_dir / "state.pt")
        temp_dir.rename(target)
    return target


def _latest_checkpoint(checkpoint_dir: Path) -> Path:
    checkpoints = sorted(checkpoint_dir.glob("step_[0-9]*"))
    if not checkpoints:
        raise FileNotFoundError(f"No adapter checkpoints in {checkpoint_dir}")
    return checkpoints[-1]


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if min(
        args.seq_len,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.max_steps,
        args.checkpoint_every,
        args.eval_every,
        args.threads,
    ) < 1 or args.max_seconds <= 0 or args.lr <= 0:
        raise ValueError("Training lengths, counts, learning rate, and max-seconds must be positive")
    if args.resume and args.output.exists():
        saved_run = json.loads((args.output / "run.json").read_text(encoding="utf-8"))
        if saved_run["model_id"] != args.model_id:
            raise ValueError("Resume requires the original base model ID")
        if args.revision not in {"main", saved_run["resolved_revision"]}:
            raise ValueError("Resume requires the original base model revision")
        revision = saved_run["resolved_revision"]
    else:
        revision = args.revision

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device_name = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_name)
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None

    try:
        from peft import PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install the optional training dependencies with pip install -e '.[finetune]'") from exc

    tokenizer_source = args.output / "tokenizer" if args.resume else args.model_id
    tokenizer_revision = None if args.resume else revision
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, revision=tokenizer_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=revision,
        torch_dtype=amp_dtype or torch.float32,
        low_cpu_mem_usage=True,
    )
    resolved_revision = saved_run["resolved_revision"] if args.resume else _resolved_revision(
        model, tokenizer
    )
    if args.resume and _resolved_revision(model, tokenizer) != resolved_revision:
        raise ValueError("Loaded base model does not match the saved revision")
    if args.resume:
        tokenizer_hash = file_sha256(args.output / "tokenizer" / "tokenizer.json")
        model = PeftModel.from_pretrained(
            model,
            _latest_checkpoint(args.output / "checkpoints") / "adapter",
            is_trainable=True,
        )
    else:
        model = get_peft_model(model, _lora_config())
        args.output.mkdir(parents=True, exist_ok=False)
        tokenizer.save_pretrained(args.output / "tokenizer")
        tokenizer_hash = file_sha256(args.output / "tokenizer" / "tokenizer.json")

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("The selected pretrained tokenizer must define a chat template")
    for token in ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise ValueError(f"The selected model tokenizer does not contain {token}")

    train_path = args.data / "train.jsonl"
    validation_path = args.data / "validation.jsonl"
    data_manifest = read_sft_manifest(args.data)
    data_manifest_sha256 = file_sha256(args.data / "manifest.json")
    identity = {
        "model_id": args.model_id,
        "resolved_revision": resolved_revision,
        "tokenizer_sha256": tokenizer_hash,
        "train_sha256": file_sha256(train_path),
        "validation_sha256": file_sha256(validation_path),
        "data_manifest_sha256": data_manifest_sha256,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "lr": args.lr,
    }
    if args.resume:
        if saved_run["identity"] != identity:
            raise ValueError("Resume requires the same base model, data, tokenizer, and schedule")
        run = saved_run
    else:
        run = {"model_id": args.model_id, "resolved_revision": resolved_revision, "identity": identity}
        (args.output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")

    model.to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
    )
    warmup_steps = max(1, args.warmup_steps)

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return max(1, step) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, args.max_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    train_data = TransformersSFTDataset(train_path, tokenizer, args.seq_len)
    validation_data = TransformersSFTDataset(validation_path, tokenizer, args.seq_len)
    collator = TransformersSFTCollator(int(tokenizer.pad_token_id))
    sampler = ResumableRandomSampler(len(train_data), args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=0,
    )

    step = 0
    last_loss = 0.0
    if args.resume:
        checkpoint = _latest_checkpoint(args.output / "checkpoints")
        state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        sampler.load_state_dict(state["sampler"])
        torch.set_rng_state(state["rng_cpu"])
        if device.type == "cuda" and "rng_cuda" in state:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        step = int(state["step"])
        last_loss = float(state["loss"])

    started_at = time.monotonic()
    iterator = iter(train_loader)
    model.train()
    while step < args.max_steps and time.monotonic() - started_at < args.max_seconds:
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(args.gradient_accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.float32,
                enabled=amp_dtype is not None,
            ):
                loss = model(**batch).loss
            (loss / args.gradient_accumulation_steps).backward()
            accumulated_loss += float(loss.detach()) / args.gradient_accumulation_steps
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
        )
        optimizer.step()
        scheduler.step()
        step += 1
        last_loss = accumulated_loss

        if step % args.eval_every == 0 or step == args.max_steps:
            validation_loss = _evaluate(model, validation_loader, device, amp_dtype)
            logging.info("step=%d loss=%.5f validation_loss=%.5f", step, last_loss, validation_loss)
        if step % args.checkpoint_every == 0:
            _save_checkpoint(
                model,
                optimizer,
                scheduler,
                sampler,
                args.output / "checkpoints",
                step,
                last_loss,
                device,
            )

    final_loss = _evaluate(model, validation_loader, device, amp_dtype)
    _save_checkpoint(
        model,
        optimizer,
        scheduler,
        sampler,
        args.output / "checkpoints",
        step,
        last_loss,
        device,
    )
    adapter_path = args.output / "adapter"
    model.save_pretrained(adapter_path)
    manifest = {
        "format_version": 1,
        "model_id": args.model_id,
        "model_revision": resolved_revision,
        "tokenizer_sha256": tokenizer_hash,
        "adapter_sha256": _tree_sha256(adapter_path),
        "train_sha256": identity["train_sha256"],
        "validation_sha256": identity["validation_sha256"],
        "data_manifest_sha256": data_manifest_sha256,
        "source_code_manifest_sha256": data_manifest["source_code_manifest"]["sha256"],
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": LORA_TARGETS},
        "training_settings": {
            **identity,
            "max_seconds_per_run": args.max_seconds,
            "checkpoint_every": args.checkpoint_every,
            "eval_every": args.eval_every,
            "device": str(device),
            "threads": args.threads,
        },
        "libraries": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "peft": __import__("peft").__version__,
        },
        "steps": step,
        "quality_verified": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = {"final_step": step, "final_train_loss": last_loss, "validation_loss": final_loss}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({**report, "adapter": str(adapter_path)}, indent=2))


if __name__ == "__main__":
    main()
