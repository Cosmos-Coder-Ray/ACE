"""Instruction and fill-in-the-middle fine-tuning for native ACE bundles."""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ace.data.collator import CausalCollator
from ace.data.preprocessing import file_sha256
from ace.data.samplers import ResumableRandomSampler
from ace.data.sft_manifest import read_sft_manifest
from ace.data.supervised import SupervisedDataset
from ace.inference.bundle import backend_name, load_bundle, save_bundle
from ace.model.backbone import AceModel
from ace.model.config import AceConfig
from ace.tokenizer.ace_tokenizer import AceTokenizer
from ace.training.trainer import AceTrainer, TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared SFT directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--init-from", type=Path, help="Start from a native ACE bundle")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.resume and args.init_from:
        raise ValueError("Use either --resume or --init-from")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    if args.init_from:
        model, tokenizer, base = load_bundle(args.init_from)
        config = model.config
        if args.config and asdict(AceConfig.from_yaml(args.config)) != asdict(config):
            raise ValueError("The supplied config does not match the base bundle")
        if args.tokenizer and file_sha256(args.tokenizer / "tokenizer.json") != base[
            "tokenizer_sha256"
        ]:
            raise ValueError("The supplied tokenizer does not match the base bundle")
        tokenizer_file = args.init_from / "tokenizer" / "tokenizer.json"
        lineage_id = base["lineage_id"]
    else:
        config_path = args.config or args.data / "config.yaml"
        tokenizer_path = args.tokenizer or args.data / "tokenizer"
        config = AceConfig.from_yaml(config_path)
        tokenizer = AceTokenizer.load(tokenizer_path)
        if tokenizer.vocab_size != config.vocab_size:
            raise ValueError("Tokenizer and model vocabularies must match")
        model = AceModel(config)
        tokenizer_file = tokenizer_path / "tokenizer.json"
        lineage_id = uuid.uuid4().hex

    train_path = args.data / "train.jsonl"
    validation_path = args.data / "validation.jsonl"
    data_manifest = read_sft_manifest(args.data)
    data_manifest_sha256 = file_sha256(args.data / "manifest.json")
    tokenizer_hash = file_sha256(tokenizer_file)
    seq_len = args.seq_len or config.max_seq_len
    if not 2 <= seq_len <= config.max_seq_len:
        raise ValueError("seq-len must be between 2 and the configured context length")
    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    identity = {
        "config": asdict(config),
        "backend": backend_name(config),
        "tokenizer_sha256": tokenizer_hash,
        "train_sha256": file_sha256(train_path),
        "validation_sha256": file_sha256(validation_path),
        "data_manifest_sha256": data_manifest_sha256,
        "seq_len": seq_len,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "lr": args.lr,
    }
    if args.resume:
        run = json.loads((args.output / "run.json").read_text(encoding="utf-8"))
        if run["identity"] != identity:
            raise ValueError("Resume requires the same model, data, tokenizer, and schedule")
        lineage_id = run["lineage_id"]
    else:
        if args.output.exists():
            raise FileExistsError("Choose a new output directory, or use --resume")
        run = {"identity": identity, "lineage_id": lineage_id}

    train_data = SupervisedDataset(train_path, tokenizer, seq_len)
    validation_data = SupervisedDataset(validation_path, tokenizer, seq_len)
    collator = CausalCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=0,
        sampler=ResumableRandomSampler(len(train_data), args.seed),
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        collate_fn=collator,
    )
    model.to(device)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if not args.resume:
        (args.output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")

    trainer = AceTrainer(
        model,
        config,
        TrainConfig(
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            lr=args.lr,
            max_seconds=args.max_seconds,
            checkpoint_dir=str(args.output / "checkpoints"),
            checkpoint_every=args.checkpoint_every,
            eval_every=args.eval_every,
            seed=args.seed,
            bf16=True,
            resume_from_checkpoint=str(args.output / "checkpoints") if args.resume else None,
        ),
    )
    metrics = trainer.train(train_loader, validation_loader)
    step = int(metrics["final_step"])
    bundle_path = args.output / f"bundle-{step:08d}"
    if not bundle_path.exists():
        save_bundle(
            model,
            tokenizer,
            bundle_path,
            lineage_id=lineage_id,
            provenance={
                "operation": "instruction-and-fim-sft",
                "steps": step,
                "train_sha256": identity["train_sha256"],
                "validation_sha256": identity["validation_sha256"],
                "data_manifest_sha256": data_manifest_sha256,
                "source_code_manifest_sha256": data_manifest["source_code_manifest"]["sha256"],
                "training_settings": {
                    "batch_size": args.batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "max_steps": args.max_steps,
                    "max_seconds": args.max_seconds,
                    "warmup_steps": args.warmup_steps,
                    "learning_rate": args.lr,
                    "seq_len": seq_len,
                    "seed": args.seed,
                    "device": device,
                    "threads": args.threads,
                },
                "quality_verified": False,
            },
        )
    report = {**metrics, "bundle": str(bundle_path)}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
