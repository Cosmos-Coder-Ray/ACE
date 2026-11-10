"""Train ACE on prepared local data with bounded sessions and portable exports."""

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
from ace.data.dataset import PackedTokenDataset
from ace.data.preprocessing import file_sha256
from ace.data.samplers import ResumableRandomSampler
from ace.inference.bundle import backend_name, load_bundle, save_bundle
from ace.model.backbone import AceModel
from ace.model.config import AceConfig
from ace.tokenizer.ace_tokenizer import AceTokenizer
from ace.training.trainer import AceTrainer, TrainConfig
from ace.utils.env_detect import HAS_MAMBA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Defaults to prepared data/config.yaml")
    parser.add_argument("--init-from", type=Path, help="Continue training a compatible base bundle")
    parser.add_argument("--resume", action="store_true", help="Resume this output directory")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, help="Defaults to the model context limit")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-parameters", type=int, default=50_000_000)
    parser.add_argument("--allow-fallback-mamba", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.resume and args.init_from:
        raise ValueError("Use either --resume or --init-from")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    config = AceConfig.from_yaml(args.config or args.data / "config.yaml")
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    tokenizer = AceTokenizer.load(args.data / "tokenizer")
    if tokenizer.vocab_size != config.vocab_size or manifest["vocab_size"] != config.vocab_size:
        raise ValueError("Tokenizer, data and model vocabularies must match")
    tokenizer_hash = file_sha256(args.data / "tokenizer" / "tokenizer.json")
    if tokenizer_hash != manifest["tokenizer_sha256"]:
        raise ValueError("Prepared tokenizer fingerprint mismatch")
    for split in ("train", "validation"):
        if file_sha256(args.data / f"{split}.bin") != manifest["streams"][split]["sha256"]:
            raise ValueError(f"Prepared {split} stream fingerprint mismatch")
    if config.mamba_layers and not HAS_MAMBA and not args.allow_fallback_mamba:
        raise ValueError(
            "Native Mamba is unavailable. Use a dense config or explicitly allow the GRU fallback"
        )
    # Count without allocating parameter storage, before a mistaken large run.
    with torch.device("meta"):
        parameter_count = AceModel(config).num_parameters
    if parameter_count > args.max_parameters:
        raise ValueError(f"{parameter_count:,} parameters exceeds budget {args.max_parameters:,}")
    seq_len = args.seq_len or config.max_seq_len
    if not 2 <= seq_len <= config.max_seq_len:
        raise ValueError("seq-len must be between 2 and the configured context length")
    device = (
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        lr=args.lr,
        max_seconds=args.max_seconds,
        checkpoint_dir=str(args.output / "checkpoints"),
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
        log_every=10,
        seed=args.seed,
        use_wandb=args.use_wandb,
        resume_from_checkpoint=str(args.output / "checkpoints") if args.resume else None,
    )
    identity = {
        "config": asdict(config),
        "backend": backend_name(config),
        "tokenizer_sha256": tokenizer_hash,
        "data_sha256": file_sha256(args.data / "manifest.json"),
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
            raise ValueError(
                "Resume requires the same model, data, tokenizer and training schedule"
            )
    else:
        if args.output.exists():
            raise FileExistsError("Choose a new output directory, or use --resume")
        run = {
            "identity": identity,
            "lineage_id": uuid.uuid4().hex,
            "parameters": parameter_count,
            "source_license": manifest["source_license"],
        }

    if args.init_from:
        model, _, base = load_bundle(args.init_from)
        if base["config"] != asdict(config) or base["tokenizer_sha256"] != tokenizer_hash:
            raise ValueError("Base bundle must match configuration and tokenizer exactly")
        run["lineage_id"] = base["lineage_id"]
        run["parent"] = file_sha256(args.init_from / "manifest.json")
    else:
        model = AceModel(config)
    model.to(device)  # Keep master parameters in float32; trainer chooses supported autocast.
    if not args.resume:
        args.output.mkdir(parents=True)
        (args.output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    train_data = PackedTokenDataset(args.data / "train.bin", seq_len, config.vocab_size)
    eval_data = PackedTokenDataset(args.data / "validation.bin", seq_len, config.vocab_size)
    collator = CausalCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=0,
        sampler=ResumableRandomSampler(len(train_data), args.seed),
        generator=torch.Generator().manual_seed(args.seed),
    )
    eval_loader = DataLoader(
        eval_data,
        batch_size=args.batch_size,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(args.seed),
    )
    trainer = AceTrainer(model, config, train_config)
    metrics = trainer.train(train_loader, eval_loader)
    step = int(metrics["final_step"])
    bundle_path = args.output / f"bundle-{step:08d}"
    if not bundle_path.exists():
        save_bundle(
            model,
            tokenizer,
            bundle_path,
            lineage_id=run["lineage_id"],
            provenance={
                "operation": "training",
                "steps": step,
                "data_sha256": identity["data_sha256"],
                "source_license": manifest["source_license"],
                "quality_verified": False,
            },
        )
    report = {**metrics, "bundle": str(bundle_path)}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
