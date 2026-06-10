#!/usr/bin/env python3
"""ACE pretraining entry point.

Usage::

    python scripts/pretrain.py --config configs/ace_micro.yaml
    python scripts/pretrain.py --config configs/ace_micro.yaml --max_steps 5000 --lr 1e-4
    python scripts/pretrain.py --config configs/ace_micro.yaml --resume_from_checkpoint checkpoints/run_001

Until the data pipeline (``ace/data/``) is implemented, this script
uses a **synthetic random-token dataset** so the training loop can be
validated end-to-end.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

# ── Ensure project root is on sys.path ─────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ace.model.backbone import AceModel  # noqa: E402
from ace.model.config import AceConfig  # noqa: E402
from ace.training.trainer import AceTrainer, TrainConfig  # noqa: E402

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic dataset (placeholder until ace/data/ is built)
# ═══════════════════════════════════════════════════════════════════════════


class _SyntheticTokenDataset(Dataset[dict[str, torch.Tensor]]):
    """Random-token dataset for end-to-end training-loop validation.

    Each sample is a dict ``{"input_ids": Tensor(L,)}`` with random
    token ids drawn uniformly from ``[0, vocab_size)``.

    Attributes:
        n_samples: Number of synthetic samples.
        seq_len: Sequence length per sample.
        vocab_size: Vocabulary size for random token generation.
    """

    def __init__(self, n_samples: int, seq_len: int, vocab_size: int) -> None:
        """Initialise the synthetic dataset.

        Args:
            n_samples: Number of samples in the dataset.
            seq_len: Sequence length for each sample.
            vocab_size: Upper bound for random token ids.
        """
        self.n_samples: int = n_samples
        self.seq_len: int = seq_len
        self.vocab_size: int = vocab_size

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a random-token sample.

        Args:
            idx: Sample index (unused — every call generates fresh
                random tokens).

        Returns:
            Dict with ``"input_ids"`` of shape ``(L,)``.
        """
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))  # (L,)
        return {"input_ids": input_ids}


# ═══════════════════════════════════════════════════════════════════════════
# CLI argument parsing
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="ACE pretraining script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model config
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to AceConfig YAML file (e.g. configs/ace_micro.yaml)",
    )

    # Training hyper-parameters (override TrainConfig defaults)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--aux_loss_coeff", type=float, default=0.01)

    # Checkpointing
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=10)

    # W&B
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="ace-pretrain")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    # Mixed precision
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 mixed precision (default: True)",
    )
    parser.add_argument(
        "--no_bf16",
        action="store_true",
        help="Disable bfloat16 mixed precision",
    )

    # Resume
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from",
    )

    # Data (synthetic placeholder)
    parser.add_argument(
        "--n_synthetic_samples",
        type=int,
        default=10_000,
        help="Number of synthetic samples (placeholder until data pipeline)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=512,
        help="Sequence length for synthetic data",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Run ACE pretraining."""
    args = parse_args()

    # ── Logging ────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Seed ──────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Load model config ──────────────────────────────────────────────
    config_path = Path(args.config)
    logger.info("Loading model config from %s", config_path)
    model_config = AceConfig.from_yaml(config_path)
    logger.info("\n%s", model_config.summary())

    # ── Determine device ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ── Build model ───────────────────────────────────────────────────
    logger.info("Building AceModel...")
    model = AceModel.from_config(model_config)

    # Set dtype
    model_dtype = torch.float32
    if args.bf16 and not args.no_bf16 and device.type == "cuda":
        model_dtype = torch.bfloat16

    model = model.to(device=device, dtype=model_dtype)
    logger.info(
        "Model on %s with dtype=%s, params=%s", device, model_dtype, f"{model.num_parameters:,}"
    )

    # ── Build train config ────────────────────────────────────────────
    use_bf16 = args.bf16 and not args.no_bf16
    train_config = TrainConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
        log_every=args.log_every,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        bf16=use_bf16,
        gradient_clip=args.gradient_clip,
        resume_from_checkpoint=args.resume_from_checkpoint,
        aux_loss_coeff=args.aux_loss_coeff,
        seed=args.seed,
    )

    # ── Build dataset ─────────────────────────────────────────────────
    logger.info(
        "Using synthetic dataset: %d samples, seq_len=%d",
        args.n_synthetic_samples,
        args.seq_len,
    )
    dataset = _SyntheticTokenDataset(
        n_samples=args.n_synthetic_samples,
        seq_len=args.seq_len,
        vocab_size=model_config.vocab_size,
    )
    train_loader: DataLoader[Any] = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    # ── Build trainer and run ──────────────────────────────────────────
    trainer = AceTrainer(model, model_config, train_config)

    logger.info("Starting pretraining...")
    result = trainer.train(train_loader)

    logger.info("=" * 60)
    logger.info("Pretraining complete!")
    logger.info("  Final step: %d", int(result["final_step"]))
    logger.info("  Final loss: %.6f", result["final_loss"])
    logger.info("  Checkpoints: %s", train_config.checkpoint_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
