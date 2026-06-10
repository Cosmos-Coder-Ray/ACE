"""ACE Trainer — full pretraining loop with mixed-precision, gradient
accumulation, checkpointing, and optional W&B logging.

Provides:
    - :class:`TrainConfig` — All training hyper-parameters as a dataclass.
    - :class:`AceTrainer` — Orchestrates the training loop.

Usage::

    model = AceModel(model_config)
    trainer = AceTrainer(model, model_config, train_config)
    trainer.train(train_loader)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import torch
import torch.nn as nn
from torch.amp import autocast  # type: ignore[attr-defined]
from torch.utils.data import DataLoader

from ace.training.checkpoint import CheckpointManager
from ace.training.loss import compute_total_loss
from ace.training.optimizer import create_optimizer, create_scheduler, get_grad_norm

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# TrainConfig
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TrainConfig:
    """Training hyper-parameters for ACE pretraining.

    All tuneable knobs for the training loop live here so that they can
    be serialised, logged, and reproduced.

    Attributes:
        batch_size: Micro-batch size per forward pass.
        gradient_accumulation_steps: Number of micro-batches to
            accumulate before an optimiser step.
        max_steps: Total number of optimiser steps.
        lr: Peak learning rate.
        weight_decay: AdamW weight-decay coefficient.
        warmup_steps: Linear LR warmup steps.
        min_lr_ratio: Floor LR as a fraction of *lr* (cosine schedule).
        betas: AdamW beta coefficients.
        checkpoint_dir: Directory for checkpoint files.
        checkpoint_every: Save a checkpoint every N optimiser steps.
        eval_every: Run evaluation every N optimiser steps.
        log_every: Log metrics every N optimiser steps.
        use_wandb: Enable Weights & Biases logging.
        wandb_project: W&B project name.
        wandb_run_name: W&B run name (``None`` = auto-generated).
        bf16: Use bfloat16 mixed-precision training.
        gradient_clip: Maximum gradient norm for clipping.
        resume_from_checkpoint: Path to checkpoint dir to resume from
            (``None`` = train from scratch).
        aux_loss_coeff: Coefficient for MoE auxiliary loss.
        seed: Random seed for reproducibility.
    """

    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = 100_000
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    min_lr_ratio: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 1000
    eval_every: int = 500
    log_every: int = 10
    use_wandb: bool = False
    wandb_project: str = "ace-pretrain"
    wandb_run_name: Optional[str] = None
    bf16: bool = True
    gradient_clip: float = 1.0
    resume_from_checkpoint: Optional[str] = None
    aux_loss_coeff: float = 0.01
    seed: int = 42


# ═══════════════════════════════════════════════════════════════════════════
# AceTrainer
# ═══════════════════════════════════════════════════════════════════════════


class AceTrainer:
    """Orchestrates ACE pretraining.

    Handles:
        - Mixed-precision (bfloat16) autocast
        - Gradient accumulation across micro-batches
        - Gradient clipping
        - Cosine LR schedule with linear warmup
        - Periodic checkpoint saving
        - Optional W&B metric logging
        - Graceful ``KeyboardInterrupt`` handling with emergency checkpoint
        - Checkpoint resume

    Attributes:
        model: The :class:`AceModel` being trained.
        model_config: The model's :class:`AceConfig`.
        train_config: Training hyper-parameters.
        device: The device the model lives on.
        optimizer: The AdamW optimiser.
        scheduler: The LR scheduler.
        checkpoint_mgr: The :class:`CheckpointManager`.
        global_step: Current optimiser step (updated during training).
    """

    def __init__(
        self,
        model: nn.Module,
        model_config: Any,
        train_config: TrainConfig,
    ) -> None:
        """Initialise the trainer.

        Args:
            model: The model to train (should already be on the target
                device).
            model_config: The model configuration dataclass (persisted
                in checkpoints).
            train_config: Training hyper-parameters.
        """
        self.model: nn.Module = model
        self.model_config: Any = model_config
        self.train_config: TrainConfig = train_config
        self.device: torch.device = next(model.parameters()).device
        self.global_step: int = 0

        # ── Optimizer & scheduler ──────────────────────────────────────
        self.optimizer: torch.optim.AdamW = create_optimizer(
            model,
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
            betas=train_config.betas,
        )

        self.scheduler = create_scheduler(
            self.optimizer,
            warmup_steps=train_config.warmup_steps,
            max_steps=train_config.max_steps,
            min_lr_ratio=train_config.min_lr_ratio,
        )

        # ── Checkpoint manager ─────────────────────────────────────────
        self.checkpoint_mgr = CheckpointManager(train_config.checkpoint_dir)

        # ── Mixed precision ────────────────────────────────────────────
        self._amp_enabled: bool = train_config.bf16 and self.device.type == "cuda"
        self._amp_dtype: torch.dtype = torch.bfloat16

        # ── W&B ───────────────────────────────────────────────────────
        self._wandb: Any = None
        if train_config.use_wandb:
            try:
                import wandb

                self._wandb = wandb
            except ImportError as exc:
                raise ImportError(
                    "use_wandb=True but wandb is not installed. " "Install with: pip install wandb"
                ) from exc

        # ── Resume from checkpoint ────────────────────────────────────
        if train_config.resume_from_checkpoint is not None:
            self._resume_from_checkpoint(train_config.resume_from_checkpoint)

        logger.info(
            "AceTrainer initialised — device=%s, amp=%s, grad_accum=%d",
            self.device,
            self._amp_enabled,
            train_config.gradient_accumulation_steps,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader[Any],
        eval_loader: Optional[DataLoader[Any]] = None,
    ) -> dict[str, float]:
        """Run the training loop.

        Args:
            train_loader: DataLoader yielding dicts with an
                ``"input_ids"`` key (tensor of shape ``(B, L)``).
            eval_loader: Optional evaluation DataLoader (same format).

        Returns:
            A dict with final metrics (``"final_loss"``,
            ``"final_step"``).
        """
        cfg = self.train_config
        model = self.model
        model.train()

        # ── Initialise W&B run ────────────────────────────────────────
        if self._wandb is not None:
            self._wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config={
                    "train_config": cfg.__dict__,
                },
                resume="allow",
            )

        # ── Infinite data iterator ────────────────────────────────────
        data_iter: Iterator[dict[str, torch.Tensor]] = _InfiniteDataIterator(train_loader)

        # ── Progress bar (optional tqdm) ──────────────────────────────
        pbar = None
        try:
            from tqdm import tqdm

            pbar = tqdm(
                total=cfg.max_steps,
                initial=self.global_step,
                desc="Training",
                unit="step",
            )
        except ImportError:
            logger.info("tqdm not installed — progress bar disabled")

        last_log_loss: float = 0.0

        try:
            while self.global_step < cfg.max_steps:
                step_start = time.perf_counter()

                # ── Gradient accumulation loop ────────────────────────
                accumulated_metrics: dict[str, float] = {
                    "lm_loss": 0.0,
                    "aux_loss": 0.0,
                    "total_loss": 0.0,
                }

                for _micro in range(cfg.gradient_accumulation_steps):
                    batch = next(data_iter)
                    micro_metrics = self._train_step(batch)

                    for k in accumulated_metrics:
                        accumulated_metrics[k] += micro_metrics[k]

                # Average accumulated metrics
                for k in accumulated_metrics:
                    accumulated_metrics[k] /= cfg.gradient_accumulation_steps

                # ── Gradient clipping ─────────────────────────────────
                grad_norm = get_grad_norm(model)
                if cfg.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)

                # ── Optimiser step ────────────────────────────────────
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                self.global_step += 1
                step_time = time.perf_counter() - step_start

                # ── Compute tokens/sec ────────────────────────────────
                batch_input_ids = batch["input_ids"]  # (B, L)
                seq_len = batch_input_ids.shape[1]
                tokens_per_step = cfg.batch_size * cfg.gradient_accumulation_steps * seq_len
                tok_per_sec = tokens_per_step / max(step_time, 1e-9)

                # ── Get current LR ────────────────────────────────────
                current_lr = self.optimizer.param_groups[0]["lr"]

                last_log_loss = accumulated_metrics["total_loss"]

                # ── Logging ───────────────────────────────────────────
                if self.global_step % cfg.log_every == 0:
                    metrics = {
                        "step": self.global_step,
                        "loss": accumulated_metrics["total_loss"],
                        "lm_loss": accumulated_metrics["lm_loss"],
                        "aux_loss": accumulated_metrics["aux_loss"],
                        "lr": current_lr,
                        "grad_norm": grad_norm,
                        "tok/sec": tok_per_sec,
                    }
                    self._log_metrics(metrics, self.global_step)

                # ── Checkpoint saving ─────────────────────────────────
                if self.global_step % cfg.checkpoint_every == 0:
                    self._save_checkpoint(
                        self.global_step,
                        accumulated_metrics["total_loss"],
                    )

                # ── Evaluation ────────────────────────────────────────
                if eval_loader is not None and self.global_step % cfg.eval_every == 0:
                    eval_metrics = self._evaluate(eval_loader)
                    self._log_metrics(
                        {"eval_" + k: v for k, v in eval_metrics.items()},
                        self.global_step,
                    )
                    model.train()

                # ── Update progress bar ───────────────────────────────
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(
                        loss=f"{accumulated_metrics['total_loss']:.4f}",
                        lr=f"{current_lr:.2e}",
                    )

        except KeyboardInterrupt:
            logger.warning(
                "KeyboardInterrupt at step %d — saving emergency checkpoint",
                self.global_step,
            )
            self._save_checkpoint(self.global_step, last_log_loss)
            if pbar is not None:
                pbar.close()
            raise

        # ── Final checkpoint ──────────────────────────────────────────
        self._save_checkpoint(self.global_step, last_log_loss)

        if pbar is not None:
            pbar.close()

        if self._wandb is not None:
            self._wandb.finish()

        logger.info(
            "Training complete — %d steps, final loss=%.6f",
            self.global_step,
            last_log_loss,
        )

        return {
            "final_loss": last_log_loss,
            "final_step": float(self.global_step),
        }

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Execute a single micro-batch forward + backward pass.

        The loss is scaled by ``1 / gradient_accumulation_steps`` before
        backward so that the accumulated gradient matches the true
        average across micro-batches.

        Args:
            batch: A dict with at least ``"input_ids"`` of shape
                ``(B, L)``.  An optional ``"labels"`` key is used if
                present; otherwise labels = input_ids.

        Returns:
            Dict with ``"lm_loss"``, ``"aux_loss"``, ``"total_loss"``
            (un-scaled values for logging).
        """
        input_ids = batch["input_ids"].to(self.device)  # (B, L)
        labels = batch.get("labels", input_ids).to(self.device)  # (B, L)

        with autocast(
            device_type=self.device.type,
            dtype=self._amp_dtype,
            enabled=self._amp_enabled,
        ):
            output = self.model(input_ids)  # AceModelOutput
            logits = output.logits  # (B, L, V)
            aux_loss = output.total_aux_loss  # scalar

            loss, metrics = compute_total_loss(
                logits,
                labels,
                aux_loss,
                aux_loss_coeff=self.train_config.aux_loss_coeff,
            )

            # Scale for gradient accumulation
            scaled_loss = loss / self.train_config.gradient_accumulation_steps  # scalar

        scaled_loss.backward()  # type: ignore[no-untyped-call]

        return metrics

    def _evaluate(self, eval_loader: DataLoader[Any]) -> dict[str, float]:
        """Run a full evaluation pass over the eval DataLoader.

        Args:
            eval_loader: DataLoader yielding dicts with ``"input_ids"``.

        Returns:
            Dict with averaged ``"loss"``, ``"lm_loss"``, ``"aux_loss"``.
        """
        self.model.eval()
        total_loss: float = 0.0
        total_lm: float = 0.0
        total_aux: float = 0.0
        n_batches: int = 0

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(self.device)  # (B, L)
                labels = batch.get("labels", input_ids).to(self.device)  # (B, L)

                with autocast(
                    device_type=self.device.type,
                    dtype=self._amp_dtype,
                    enabled=self._amp_enabled,
                ):
                    output = self.model(input_ids)
                    _, metrics = compute_total_loss(
                        output.logits,
                        labels,
                        output.total_aux_loss,
                        aux_loss_coeff=self.train_config.aux_loss_coeff,
                    )

                total_loss += metrics["total_loss"]
                total_lm += metrics["lm_loss"]
                total_aux += metrics["aux_loss"]
                n_batches += 1

        n = max(n_batches, 1)
        return {
            "loss": total_loss / n,
            "lm_loss": total_lm / n,
            "aux_loss": total_aux / n,
        }

    def _log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        """Log metrics to the console and optionally to W&B.

        Args:
            metrics: Key-value pairs to log.
            step: The current training step.
        """
        parts = [f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
        logger.info("Step %d — %s", step, " | ".join(parts))

        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def _save_checkpoint(self, step: int, loss: float) -> None:
        """Save a checkpoint and clean up old ones.

        Args:
            step: Current training step.
            loss: Current loss value for metadata.
        """
        self.checkpoint_mgr.save(
            self.model,
            self.optimizer,
            self.scheduler,
            step=step,
            loss=loss,
            config=self.model_config,
        )
        self.checkpoint_mgr.delete_old_checkpoints(keep_last_n=3)

    def _resume_from_checkpoint(self, checkpoint_dir: str) -> None:
        """Resume training state from the latest checkpoint in a directory.

        Loads model weights, optimizer state, scheduler state, and the
        global step counter.

        Args:
            checkpoint_dir: Path to the checkpoint directory.

        Raises:
            FileNotFoundError: If no checkpoints are found.
        """
        resume_mgr = CheckpointManager(checkpoint_dir)
        ckpt = resume_mgr.load_latest()

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt["step"]

        logger.info(
            "Resumed from checkpoint: step=%d, loss=%.6f",
            ckpt["step"],
            ckpt["loss"],
        )


# ═══════════════════════════════════════════════════════════════════════════
# Infinite data iterator
# ═══════════════════════════════════════════════════════════════════════════


class _InfiniteDataIterator:
    """Wraps a DataLoader into an infinite cycling iterator.

    This supports step-based (rather than epoch-based) training: the
    iterator automatically restarts the DataLoader when it is exhausted.

    Attributes:
        loader: The underlying DataLoader.
    """

    def __init__(self, loader: DataLoader[Any]) -> None:
        """Initialise the infinite iterator.

        Args:
            loader: The DataLoader to cycle over.
        """
        self.loader: DataLoader[Any] = loader
        self._iterator: Iterator[Any] = iter(loader)

    def __iter__(self) -> _InfiniteDataIterator:
        """Return self (iterator protocol)."""
        return self

    def __next__(self) -> Any:
        """Return the next batch, restarting the DataLoader if needed."""
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.loader)
            return next(self._iterator)


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import tempfile

    from torch.utils.data import TensorDataset

    from ace.model.backbone import AceModel

    # Import AceModel + AceConfig
    from ace.model.config import AceConfig

    print("=" * 60)
    print("trainer.py — smoke test")
    print("=" * 60)

    torch.manual_seed(42)

    # ── Micro config (no Mamba) ────────────────────────────────────────
    model_cfg = AceConfig(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        d_ff=256,
        n_layers=4,
        max_seq_len=128,
        vocab_size=256,
        num_experts=4,
        top_k_experts=2,
        mamba_layers=[],
        moe_layers=[2, 3],
        retnet_layers=[],
        dropout=0.0,
        tie_embeddings=True,
        use_gradient_checkpointing=False,
    )

    device = torch.device("cpu")
    model = AceModel(model_cfg).to(device).to(torch.float32)

    # ── Synthetic dataset ──────────────────────────────────────────────
    B, L = 2, 32
    n_samples = 20
    input_ids = torch.randint(0, model_cfg.vocab_size, (n_samples, L))  # (N, L)
    dataset = TensorDataset(input_ids)

    # Wrap TensorDataset to return dicts
    class _DictDataset(torch.utils.data.Dataset):  # type: ignore[type-arg]
        """Wraps a TensorDataset to yield dicts with 'input_ids'."""

        def __init__(self, tensor_ds: TensorDataset) -> None:
            self._ds = tensor_ds

        def __len__(self) -> int:
            return len(self._ds)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return {"input_ids": self._ds[idx][0]}

    dict_dataset = _DictDataset(dataset)
    train_loader: DataLoader[Any] = DataLoader(
        dict_dataset, batch_size=B, shuffle=True, drop_last=True
    )

    with tempfile.TemporaryDirectory() as tmp_ckpt:
        # ── TrainConfig ────────────────────────────────────────────────
        train_cfg = TrainConfig(
            batch_size=B,
            gradient_accumulation_steps=2,
            max_steps=5,
            lr=1e-3,
            weight_decay=0.01,
            warmup_steps=2,
            checkpoint_dir=tmp_ckpt,
            checkpoint_every=3,
            eval_every=5,
            log_every=1,
            use_wandb=False,
            bf16=False,  # CPU — no bf16
            gradient_clip=1.0,
            aux_loss_coeff=0.01,
            seed=42,
        )

        # ── Create trainer ─────────────────────────────────────────────
        trainer = AceTrainer(model, model_cfg, train_cfg)
        assert trainer.global_step == 0, "Initial step should be 0"
        print("[PASS] AceTrainer initialised")

        # ── Train ──────────────────────────────────────────────────────
        result = trainer.train(train_loader)
        assert "final_loss" in result, "Should return final_loss"
        assert "final_step" in result, "Should return final_step"
        assert result["final_step"] == 5.0, f"Should be at step 5, got {result['final_step']}"
        assert trainer.global_step == 5, f"global_step should be 5, got {trainer.global_step}"
        print(
            f"[PASS] Training completed: step={trainer.global_step}, loss={result['final_loss']:.4f}"
        )

        # ── Verify checkpoint was saved ────────────────────────────────
        ckpts = trainer.checkpoint_mgr.list_checkpoints()
        assert len(ckpts) > 0, "Should have saved at least one checkpoint"
        print(f"[PASS] Checkpoints saved: {len(ckpts)} files")

        # ── Test checkpoint resume ─────────────────────────────────────
        model2 = AceModel(model_cfg).to(device).to(torch.float32)
        train_cfg2 = TrainConfig(
            batch_size=B,
            gradient_accumulation_steps=2,
            max_steps=8,
            lr=1e-3,
            weight_decay=0.01,
            warmup_steps=2,
            checkpoint_dir=tmp_ckpt,
            checkpoint_every=10,
            log_every=1,
            use_wandb=False,
            bf16=False,
            gradient_clip=1.0,
            aux_loss_coeff=0.01,
            resume_from_checkpoint=tmp_ckpt,
        )

        trainer2 = AceTrainer(model2, model_cfg, train_cfg2)
        assert trainer2.global_step == 5, f"Resumed step should be 5, got {trainer2.global_step}"
        print("[PASS] Checkpoint resume: step correctly restored to 5")

        result2 = trainer2.train(train_loader)
        assert trainer2.global_step == 8, f"Should reach step 8, got {trainer2.global_step}"
        print(f"[PASS] Resumed training completed: step={trainer2.global_step}")

    # ── Infinite iterator test ─────────────────────────────────────────
    small_loader: DataLoader[Any] = DataLoader(dict_dataset, batch_size=B, drop_last=True)
    inf_iter = _InfiniteDataIterator(small_loader)
    batches_drawn = 0
    for _ in range(50):
        batch = next(inf_iter)
        assert "input_ids" in batch
        batches_drawn += 1
    assert batches_drawn == 50, f"Should draw 50 batches, got {batches_drawn}"
    print("[PASS] _InfiniteDataIterator cycles correctly")

    print("=" * 60)
    print("[ALL PASS] trainer.py smoke test complete")
    print("=" * 60)
