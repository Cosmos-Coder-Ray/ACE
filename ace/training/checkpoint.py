"""Checkpoint management for ACE training.

Provides :class:`CheckpointManager` for saving, loading, listing, and
cleaning up model checkpoints.  Checkpoints are self-contained — they
include model, optimizer, and scheduler state dicts, the training step,
the current loss, the full :class:`AceConfig` (serialised via
``dataclasses.asdict``), and a UTC timestamp.

File naming convention::

    step_00000042.pt   ← zero-padded for natural sort order

All file I/O uses :mod:`pathlib` — no ``os.path``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Regex to extract the step number from a checkpoint filename.
_STEP_RE = re.compile(r"^step_(\d+)\.pt$")


class CheckpointManager:
    """Manages the lifecycle of training checkpoints on disk.

    Responsibilities:
        - **Save** model, optimizer, scheduler, and metadata to disk.
        - **Load** the latest or a specific checkpoint.
        - **List** available checkpoints (sorted by step).
        - **Delete** old checkpoints to reclaim storage.

    Attributes:
        checkpoint_dir: Directory where checkpoint files are stored.
    """

    def __init__(self, checkpoint_dir: str | Path) -> None:
        """Initialise the checkpoint manager.

        Creates the checkpoint directory if it does not already exist.

        Args:
            checkpoint_dir: Path to the directory for checkpoint storage.
        """
        self.checkpoint_dir: Path = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info("CheckpointManager initialised at %s", self.checkpoint_dir)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        loss: float,
        config: Any,
    ) -> Path:
        """Save a training checkpoint to disk.

        The checkpoint file is named ``step_{step:08d}.pt`` and contains
        all state needed to resume training.

        Args:
            model: The model to checkpoint.
            optimizer: The optimizer to checkpoint.
            scheduler: The LR scheduler to checkpoint (must have
                ``state_dict()``).
            step: Current global training step.
            loss: Current loss value (for metadata / logging).
            config: The model config dataclass.  Serialised via
                ``dataclasses.asdict``.

        Returns:
            Path to the saved checkpoint file.

        Raises:
            ValueError: If ``step`` is negative.
        """
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")

        filename = f"step_{step:08d}.pt"
        filepath = self.checkpoint_dir / filename

        checkpoint: dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "loss": loss,
            "config": asdict(config),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        torch.save(checkpoint, filepath)
        logger.info("Checkpoint saved: %s (step=%d, loss=%.6f)", filepath.name, step, loss)
        return filepath

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_latest(self) -> dict[str, Any]:
        """Load the most recent checkpoint (highest step).

        Returns:
            The checkpoint dictionary.

        Raises:
            FileNotFoundError: If no checkpoints exist in the directory.
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")

        latest = checkpoints[-1]  # highest step (list is sorted ascending)
        logger.info(
            "Loading latest checkpoint: step=%d from %s",
            latest["step"],
            latest["path"],
        )
        checkpoint: dict[str, Any] = torch.load(
            latest["path"], map_location="cpu", weights_only=False
        )
        return checkpoint

    def load_specific(self, step: int) -> dict[str, Any]:
        """Load a checkpoint for a specific training step.

        Args:
            step: The training step whose checkpoint to load.

        Returns:
            The checkpoint dictionary.

        Raises:
            ValueError: If ``step`` is negative.
            FileNotFoundError: If no checkpoint exists for the given step.
        """
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")

        filename = f"step_{step:08d}.pt"
        filepath = self.checkpoint_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        logger.info("Loading checkpoint: step=%d from %s", step, filepath)
        checkpoint: dict[str, Any] = torch.load(filepath, map_location="cpu", weights_only=False)
        return checkpoint

    # ------------------------------------------------------------------
    # List & cleanup
    # ------------------------------------------------------------------

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all available checkpoints, sorted by step (ascending).

        Returns:
            A list of dicts, each with keys ``"step"`` (int),
            ``"path"`` (:class:`Path`), and ``"timestamp"`` (str,
            ISO-format file modification time).
        """
        entries: list[dict[str, Any]] = []

        for path in sorted(self.checkpoint_dir.glob("step_*.pt")):
            match = _STEP_RE.match(path.name)
            if match is None:
                continue  # skip files that don't match naming convention

            step = int(match.group(1))
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()

            entries.append({"step": step, "path": path, "timestamp": mtime})

        # Sort by step ascending (glob is already sorted by name, but be safe)
        entries.sort(key=lambda e: e["step"])
        return entries

    def delete_old_checkpoints(self, keep_last_n: int = 3) -> list[Path]:
        """Delete old checkpoints, keeping only the *N* most recent.

        Args:
            keep_last_n: Number of most-recent checkpoints to retain.

        Returns:
            List of :class:`Path` objects that were deleted.

        Raises:
            ValueError: If ``keep_last_n < 1``.
        """
        if keep_last_n < 1:
            raise ValueError(f"keep_last_n must be >= 1, got {keep_last_n}")

        checkpoints = self.list_checkpoints()

        if len(checkpoints) <= keep_last_n:
            return []

        to_delete = checkpoints[:-keep_last_n]
        deleted: list[Path] = []

        for entry in to_delete:
            path: Path = entry["path"]
            path.unlink()
            logger.info("Deleted old checkpoint: %s (step=%d)", path.name, entry["step"])
            deleted.append(path)

        return deleted


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from dataclasses import dataclass

    from torch.optim import SGD
    from torch.optim.lr_scheduler import LambdaLR

    print("=" * 60)
    print("checkpoint.py — smoke test")
    print("=" * 60)

    # ── Mock config dataclass ──────────────────────────────────────────
    @dataclass
    class _MockConfig:
        """Tiny config for testing."""

        d_model: int = 32
        n_layers: int = 2
        vocab_size: int = 64

    mock_cfg = _MockConfig()

    with tempfile.TemporaryDirectory() as tmp_dir:
        mgr = CheckpointManager(tmp_dir)

        # ── Tiny model + optimizer + scheduler ─────────────────────────
        model = nn.Linear(32, 16)
        opt = SGD(model.parameters(), lr=0.01)
        sched = LambdaLR(opt, lr_lambda=lambda s: 1.0)

        # ── Test 1: save and verify file exists ────────────────────────
        path = mgr.save(model, opt, sched, step=0, loss=5.0, config=mock_cfg)
        assert path.exists(), f"Checkpoint file should exist: {path}"
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        for key in (
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "step",
            "loss",
            "config",
            "timestamp",
        ):
            assert key in ckpt, f"Missing key in checkpoint: {key}"
        assert ckpt["step"] == 0
        assert ckpt["loss"] == 5.0
        print("[PASS] save: file exists and contains all expected keys")

        # ── Test 2: save 5 checkpoints, list returns 5 sorted ─────────
        for s in range(1, 5):
            mgr.save(model, opt, sched, step=s, loss=5.0 - s, config=mock_cfg)

        ckpts = mgr.list_checkpoints()
        assert len(ckpts) == 5, f"Expected 5 checkpoints, got {len(ckpts)}"
        steps = [c["step"] for c in ckpts]
        assert steps == [0, 1, 2, 3, 4], f"Steps should be sorted: {steps}"
        print("[PASS] list_checkpoints: 5 checkpoints, sorted ascending")

        # ── Test 3: load_latest returns highest step ───────────────────
        latest = mgr.load_latest()
        assert latest["step"] == 4, f"Latest step should be 4, got {latest['step']}"
        print("[PASS] load_latest: returns step 4")

        # ── Test 4: load_specific returns correct step ─────────────────
        specific = mgr.load_specific(step=2)
        assert specific["step"] == 2, f"Expected step 2, got {specific['step']}"
        print("[PASS] load_specific: returns step 2")

        # ── Test 5: delete_old_checkpoints keeps last 2 ───────────────
        deleted = mgr.delete_old_checkpoints(keep_last_n=2)
        assert len(deleted) == 3, f"Expected 3 deletions, got {len(deleted)}"
        remaining = mgr.list_checkpoints()
        assert len(remaining) == 2, f"Expected 2 remaining, got {len(remaining)}"
        remaining_steps = [c["step"] for c in remaining]
        assert remaining_steps == [3, 4], f"Should keep steps 3,4: {remaining_steps}"
        print("[PASS] delete_old_checkpoints(keep_last_n=2): 3 deleted, 2 remain")

    # ── Test 6: error cases ────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp_dir2:
        mgr2 = CheckpointManager(tmp_dir2)

        # load_latest with no checkpoints
        try:
            mgr2.load_latest()
            raise AssertionError("Should have raised FileNotFoundError")
        except FileNotFoundError:
            pass
        print("[PASS] FileNotFoundError for load_latest with no checkpoints")

        # load_specific with nonexistent step
        try:
            mgr2.load_specific(step=999)
            raise AssertionError("Should have raised FileNotFoundError")
        except FileNotFoundError:
            pass
        print("[PASS] FileNotFoundError for load_specific with bad step")

        # negative step in save
        try:
            mgr2.save(model, opt, sched, step=-1, loss=0.0, config=mock_cfg)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass
        print("[PASS] ValueError for negative step in save")

        # negative step in load_specific
        try:
            mgr2.load_specific(step=-1)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass
        print("[PASS] ValueError for negative step in load_specific")

        # keep_last_n < 1
        try:
            mgr2.delete_old_checkpoints(keep_last_n=0)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass
        print("[PASS] ValueError for keep_last_n < 1")

    print("=" * 60)
    print("[ALL PASS] checkpoint.py smoke test complete")
    print("=" * 60)
