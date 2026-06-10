"""Optimizer utilities for ACE pretraining.

Provides:
    - :func:`create_optimizer` — AdamW with separate decay / no-decay param
      groups (bias, norm weights, and embeddings are excluded from decay).
    - :func:`create_scheduler` — Cosine-decay learning-rate schedule with
      linear warmup.
    - :func:`get_grad_norm` — Global L2 gradient norm for monitoring.

All hyper-parameters are passed explicitly — no magic numbers.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)

# Substrings that identify parameters which should NOT receive weight decay.
# Matches: bias terms, RMSNorm/LayerNorm weights, and embedding tables.
_NO_DECAY_KEYWORDS: tuple[str, ...] = ("bias", "norm", "embed")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> AdamW:
    """Create an AdamW optimizer with decay / no-decay parameter groups.

    Parameters that match any of the no-decay keywords (``bias``, ``norm``,
    ``embed``) in their name are placed in the *no-decay* group (weight_decay
    = 0).  All other trainable parameters go in the *decay* group.

    Args:
        model: The model whose parameters will be optimised.
        lr: Peak learning rate.
        weight_decay: Weight-decay coefficient for the decay group.
        betas: Adam beta coefficients.
        eps: Adam epsilon for numerical stability.

    Returns:
        A configured :class:`torch.optim.AdamW` instance.

    Raises:
        ValueError: If ``lr <= 0`` or ``weight_decay < 0``.
    """
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")

    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    decay_names: list[str] = []
    no_decay_names: list[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(kw in name for kw in _NO_DECAY_KEYWORDS):
            no_decay_params.append(param)
            no_decay_names.append(name)
        else:
            decay_params.append(param)
            decay_names.append(name)

    # Sanity checks: every param must be in exactly one group
    decay_set = set(id(p) for p in decay_params)
    no_decay_set = set(id(p) for p in no_decay_params)
    assert decay_set.isdisjoint(
        no_decay_set
    ), "A parameter ended up in both decay and no-decay groups"

    total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert len(decay_params) + len(no_decay_params) == total_trainable, (
        f"Parameter accounting mismatch: decay({len(decay_params)}) + "
        f"no_decay({len(no_decay_params)}) != total({total_trainable})"
    )

    logger.info(
        "Optimizer param groups — decay: %d params, no_decay: %d params",
        len(decay_params),
        len(no_decay_params),
    )

    param_groups: list[dict[str, object]] = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(param_groups, lr=lr, betas=betas, eps=eps)
    return optimizer


def create_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create a cosine-decay LR scheduler with linear warmup.

    Schedule:
        - **Warmup** (step < warmup_steps): LR ramps linearly from 0 to
          the peak value.
        - **Cosine decay** (step >= warmup_steps): LR decays from peak
          to ``peak_lr * min_lr_ratio`` following a cosine curve.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of linear-warmup steps.
        max_steps: Total training steps (warmup + decay).
        min_lr_ratio: Floor LR as a fraction of peak LR (0–1).

    Returns:
        A :class:`LambdaLR` scheduler.

    Raises:
        ValueError: If arguments are out of valid ranges.
    """
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if max_steps <= 0:
        raise ValueError(f"max_steps must be > 0, got {max_steps}")
    if warmup_steps >= max_steps:
        raise ValueError(f"warmup_steps ({warmup_steps}) must be < max_steps ({max_steps})")
    if min_lr_ratio < 0 or min_lr_ratio > 1:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")

    def lr_lambda(step: int) -> float:
        """Compute LR multiplier for a given step."""
        if step < warmup_steps:
            # Linear warmup: 0 → 1
            return step / max(1, warmup_steps)

        # Cosine decay: 1 → min_lr_ratio
        decay_steps = max(1, max_steps - warmup_steps)
        progress = (step - warmup_steps) / decay_steps
        # Clamp progress to [0, 1] for safety beyond max_steps
        progress = min(progress, 1.0)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


def get_grad_norm(model: nn.Module) -> float:
    """Compute the global L2 gradient norm across all model parameters.

    Only parameters with a non-``None`` ``.grad`` attribute are included.
    Returns ``0.0`` if no parameters have gradients.

    Args:
        model: The model whose gradients to inspect.

    Returns:
        The global L2 norm as a Python float.
    """
    total_norm_sq: float = 0.0

    for param in model.parameters():
        if param.grad is not None:
            param_norm = torch.norm(param.grad.detach(), 2.0)  # scalar
            total_norm_sq += param_norm.item() ** 2

    return math.sqrt(total_norm_sq)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("optimizer.py — smoke test")
    print("=" * 60)

    # ── Tiny model with diverse param types ────────────────────────────
    class _TinyModel(nn.Module):
        """Minimal model with Linear, LayerNorm, and Embedding."""

        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(64, 32)  # name contains 'embed'
            self.linear = nn.Linear(32, 16)  # has .weight + .bias
            self.norm = nn.LayerNorm(16)  # name contains 'norm'
            self.head = nn.Linear(16, 4, bias=False)  # weight only

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.embed(x)  # (B, L, 32)
            h = self.linear(h)  # (B, L, 16)
            h = self.norm(h)  # (B, L, 16)
            out: torch.Tensor = self.head(h)  # (B, L, 4)
            return out

    model = _TinyModel()

    # ── Test 1: create_optimizer param groups ──────────────────────────
    opt = create_optimizer(model, lr=1e-3, weight_decay=0.1)
    assert len(opt.param_groups) == 2, "Expected 2 param groups"
    assert opt.param_groups[0]["weight_decay"] == 0.1, "Decay group should have weight_decay"
    assert opt.param_groups[1]["weight_decay"] == 0.0, "No-decay group should have 0.0"

    decay_count = len(opt.param_groups[0]["params"])
    no_decay_count = len(opt.param_groups[1]["params"])
    total = sum(1 for p in model.parameters() if p.requires_grad)
    assert decay_count + no_decay_count == total, "All params must be accounted for"
    # embed.weight, linear.bias, norm.weight, norm.bias → no-decay (4)
    # linear.weight, head.weight → decay (2)
    assert decay_count == 2, f"Expected 2 decay params, got {decay_count}"
    assert no_decay_count == 4, f"Expected 4 no-decay params, got {no_decay_count}"
    print("[PASS] create_optimizer: correct param groups and counts")

    # ── Test 2: create_scheduler LR curve ──────────────────────────────
    peak_lr = 1e-3
    warmup = 10
    max_steps = 100
    min_ratio = 0.1

    opt2 = AdamW(model.parameters(), lr=peak_lr)
    sched = create_scheduler(opt2, warmup_steps=warmup, max_steps=max_steps, min_lr_ratio=min_ratio)

    lrs: list[float] = []
    for _step in range(max_steps):
        lrs.append(opt2.param_groups[0]["lr"])
        opt2.step()
        sched.step()

    # LR should rise during warmup
    for i in range(1, warmup):
        assert lrs[i] >= lrs[i - 1], f"LR should rise during warmup at step {i}"

    # LR at peak should be approximately peak_lr
    assert (
        abs(lrs[warmup] - peak_lr) < 1e-6
    ), f"LR at end of warmup should be ~peak_lr, got {lrs[warmup]}"

    # LR should generally decrease after warmup
    assert lrs[-1] < lrs[warmup], "LR at end should be less than at warmup end"

    # After warmup, LR should never go below min_lr_ratio * peak_lr
    min_lr = min_ratio * peak_lr
    for step_idx in range(warmup, len(lrs)):
        assert (
            lrs[step_idx] >= min_lr - 1e-9
        ), f"LR at step {step_idx} ({lrs[step_idx]}) below floor ({min_lr})"

    print("[PASS] create_scheduler: LR rises during warmup, decays, respects floor")

    # ── Test 3: get_grad_norm ──────────────────────────────────────────
    model.zero_grad()
    x = torch.randint(0, 64, (2, 8))  # (B, L)
    out = model(x)  # (B, L, 4)
    out.sum().backward()

    norm = get_grad_norm(model)
    assert norm > 0.0, f"Grad norm should be > 0 after backward, got {norm}"
    print(f"[PASS] get_grad_norm = {norm:.6f} (> 0 after backward)")

    # Zero grads → norm == 0
    model.zero_grad()
    norm_zero = get_grad_norm(model)
    assert norm_zero == 0.0, f"Grad norm should be 0 after zero_grad, got {norm_zero}"
    print("[PASS] get_grad_norm = 0.0 after zero_grad")

    # ── Test 4: validation errors ──────────────────────────────────────
    try:
        create_optimizer(model, lr=-1e-3, weight_decay=0.1)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
    print("[PASS] ValueError for negative lr")

    try:
        create_optimizer(model, lr=1e-3, weight_decay=-0.1)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
    print("[PASS] ValueError for negative weight_decay")

    try:
        create_scheduler(opt, warmup_steps=-1, max_steps=100)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
    print("[PASS] ValueError for negative warmup_steps")

    try:
        create_scheduler(opt, warmup_steps=0, max_steps=0)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
    print("[PASS] ValueError for max_steps <= 0")

    try:
        create_scheduler(opt, warmup_steps=100, max_steps=100)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
    print("[PASS] ValueError for warmup_steps >= max_steps")

    try:
        create_scheduler(opt, warmup_steps=10, max_steps=100, min_lr_ratio=-0.1)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
    print("[PASS] ValueError for min_lr_ratio < 0")

    try:
        create_scheduler(opt, warmup_steps=10, max_steps=100, min_lr_ratio=1.5)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
    print("[PASS] ValueError for min_lr_ratio > 1")

    print("=" * 60)
    print("[ALL PASS] optimizer.py smoke test complete")
    print("=" * 60)
