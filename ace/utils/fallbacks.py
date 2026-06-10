"""Pure-PyTorch fallback implementations for GPU-only libraries.

When ``mamba_ssm``, ``flash_attn``, ``megablocks``, or ``causal_conv1d`` are
unavailable (e.g. Windows CPU development), this module provides drop-in
replacements with **identical interfaces** so the rest of ACE can be developed
and unit-tested without a GPU.

.. warning::

    These fallbacks are **not** performance- or quality-equivalent to the real
    libraries.  They exist solely to keep the code runnable on CPU.  Always
    use the real libraries for training and benchmarking.

Provided:
    - :class:`FallbackMamba` — replaces ``mamba_ssm.Mamba``
    - :func:`fallback_flash_attention` — replaces ``flash_attn.flash_attn_func``
    - :class:`FallbackCausalConv1d` — replaces ``causal_conv1d.causal_conv1d_fn``
    - :class:`FallbackMoELayer` — replaces megablocks sparse MoE
    - :func:`get_mamba_class` — factory returning real or fallback Mamba
    - :func:`get_attention_fn` — factory returning real or fallback attention
    - :func:`get_moe_class` — factory returning real or fallback MoE layer

Usage::

    from ace.utils.fallbacks import get_mamba_class, get_attention_fn

    MambaImpl = get_mamba_class()
    mamba = MambaImpl(d_model=2048, d_state=16, d_conv=4, expand=2)
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ace.model.config import AceConfig
from ace.utils.env_detect import HAS_FLASH_ATTN, HAS_MAMBA, HAS_MEGABLOCKS

logger: logging.Logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Fallback Mamba (replaces mamba_ssm.Mamba)
# ═══════════════════════════════════════════════════════════════════════════


class FallbackMamba(nn.Module):
    """Pure-PyTorch approximation of ``mamba_ssm.Mamba``.

    Uses a GRU-like recurrent cell to approximate the selective state-space
    model.  The interface matches ``mamba_ssm.Mamba`` exactly so upstream
    code (e.g. :class:`MambaBlock`) can instantiate this transparently.

    .. note::

        This is **not** a faithful re-implementation of the Mamba S6 selective
        scan.  It merely provides the same tensor shapes and a reasonable
        non-trivial transformation so that gradient flow, shape checks, and
        integration tests pass on CPU.

    Attributes:
        IS_FALLBACK: Sentinel for test introspection.
        d_model: Input/output feature dimension.
        d_state: SSM state expansion factor.
        d_conv: Causal convolution width.
        expand: Channel expansion factor.
        d_inner: Expanded inner dimension (``d_model * expand``).
    """

    IS_FALLBACK: bool = True

    _warned: bool = False  # class-level one-shot warning flag

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        """Initialise FallbackMamba.

        Args:
            d_model: Input and output feature dimension.
            d_state: SSM state dimension (used by the recurrent cell).
            d_conv: Width of the causal 1-D convolution.
            expand: Channel expansion factor.

        Raises:
            ValueError: If any dimension is not positive.
        """
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if d_state <= 0:
            raise ValueError(f"d_state must be positive, got {d_state}")
        if d_conv <= 0:
            raise ValueError(f"d_conv must be positive, got {d_conv}")
        if expand <= 0:
            raise ValueError(f"expand must be positive, got {expand}")

        self.d_model: int = d_model
        self.d_state: int = d_state
        self.d_conv: int = d_conv
        self.expand: int = expand
        self.d_inner: int = d_model * expand

        if not FallbackMamba._warned:
            logger.warning(
                "Using FallbackMamba (pure PyTorch) — this is NOT equivalent "
                "to mamba_ssm.Mamba. Install mamba-ssm for full performance."
            )
            FallbackMamba._warned = True

        # ── Input projection ──────────────────────────────────────────
        self.in_proj = nn.Linear(
            d_model, self.d_inner * 2, bias=False
        )  # (D) -> (2 * d_inner)

        # ── Causal 1-D convolution ────────────────────────────────────
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )  # depthwise conv

        # ── GRU-like recurrent cell (approximates selective scan) ─────
        self.gru_cell = nn.GRUCell(
            input_size=self.d_inner,
            hidden_size=d_state,
        )  # (d_inner) -> (d_state)

        # ── State-to-output projection ────────────────────────────────
        self.state_proj = nn.Linear(
            d_state, self.d_inner, bias=False
        )  # (d_state) -> (d_inner)

        # ── Output projection ─────────────────────────────────────────
        self.out_proj = nn.Linear(
            self.d_inner, d_model, bias=False
        )  # (d_inner) -> (D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass mimicking ``mamba_ssm.Mamba``.

        Args:
            x: Input tensor of shape ``(B, L, D)``.

        Returns:
            Output tensor of shape ``(B, L, D)``.

        Raises:
            ValueError: If input last dimension does not match ``d_model``.
        """
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected last dimension {self.d_model}, got {x.shape[-1]}"
            )

        B, L, D = x.shape  # (B, L, D)

        # ── Input projection + gate split ─────────────────────────────
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_proj, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # ── Causal 1-D convolution ────────────────────────────────────
        # Conv1d expects (B, C, L)
        x_conv = x_proj.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]  # (B, d_inner, L) — trim causal padding
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)  # (B, L, d_inner)

        # ── Parallel GRU scan (approximates selective SSM) ────────────
        # For CPU efficiency, process full sequence at once via an
        # unrolled scan.  Not fast, but correct.
        h = torch.zeros(
            B, self.d_state, device=x.device, dtype=x.dtype
        )  # (B, d_state)

        outputs: list[torch.Tensor] = []
        for t in range(L):
            inp_t = x_conv[:, t, :]  # (B, d_inner)
            # GRUCell expects float32 inputs on CPU
            h = self.gru_cell(
                inp_t.float(), h.float()
            ).to(x.dtype)  # (B, d_state)
            out_t = self.state_proj(h)  # (B, d_inner)
            outputs.append(out_t)

        ssm_out = torch.stack(outputs, dim=1)  # (B, L, d_inner)

        # ── Gated output ──────────────────────────────────────────────
        y = ssm_out * F.silu(z)  # (B, L, d_inner)

        # ── Output projection ─────────────────────────────────────────
        out: torch.Tensor = self.out_proj(y)  # (B, L, D)
        return out  # (B, L, D)

    def extra_repr(self) -> str:
        """Show key hyper-parameters in repr."""
        return (
            f"d_model={self.d_model}, d_state={self.d_state}, "
            f"d_conv={self.d_conv}, expand={self.expand}, IS_FALLBACK=True"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Fallback Flash Attention (replaces flash_attn.flash_attn_func)
# ═══════════════════════════════════════════════════════════════════════════


def fallback_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Pure-PyTorch replacement for ``flash_attn.flash_attn_func``.

    Uses :func:`torch.nn.functional.scaled_dot_product_attention` internally.
    Handles GQA by expanding ``k`` and ``v`` heads to match ``q`` heads when
    they differ.

    .. note::

        This function has ``IS_FALLBACK = True`` as a function attribute
        so tests can detect the fallback path.

    Args:
        q: Queries of shape ``(B, L_q, H_q, D)``.
        k: Keys of shape ``(B, L_kv, H_kv, D)``.
        v: Values of shape ``(B, L_kv, H_kv, D)``.
        dropout_p: Dropout probability (applied during training only).
        softmax_scale: Optional scaling factor.  Defaults to
            ``1 / sqrt(D)`` if *None*.
        causal: If *True*, apply causal masking.
        **kwargs: Absorbed for forward-compatibility with ``flash_attn``
            keyword arguments.

    Returns:
        Attention output of shape ``(B, L_q, H_q, D)``.

    Raises:
        ValueError: If head dimensions of ``q``, ``k``, ``v`` don't match
            or GQA group size is not an integer.
    """
    B, L_q, H_q, D = q.shape  # (B, L_q, H_q, D)
    _, L_kv, H_kv, _ = k.shape  # (B, L_kv, H_kv, D)

    if k.shape[-1] != D or v.shape[-1] != D:
        raise ValueError(
            f"Head dimension mismatch: q has D={D}, "
            f"k has D={k.shape[-1]}, v has D={v.shape[-1]}"
        )

    # ── Handle GQA: expand k,v heads to match q heads ────────────────
    if H_kv != H_q:
        if H_q % H_kv != 0:
            raise ValueError(
                f"H_q ({H_q}) must be divisible by H_kv ({H_kv}) for GQA"
            )
        n_groups: int = H_q // H_kv
        # Expand (B, L, H_kv, D) -> (B, L, H_q, D)
        k = k.repeat_interleave(n_groups, dim=2)  # (B, L_kv, H_q, D)
        v = v.repeat_interleave(n_groups, dim=2)  # (B, L_kv, H_q, D)

    # ── Transpose to (B, H, L, D) for F.scaled_dot_product_attention ──
    q = q.transpose(1, 2)  # (B, H_q, L_q, D)
    k = k.transpose(1, 2)  # (B, H_q, L_kv, D)
    v = v.transpose(1, 2)  # (B, H_q, L_kv, D)

    # ── Scale ─────────────────────────────────────────────────────────
    if softmax_scale is not None:
        q = q * softmax_scale  # (B, H_q, L_q, D)

    # ── Compute attention ─────────────────────────────────────────────
    out = F.scaled_dot_product_attention(
        q,  # (B, H_q, L_q, D)
        k,  # (B, H_q, L_kv, D)
        v,  # (B, H_q, L_kv, D)
        dropout_p=dropout_p,
        is_causal=causal,
        scale=1.0 if softmax_scale is not None else None,
    )  # (B, H_q, L_q, D)

    # ── Transpose back to flash_attn output layout (B, L, H, D) ──────
    out = out.transpose(1, 2).contiguous()  # (B, L_q, H_q, D)
    return out  # (B, L_q, H_q, D)


# Tag the function so tests can detect fallback usage
fallback_flash_attention.IS_FALLBACK = True  # type: ignore[attr-defined]


# ═══════════════════════════════════════════════════════════════════════════
# Fallback Causal Conv1D (replaces causal_conv1d.causal_conv1d_fn)
# ═══════════════════════════════════════════════════════════════════════════


class FallbackCausalConv1d(nn.Module):
    """Pure-PyTorch causal 1-D convolution fallback.

    Drop-in replacement for ``causal_conv1d`` when the CUDA package is
    not available.

    Attributes:
        IS_FALLBACK: Sentinel for test introspection.
    """

    IS_FALLBACK: bool = True

    _warned: bool = False

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 4,
        bias: bool = True,
    ) -> None:
        """Initialise FallbackCausalConv1d.

        Args:
            d_model: Number of input/output channels.
            kernel_size: Width of the causal convolution.
            bias: Whether to include a bias term.

        Raises:
            ValueError: If *d_model* or *kernel_size* is not positive.
        """
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")

        self.d_model: int = d_model
        self.kernel_size: int = kernel_size

        if not FallbackCausalConv1d._warned:
            logger.warning(
                "Using FallbackCausalConv1d (pure PyTorch) — install "
                "causal-conv1d for CUDA-accelerated causal convolutions."
            )
            FallbackCausalConv1d._warned = True

        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_model,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal 1-D convolution.

        Args:
            x: Input tensor of shape ``(B, L, D)`` or ``(B, D, L)``.
                Auto-detects layout: if last dim == ``d_model``, assumes
                ``(B, L, D)`` and transposes internally.

        Returns:
            Output tensor matching the input layout.
        """
        needs_transpose: bool = x.shape[-1] == self.d_model and x.ndim == 3

        if needs_transpose:
            x = x.transpose(1, 2)  # (B, D, L)

        L: int = x.shape[-1]
        out: torch.Tensor = self.conv(x)[:, :, :L]  # (B, D, L) — trim causal padding

        if needs_transpose:
            out = out.transpose(1, 2)  # (B, L, D)

        return out


# ═══════════════════════════════════════════════════════════════════════════
# Fallback MoE Layer (replaces megablocks sparse MoE)
# ═══════════════════════════════════════════════════════════════════════════


class FallbackMoELayer(nn.Module):
    """Pure-PyTorch MoE layer matching :class:`MoEFeedForward` interface.

    Uses a simple loop over experts instead of megablocks' sparse
    block-diagonal matmuls.  Slower but correct — good enough for CPU
    development and testing.

    Attributes:
        IS_FALLBACK: Sentinel for test introspection.
        num_experts: Total number of expert FFNs.
        top_k: Number of experts activated per token.
        d_model: Model hidden dimension.
    """

    IS_FALLBACK: bool = True

    _warned: bool = False

    def __init__(self, config: AceConfig) -> None:
        """Initialise FallbackMoELayer from AceConfig.

        Args:
            config: Global model configuration.  Uses ``d_model``, ``d_ff``,
                ``num_experts``, and ``top_k_experts``.

        Raises:
            ValueError: If ``top_k_experts > num_experts`` or either is <= 0.
        """
        super().__init__()

        self.num_experts: int = config.num_experts
        self.top_k: int = config.top_k_experts
        self.d_model: int = config.d_model
        self.d_ff: int = config.d_ff

        if self.num_experts <= 0:
            raise ValueError(
                f"num_experts must be positive, got {self.num_experts}"
            )
        if self.top_k <= 0:
            raise ValueError(
                f"top_k_experts must be positive, got {self.top_k}"
            )
        if self.top_k > self.num_experts:
            raise ValueError(
                f"top_k_experts ({self.top_k}) must be <= "
                f"num_experts ({self.num_experts})"
            )

        if not FallbackMoELayer._warned:
            logger.warning(
                "Using FallbackMoELayer (pure PyTorch loop-over-experts) — "
                "install megablocks for sparse GPU-accelerated MoE."
            )
            FallbackMoELayer._warned = True

        # ── Router ────────────────────────────────────────────────────
        self.gate = nn.Linear(
            config.d_model, config.num_experts, bias=False
        )  # (D) -> (num_experts)

        # ── Expert FFNs (SwiGLU) ──────────────────────────────────────
        self.experts = nn.ModuleList([
            self._make_expert(config) for _ in range(config.num_experts)
        ])

    @staticmethod
    def _make_expert(config: AceConfig) -> nn.Module:
        """Create a single SwiGLU expert FFN.

        Args:
            config: Global model configuration.

        Returns:
            An ``nn.Module`` implementing the SwiGLU expert.
        """
        return _SwiGLUExpert(
            d_model=config.d_model,
            d_ff=config.d_ff,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch tokens to top-k experts and combine outputs.

        Args:
            x: Input tensor of shape ``(B, L, D)``.

        Returns:
            Tuple of:
                - ``output``: Combined expert outputs, shape ``(B, L, D)``.
                - ``aux_loss``: Scalar load-balancing loss.
        """
        B, L, D = x.shape  # (B, L, D)

        # ── Routing ───────────────────────────────────────────────────
        router_logits = self.gate(x)  # (B, L, num_experts)

        top_k_logits, expert_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )  # (B, L, top_k) each

        routing_weights = F.softmax(
            top_k_logits, dim=-1, dtype=torch.float32
        ).to(x.dtype)  # (B, L, top_k)

        # ── Auxiliary load-balancing loss ──────────────────────────────
        aux_loss = self._compute_aux_loss(router_logits)  # scalar

        # ── Expert dispatch (loop — simple but correct) ───────────────
        x_flat = x.reshape(-1, D)  # (B*L, D)
        routing_weights_flat = routing_weights.reshape(
            -1, self.top_k
        )  # (B*L, top_k)
        expert_indices_flat = expert_indices.reshape(
            -1, self.top_k
        )  # (B*L, top_k)

        output_flat = torch.zeros_like(x_flat)  # (B*L, D)

        for k_idx in range(self.top_k):
            expert_idx_k = expert_indices_flat[:, k_idx]  # (B*L,)
            weight_k = routing_weights_flat[:, k_idx]  # (B*L,)

            for e_idx in range(self.num_experts):
                token_mask = expert_idx_k == e_idx  # (B*L,)

                if not token_mask.any():
                    continue

                expert_input = x_flat[token_mask]  # (num_tokens, D)
                expert_output = self.experts[e_idx](expert_input)  # (num_tokens, D)
                expert_weight = weight_k[token_mask].unsqueeze(-1)  # (num_tokens, 1)
                output_flat[token_mask] += expert_output * expert_weight  # (num_tokens, D)

        output = output_flat.reshape(B, L, D)  # (B, L, D)
        return output, aux_loss  # (B, L, D), scalar

    def _compute_aux_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Compute Switch Transformer auxiliary load-balancing loss.

        Args:
            router_logits: Raw router logits of shape ``(B, L, num_experts)``.

        Returns:
            Scalar auxiliary loss tensor.
        """
        B, L, E = router_logits.shape  # (B, L, num_experts)
        logits_flat = router_logits.reshape(-1, E)  # (B*L, E)

        probs = F.softmax(logits_flat, dim=-1, dtype=torch.float32)  # (B*L, E)

        _, top1_indices = torch.topk(logits_flat, k=1, dim=-1)  # (B*L, 1)
        expert_mask = F.one_hot(
            top1_indices.squeeze(-1), num_classes=E
        ).float()  # (B*L, E)

        tokens_per_expert = expert_mask.mean(dim=0)  # (E,)
        mean_probs = probs.mean(dim=0)  # (E,)

        aux_loss: torch.Tensor = self.num_experts * torch.sum(
            tokens_per_expert * mean_probs
        )  # scalar
        return aux_loss

    def extra_repr(self) -> str:
        """Show MoE parameters in repr."""
        return (
            f"num_experts={self.num_experts}, top_k={self.top_k}, "
            f"d_model={self.d_model}, IS_FALLBACK=True"
        )


class _SwiGLUExpert(nn.Module):
    """Internal SwiGLU expert used by :class:`FallbackMoELayer`.

    Attributes:
        d_model: Input/output dimension.
        d_ff: Intermediate dimension.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        """Initialise SwiGLU projections.

        Args:
            d_model: Input/output feature dimension.
            d_ff: Intermediate (hidden) dimension.
        """
        super().__init__()
        self.d_model: int = d_model
        self.d_ff: int = d_ff

        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)  # (D) -> (d_ff)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)  # (D) -> (d_ff)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)  # (d_ff) -> (D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU FFN.

        Args:
            x: Input of shape ``(*, D)``.

        Returns:
            Output of shape ``(*, D)``.
        """
        gate = self.gate_proj(x)  # (*, d_ff)
        up = self.up_proj(x)  # (*, d_ff)
        h = F.silu(gate) * up  # (*, d_ff)
        out: torch.Tensor = self.down_proj(h)  # (*, D)
        return out


# ═══════════════════════════════════════════════════════════════════════════
# Factory Functions
# ═══════════════════════════════════════════════════════════════════════════


def get_mamba_class() -> type:
    """Return the best available Mamba implementation.

    Returns:
        ``mamba_ssm.Mamba`` if installed, else :class:`FallbackMamba`.
    """
    if HAS_MAMBA:
        from mamba_ssm import Mamba
        return Mamba
    logger.info("get_mamba_class() → FallbackMamba (mamba_ssm unavailable)")
    return FallbackMamba


def get_attention_fn() -> Callable[..., torch.Tensor]:
    """Return the best available flash-attention function.

    Returns:
        ``flash_attn.flash_attn_func`` if installed, else
        :func:`fallback_flash_attention`.
    """
    if HAS_FLASH_ATTN:
        from flash_attn import flash_attn_func
        return flash_attn_func
    logger.info(
        "get_attention_fn() → fallback_flash_attention "
        "(flash_attn unavailable)"
    )
    return fallback_flash_attention


def get_moe_class() -> type:
    """Return the best available MoE layer implementation.

    Returns:
        A megablocks-backed MoE class if installed, else
        :class:`FallbackMoELayer`.

    .. note::

        When megablocks *is* available we still return
        :class:`FallbackMoELayer` because the megablocks integration
        requires a custom wrapper (future work).  This function exists
        so the call-site is already correct once that wrapper lands.
    """
    if HAS_MEGABLOCKS:
        # TODO: return a megablocks-backed MoE wrapper once integrated
        logger.info(
            "megablocks is available but the ACE wrapper is not yet "
            "implemented — using FallbackMoELayer for now."
        )
    else:
        logger.info(
            "get_moe_class() → FallbackMoELayer (megablocks unavailable)"
        )
    return FallbackMoELayer


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    print("=" * 60)
    print("fallbacks.py — smoke test")
    print("=" * 60)

    device = torch.device("cpu")
    dtype = torch.float32
    B, L, D = 2, 16, 64

    # ── FallbackMamba ─────────────────────────────────────────────────
    mamba = FallbackMamba(d_model=D, d_state=16, d_conv=4, expand=2)
    assert mamba.IS_FALLBACK is True, "IS_FALLBACK must be True"

    x_mamba = torch.randn(B, L, D, device=device, dtype=dtype)  # (B, L, D)
    out_mamba = mamba(x_mamba)  # (B, L, D)
    assert out_mamba.shape == (B, L, D), f"FallbackMamba shape: {out_mamba.shape}"
    assert not torch.isnan(out_mamba).any(), "NaN in FallbackMamba output"
    print("[PASS] FallbackMamba forward")

    # Validation
    try:
        FallbackMamba(d_model=0)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
    print("[PASS] FallbackMamba rejects d_model=0")

    # Gradient flow
    x_grad = torch.randn(B, L, D, requires_grad=True)  # (B, L, D)
    out_grad = mamba(x_grad)  # (B, L, D)
    out_grad.sum().backward()
    assert x_grad.grad is not None, "No gradient flow through FallbackMamba"
    print("[PASS] FallbackMamba gradient flow")

    # ── fallback_flash_attention ──────────────────────────────────────
    H_q, H_kv, D_head = 8, 4, 16
    q = torch.randn(B, L, H_q, D_head, device=device, dtype=dtype)  # (B, L, H_q, D)
    k = torch.randn(B, L, H_kv, D_head, device=device, dtype=dtype)  # (B, L, H_kv, D)
    v = torch.randn(B, L, H_kv, D_head, device=device, dtype=dtype)  # (B, L, H_kv, D)

    out_attn = fallback_flash_attention(q, k, v, dropout_p=0.0, causal=True)
    assert out_attn.shape == (B, L, H_q, D_head), f"Attention shape: {out_attn.shape}"
    assert not torch.isnan(out_attn).any(), "NaN in fallback attention"
    assert fallback_flash_attention.IS_FALLBACK is True  # type: ignore[attr-defined]
    print("[PASS] fallback_flash_attention (with GQA)")

    # Non-GQA case
    k_full = torch.randn(B, L, H_q, D_head)  # (B, L, H_q, D)
    v_full = torch.randn(B, L, H_q, D_head)  # (B, L, H_q, D)
    out_full = fallback_flash_attention(q, k_full, v_full, causal=False)
    assert out_full.shape == (B, L, H_q, D_head), f"Non-GQA shape: {out_full.shape}"
    print("[PASS] fallback_flash_attention (non-GQA, non-causal)")

    # ── FallbackCausalConv1d ──────────────────────────────────────────
    conv = FallbackCausalConv1d(d_model=D, kernel_size=4)
    assert conv.IS_FALLBACK is True

    x_conv = torch.randn(B, L, D)  # (B, L, D)
    out_conv = conv(x_conv)  # (B, L, D)
    assert out_conv.shape == (B, L, D), f"CausalConv1d shape: {out_conv.shape}"
    print("[PASS] FallbackCausalConv1d")

    # ── FallbackMoELayer ──────────────────────────────────────────────
    cfg = AceConfig(
        d_model=D,
        n_heads=4,
        n_kv_heads=2,
        d_ff=256,
        n_layers=4,
        max_seq_len=128,
        vocab_size=256,
        num_experts=8,
        top_k_experts=2,
        mamba_layers=[],
        moe_layers=[0, 1, 2, 3],
        retnet_layers=[],
        dropout=0.0,
    )

    moe = FallbackMoELayer(cfg)
    assert moe.IS_FALLBACK is True

    x_moe = torch.randn(B, L, D, device=device, dtype=dtype)  # (B, L, D)
    out_moe, aux_loss = moe(x_moe)
    assert out_moe.shape == (B, L, D), f"MoE output shape: {out_moe.shape}"
    assert aux_loss.dim() == 0, f"aux_loss should be scalar, got dim={aux_loss.dim()}"
    assert aux_loss.item() >= 0, f"aux_loss should be non-negative: {aux_loss.item()}"
    assert not torch.isnan(out_moe).any(), "NaN in FallbackMoELayer output"
    print("[PASS] FallbackMoELayer forward")

    # Gradient flow through MoE
    x_moe_grad = torch.randn(B, L, D, requires_grad=True)  # (B, L, D)
    out_moe_grad, loss_moe_grad = moe(x_moe_grad)
    total_loss = out_moe_grad.sum() + loss_moe_grad
    total_loss.backward()
    assert x_moe_grad.grad is not None, "No gradient flow through FallbackMoELayer"
    assert not torch.isnan(x_moe_grad.grad).any(), "NaN in MoE gradient"
    print("[PASS] FallbackMoELayer gradient flow")

    # ── Factory functions ─────────────────────────────────────────────
    MambaClass = get_mamba_class()
    attn_fn = get_attention_fn()
    MoEClass = get_moe_class()

    # On CPU/Windows without GPU libs, all should be fallbacks
    if not HAS_MAMBA:
        assert MambaClass is FallbackMamba, "Factory should return FallbackMamba"
        print("[PASS] get_mamba_class() -> FallbackMamba")

    if not HAS_FLASH_ATTN:
        assert attn_fn is fallback_flash_attention, "Factory should return fallback"
        print("[PASS] get_attention_fn() -> fallback_flash_attention")

    assert MoEClass is FallbackMoELayer, "Factory should return FallbackMoELayer"
    print("[PASS] get_moe_class() -> FallbackMoELayer")

    print("=" * 60)
    print("[ALL PASS] fallbacks.py smoke test complete")
    print("=" * 60)
