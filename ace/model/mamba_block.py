"""Mamba SSM block for the ACE model.

Provides:
    - ``RMSNorm`` — Root Mean Square Layer Normalisation (pre-norm).
    - ``MambaBlock`` — Pre-norm RMSNorm → Mamba SSM → residual connection.

The block wraps ``mamba_ssm.Mamba`` which provides O(L) selective state-space
modelling, enabling ACE's 32k context window without quadratic attention cost.

If ``mamba_ssm`` is **not** installed, importing this module still succeeds —
``MambaBlock`` will raise a clear ``ImportError`` only when instantiated.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from ace.model.config import AceConfig
from ace.utils.fallbacks import get_mamba_class

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Mamba implementation: real mamba_ssm.Mamba or pure-PyTorch FallbackMamba
# ──────────────────────────────────────────────────────────────────────────

Mamba = get_mamba_class()


# ═══════════════════════════════════════════════════════════════════════════
# RMS Norm
# ═══════════════════════════════════════════════════════════════════════════


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation.

    Applies the transformation::

        y = x * (1 / rms(x)) * weight

    where ``rms(x) = sqrt(mean(x²) + eps)``.

    This is cheaper than full LayerNorm because it skips the mean-centering
    step, while performing similarly in practice for Transformer-family models.

    Attributes:
        d_model: Feature dimension being normalised.
        eps: Numerical stability epsilon.
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        """Initialise RMSNorm.

        Args:
            d_model: Size of the feature dimension to normalise.
            eps: Small constant for numerical stability.

        Raises:
            ValueError: If *d_model* is not positive.
        """
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # (D,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalisation.

        Args:
            x: Input tensor of shape ``(B, L, D)`` or ``(*, D)``.

        Returns:
            Normalised tensor with the same shape as *x*.
        """
        # x: (B, L, D) or (*, D)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # (B, L, 1) or (*, 1)
        x_normed = x / rms  # (B, L, D)
        return x_normed * self.weight  # (B, L, D)

    def extra_repr(self) -> str:
        """Show d_model and eps in repr."""
        return f"d_model={self.d_model}, eps={self.eps}"


# ═══════════════════════════════════════════════════════════════════════════
# Mamba Block
# ═══════════════════════════════════════════════════════════════════════════


class MambaBlock(nn.Module):
    """Pre-norm Mamba SSM block with residual connection.

    Architecture::

        residual = x                    # (B, L, D)
        x = RMSNorm(x)                  # (B, L, D)
        x = Mamba(x)                    # (B, L, D)
        output = residual + x           # (B, L, D)

    The inner ``Mamba`` module from ``mamba_ssm`` implements the selective
    state-space model with O(L) complexity, which is critical for handling
    ACE's 32k-token context window efficiently.

    Attributes:
        d_model: Model hidden dimension.
        d_state: SSM state expansion factor.
        d_conv: Width of the 1-D convolution inside Mamba.
        expand: Channel expansion factor inside Mamba.
        norm: Pre-norm RMSNorm layer.
        mamba: The core ``mamba_ssm.Mamba`` module.
    """

    def __init__(self, config: AceConfig) -> None:
        """Initialise MambaBlock from AceConfig.

        Uses ``mamba_ssm.Mamba`` when available, otherwise falls back to
        :class:`~ace.utils.fallbacks.FallbackMamba` (pure PyTorch) so the
        block works on Windows CPU without GPU libraries.

        Args:
            config: Global model configuration.  Uses ``d_model``,
                ``norm_eps``, and Mamba-specific defaults.
        """
        super().__init__()

        self.d_model: int = config.d_model
        # Mamba-specific hyper-parameters (sensible defaults from the paper)
        self.d_state: int = 16
        self.d_conv: int = 4
        self.expand: int = 2

        # ── Pre-norm ──────────────────────────────────────────────────
        self.norm = RMSNorm(
            d_model=config.d_model,
            eps=config.norm_eps,
        )

        # ── Core Mamba SSM (real or fallback) ─────────────────────────
        self.mamba = Mamba(
            d_model=config.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pre-norm → Mamba SSM → residual.

        Args:
            x: Input tensor of shape ``(B, L, D)``.

        Returns:
            Output tensor of shape ``(B, L, D)``.

        Raises:
            ValueError: If the last dimension of *x* does not match
                ``d_model``.
        """
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected last dimension {self.d_model}, " f"got {x.shape[-1]}")

        residual = x  # (B, L, D)
        x_normed = self.norm(x)  # (B, L, D)
        mamba_out: torch.Tensor = self.mamba(x_normed)  # (B, L, D)
        output = residual + mamba_out  # (B, L, D)
        return output  # (B, L, D)

    def extra_repr(self) -> str:
        """Show key hyper-parameters in repr."""
        return (
            f"d_model={self.d_model}, d_state={self.d_state}, "
            f"d_conv={self.d_conv}, expand={self.expand}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("mamba_block.py — smoke test")
    print("=" * 60)

    # ── RMSNorm standalone test ───────────────────────────────────────
    norm = RMSNorm(d_model=64, eps=1e-5)
    x_norm = torch.randn(2, 16, 64)  # (B, L, D)
    y_norm = norm(x_norm)  # (B, L, D)
    assert y_norm.shape == x_norm.shape, f"RMSNorm shape mismatch: {y_norm.shape}"
    assert not torch.isnan(y_norm).any(), "NaN in RMSNorm output"
    print("[PASS] RMSNorm")

    # ── RMSNorm validation ────────────────────────────────────────────
    try:
        RMSNorm(d_model=0)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
    print("[PASS] RMSNorm rejects d_model=0")

    # ── MambaBlock test (always works — real or fallback) ───────────────
    is_fallback: bool = getattr(Mamba, "IS_FALLBACK", False)
    impl_name: str = "FallbackMamba" if is_fallback else "mamba_ssm.Mamba"
    print(f"  Using: {impl_name}")

    cfg = AceConfig(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        d_ff=256,
        n_layers=2,
        max_seq_len=256,
        vocab_size=256,
        mamba_layers=[0, 1],
        moe_layers=[],
        retnet_layers=[],
        dropout=0.0,
    )
    block = MambaBlock(cfg)
    x = torch.randn(2, 32, 64)  # (B, L, D)
    out = block(x)  # (B, L, D)
    assert out.shape == (2, 32, 64), f"MambaBlock shape: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in MambaBlock output"
    print(f"[PASS] MambaBlock forward ({impl_name})")

    # Test dimension mismatch
    try:
        bad = torch.randn(2, 32, 128)  # wrong D
        block(bad)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
    print("[PASS] MambaBlock rejects mismatched dimensions")

    print("=" * 60)
    print("[ALL PASS] mamba_block.py smoke test complete")
    print("=" * 60)

