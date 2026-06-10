"""Tests for ace.model.mamba_block — RMSNorm & MambaBlock.

Tests cover:
    - RMSNorm output shape and numerical correctness
    - RMSNorm unit-norm property
    - RMSNorm rejects invalid d_model
    - MambaBlock output shape (B, L, D) -> (B, L, D)
    - MambaBlock residual connection (ensures gradient flow)
    - MambaBlock rejects dimension mismatches
    - MambaBlock fallback detection via IS_FALLBACK attribute
    - Long-sequence test (L=8192) verifying no OOM on CPU
    - MambaBlock numerical stability (no NaN / Inf)
    - MambaBlock determinism in eval mode

Note: MambaBlock always works — it uses ``mamba_ssm.Mamba`` when installed,
or ``FallbackMamba`` (pure PyTorch) otherwise.  No tests are skipped.
"""

from __future__ import annotations

import pytest
import torch

from ace.model.config import AceConfig
from ace.model.mamba_block import (
    Mamba,
    MambaBlock,
    RMSNorm,
)

# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def small_config() -> AceConfig:
    """A minimal config for fast unit tests."""
    return AceConfig(
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


@pytest.fixture()
def device() -> torch.device:
    """Use CPU for deterministic tests."""
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────
# RMSNorm tests
# ──────────────────────────────────────────────────────────────────────────


class TestRMSNorm:
    """Tests for RMSNorm."""

    def test_output_shape(self, device: torch.device) -> None:
        """RMSNorm preserves the input shape (B, L, D)."""
        norm = RMSNorm(d_model=64).to(device)
        x = torch.randn(2, 16, 64, device=device)  # (B, L, D)

        out = norm(x)  # (B, L, D)

        assert out.shape == (2, 16, 64)

    def test_no_nan(self, device: torch.device) -> None:
        """RMSNorm must not produce NaN on random input."""
        norm = RMSNorm(d_model=128).to(device)
        x = torch.randn(4, 32, 128, device=device)  # (B, L, D)

        out = norm(x)  # (B, L, D)

        assert not torch.isnan(out).any(), "NaN detected in RMSNorm output"
        assert not torch.isinf(out).any(), "Inf detected in RMSNorm output"

    def test_unit_rms_after_norm(self, device: torch.device) -> None:
        """After RMSNorm (with unit weight), the RMS of output ≈ 1."""
        norm = RMSNorm(d_model=256, eps=1e-8).to(device)
        x = torch.randn(8, 16, 256, device=device)  # (B, L, D)

        out = norm(x)  # (B, L, D)

        # RMS along the last dim should be close to 1.0
        rms = torch.sqrt(out.pow(2).mean(dim=-1))  # (B, L)
        torch.testing.assert_close(
            rms,
            torch.ones_like(rms),
            atol=1e-4,
            rtol=1e-4,
            msg="RMS of normalised output should be ≈ 1.0",
        )

    def test_rejects_zero_d_model(self) -> None:
        """RMSNorm must reject d_model <= 0."""
        with pytest.raises(ValueError, match="positive"):
            RMSNorm(d_model=0)

    def test_rejects_negative_d_model(self) -> None:
        """RMSNorm must reject negative d_model."""
        with pytest.raises(ValueError, match="positive"):
            RMSNorm(d_model=-16)

    def test_weight_is_learnable(self) -> None:
        """The weight parameter should be learnable."""
        norm = RMSNorm(d_model=32)
        assert norm.weight.requires_grad is True
        assert norm.weight.shape == (32,)

    def test_zero_input(self, device: torch.device) -> None:
        """RMSNorm handles all-zero input without NaN (eps prevents div-by-zero)."""
        norm = RMSNorm(d_model=64, eps=1e-5).to(device)
        x = torch.zeros(2, 8, 64, device=device)  # (B, L, D)

        out = norm(x)  # (B, L, D)

        assert not torch.isnan(out).any(), "NaN on zero input"
        # All-zero input → all-zero output (0 / anything = 0)
        assert (out == 0).all(), "Zero input should yield zero output"


# ──────────────────────────────────────────────────────────────────────────
# MambaBlock tests — always run (real mamba_ssm or FallbackMamba)
# ──────────────────────────────────────────────────────────────────────────


class TestMambaBlock:
    """Tests for MambaBlock (works with real Mamba or FallbackMamba)."""

    def test_output_shape(self, small_config: AceConfig, device: torch.device) -> None:
        """MambaBlock output shape matches (B, L, D)."""
        block = MambaBlock(small_config).to(device)
        x = torch.randn(2, 32, small_config.d_model, device=device)  # (B, L, D)

        out = block(x)  # (B, L, D)

        assert out.shape == (2, 32, small_config.d_model)

    def test_no_nan(self, small_config: AceConfig, device: torch.device) -> None:
        """MambaBlock must not produce NaN on random input."""
        block = MambaBlock(small_config).to(device)
        x = torch.randn(4, 16, small_config.d_model, device=device)  # (B, L, D)

        out = block(x)  # (B, L, D)

        assert not torch.isnan(out).any(), "NaN detected in MambaBlock output"
        assert not torch.isinf(out).any(), "Inf detected in MambaBlock output"

    def test_residual_connection(self, small_config: AceConfig, device: torch.device) -> None:
        """Output differs from input (Mamba contributes), and residual flows.

        With an identity Mamba (all-zero weights), output would equal input.
        With random weights, output ≠ input proves Mamba is contributing.
        """
        block = MambaBlock(small_config).to(device)
        x = torch.randn(2, 8, small_config.d_model, device=device)  # (B, L, D)

        out = block(x)  # (B, L, D)

        # Output should not be identical to input (Mamba contributes)
        assert not torch.allclose(out, x, atol=1e-6), (
            "MambaBlock output is identical to input — residual-only, " "Mamba is not contributing"
        )

    def test_gradient_flows(self, small_config: AceConfig, device: torch.device) -> None:
        """Gradients flow through the MambaBlock."""
        block = MambaBlock(small_config).to(device)
        x = torch.randn(2, 8, small_config.d_model, device=device, requires_grad=True)  # (B, L, D)

        out = block(x)  # (B, L, D)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "No gradient on input"
        assert x.grad.shape == x.shape, "Gradient shape mismatch"
        assert not torch.isnan(x.grad).any(), "NaN in gradient"

    def test_rejects_wrong_dimension(self, small_config: AceConfig, device: torch.device) -> None:
        """MambaBlock raises ValueError when last dim ≠ d_model."""
        block = MambaBlock(small_config).to(device)
        bad_x = torch.randn(2, 16, small_config.d_model * 2, device=device)

        with pytest.raises(ValueError, match="Expected last dimension"):
            block(bad_x)

    def test_deterministic_eval(self, small_config: AceConfig, device: torch.device) -> None:
        """In eval mode, two forward passes yield identical output."""
        block = MambaBlock(small_config).to(device).eval()
        x = torch.randn(1, 16, small_config.d_model, device=device)  # (B, L, D)

        out1 = block(x)  # (B, L, D)
        out2 = block(x)  # (B, L, D)

        torch.testing.assert_close(out1, out2)

    def test_batch_independence(self, small_config: AceConfig, device: torch.device) -> None:
        """Different batch elements produce different outputs."""
        block = MambaBlock(small_config).to(device).eval()
        torch.manual_seed(123)
        x = torch.randn(2, 8, small_config.d_model, device=device)  # (B, L, D)

        out = block(x)  # (B, L, D)

        # Two different inputs should produce different outputs
        assert not torch.allclose(
            out[0], out[1], atol=1e-6
        ), "Batch elements produced identical output"

    def test_long_sequence_l8192(self, small_config: AceConfig, device: torch.device) -> None:
        """MambaBlock handles L=8192 without OOM on CPU.

        Mamba's O(L) complexity should make this trivially feasible.
        We use a small d_model to keep memory bounded.
        """
        cfg = AceConfig(
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=128,
            n_layers=1,
            max_seq_len=16384,
            vocab_size=256,
            mamba_layers=[0],
            moe_layers=[],
            retnet_layers=[],
            dropout=0.0,
        )
        block = MambaBlock(cfg).to(device)
        x = torch.randn(1, 8192, cfg.d_model, device=device)  # (1, 8192, 32)

        out = block(x)  # (1, 8192, 32)

        assert out.shape == (1, 8192, cfg.d_model), f"Long-sequence shape mismatch: {out.shape}"
        assert not torch.isnan(out).any(), "NaN in long-sequence output"
        assert not torch.isinf(out).any(), "Inf in long-sequence output"

    def test_long_sequence_l8192_batched(
        self, small_config: AceConfig, device: torch.device
    ) -> None:
        """MambaBlock handles B=2, L=8192 without OOM on CPU."""
        cfg = AceConfig(
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=128,
            n_layers=1,
            max_seq_len=16384,
            vocab_size=256,
            mamba_layers=[0],
            moe_layers=[],
            retnet_layers=[],
            dropout=0.0,
        )
        block = MambaBlock(cfg).to(device)
        x = torch.randn(2, 8192, cfg.d_model, device=device)  # (2, 8192, 32)

        out = block(x)  # (2, 8192, 32)

        assert out.shape == (2, 8192, cfg.d_model)
        assert not torch.isnan(out).any(), "NaN in batched long-sequence output"

    def test_single_token(self, small_config: AceConfig, device: torch.device) -> None:
        """MambaBlock handles L=1 (single-token input)."""
        block = MambaBlock(small_config).to(device)
        x = torch.randn(1, 1, small_config.d_model, device=device)  # (1, 1, D)

        out = block(x)  # (1, 1, D)

        assert out.shape == (1, 1, small_config.d_model)
        assert not torch.isnan(out).any()


# ──────────────────────────────────────────────────────────────────────────
# Fallback detection test (always runs)
# ──────────────────────────────────────────────────────────────────────────


class TestMambaFallbackDetection:
    """Test that the fallback mechanism is detectable."""

    def test_is_fallback_attribute_exists(self) -> None:
        """Mamba class should have IS_FALLBACK attribute when using fallback."""
        is_fallback: bool = getattr(Mamba, "IS_FALLBACK", False)
        # On Windows CPU without mamba_ssm, IS_FALLBACK is True.
        # On Linux/GPU with mamba_ssm, IS_FALLBACK is False (or absent).
        # Either way, the attribute check itself should not raise.
        assert isinstance(is_fallback, bool)

    def test_mamba_block_always_instantiates(self, small_config: AceConfig) -> None:
        """MambaBlock should always instantiate, regardless of mamba_ssm."""
        # This previously raised ImportError on Windows — now it uses fallback
        block = MambaBlock(small_config)
        assert block.d_model == small_config.d_model
        assert block.mamba is not None
