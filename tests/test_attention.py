"""Tests for ace.model.attention — RotaryPositionalEmbedding & MultiHeadAttention.

Tests cover:
    - Output shapes for RoPE and MHA
    - KV-cache correctness (incremental generation)
    - GQA (n_kv_heads < n_heads)
    - No NaN on random input
    - Causal masking behaviour
    - Custom mask forwarding
    - Cache consistency: full-sequence vs step-by-step generation
"""

from __future__ import annotations

import pytest
import torch

from ace.model.attention import (
    MultiHeadAttention,
    RotaryPositionalEmbedding,
)
from ace.model.config import AceConfig

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
        max_seq_len=128,
        vocab_size=256,
        mamba_layers=[],
        moe_layers=[],
        retnet_layers=[],
        dropout=0.0,
    )


@pytest.fixture()
def device() -> torch.device:
    """Use CPU for deterministic tests."""
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────
# RotaryPositionalEmbedding tests
# ──────────────────────────────────────────────────────────────────────────


class TestRotaryPositionalEmbedding:
    """Tests for RotaryPositionalEmbedding."""

    def test_cos_sin_shapes(self, small_config: AceConfig, device: torch.device) -> None:
        """cos and sin caches have shape (seq_len, head_dim)."""
        rope = RotaryPositionalEmbedding(small_config).to(device)
        seq_len = 32
        cos, sin = rope(seq_len)

        assert cos.shape == (seq_len, small_config.head_dim)
        assert sin.shape == (seq_len, small_config.head_dim)

    def test_cache_extends_on_longer_seq(
        self, small_config: AceConfig, device: torch.device
    ) -> None:
        """Cache rebuilds transparently when seq_len exceeds max_seq_len."""
        rope = RotaryPositionalEmbedding(small_config).to(device)
        longer = small_config.max_seq_len + 64
        cos, sin = rope(longer)

        assert cos.shape == (longer, small_config.head_dim)
        assert sin.shape == (longer, small_config.head_dim)

    def test_apply_rotary_emb_shape(self, small_config: AceConfig, device: torch.device) -> None:
        """apply_rotary_emb preserves the input shape (B, H, L, D)."""
        rope = RotaryPositionalEmbedding(small_config).to(device)
        B, H, L = 2, 4, 16
        D = small_config.head_dim

        x = torch.randn(B, H, L, D, device=device)
        cos, sin = rope(L)
        out = RotaryPositionalEmbedding.apply_rotary_emb(x, cos, sin)

        assert out.shape == (B, H, L, D)

    def test_apply_rotary_emb_no_nan(self, small_config: AceConfig, device: torch.device) -> None:
        """Rotary embedding must not produce NaN."""
        rope = RotaryPositionalEmbedding(small_config).to(device)
        B, H, L = 3, 4, 24
        D = small_config.head_dim

        x = torch.randn(B, H, L, D, device=device)
        cos, sin = rope(L)
        out = RotaryPositionalEmbedding.apply_rotary_emb(x, cos, sin)

        assert not torch.isnan(out).any(), "NaN detected in rotary output"

    def test_even_head_dim_required(self, device: torch.device) -> None:
        """Must raise ValueError for odd head_dim."""
        # 3 heads with d_model=9 → head_dim=3 (odd)
        bad_cfg = AceConfig(
            d_model=9,
            n_heads=3,
            n_kv_heads=3,
            d_ff=36,
            n_layers=1,
            max_seq_len=32,
            vocab_size=256,
            mamba_layers=[],
            moe_layers=[],
            retnet_layers=[],
        )
        with pytest.raises(ValueError, match="even head_dim"):
            RotaryPositionalEmbedding(bad_cfg)

    def test_rotary_equivariance(self, small_config: AceConfig, device: torch.device) -> None:
        """Same position gets the same rotation regardless of batch."""
        rope = RotaryPositionalEmbedding(small_config).to(device)
        D = small_config.head_dim

        x = torch.randn(1, 1, 1, D, device=device).expand(4, 2, 1, D)  # (4,2,1,D)
        cos, sin = rope(1)
        out = RotaryPositionalEmbedding.apply_rotary_emb(x, cos, sin)

        # All batch/head slices should be identical since input is constant
        for b in range(4):
            for h in range(2):
                torch.testing.assert_close(out[0, 0], out[b, h])


# ──────────────────────────────────────────────────────────────────────────
# MultiHeadAttention tests
# ──────────────────────────────────────────────────────────────────────────


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention (with GQA)."""

    def test_output_shape_no_cache(self, small_config: AceConfig, device: torch.device) -> None:
        """Output shape matches (B, L, D) without cache."""
        mha = MultiHeadAttention(small_config).to(device)
        B, L = 2, 16
        x = torch.randn(B, L, small_config.d_model, device=device)

        out, cache = mha(x)

        assert out.shape == (B, L, small_config.d_model)
        assert cache is None

    def test_output_shape_with_cache(self, small_config: AceConfig, device: torch.device) -> None:
        """KV-cache is returned when use_cache=True."""
        mha = MultiHeadAttention(small_config).to(device)
        B, L = 2, 16
        x = torch.randn(B, L, small_config.d_model, device=device)

        out, cache = mha(x, use_cache=True)

        assert out.shape == (B, L, small_config.d_model)
        assert cache is not None

        cached_k, cached_v = cache
        assert cached_k.shape == (
            B,
            small_config.n_kv_heads,
            L,
            small_config.head_dim,
        )
        assert cached_v.shape == cached_k.shape

    def test_kv_cache_incremental(self, small_config: AceConfig, device: torch.device) -> None:
        """KV-cache length grows correctly during incremental generation."""
        mha = MultiHeadAttention(small_config).to(device)
        B = 2
        D = small_config.d_model

        # Step 1: process prompt of length 10
        prompt = torch.randn(B, 10, D, device=device)
        _, kv = mha(prompt, use_cache=True)
        assert kv is not None
        assert kv[0].shape[2] == 10

        # Step 2: generate one token at a time
        for step in range(1, 4):
            token = torch.randn(B, 1, D, device=device)
            out, kv = mha(token, past_key_value=kv, use_cache=True)

            assert out.shape == (B, 1, D)
            assert kv is not None
            assert (
                kv[0].shape[2] == 10 + step
            ), f"Expected cache length {10 + step}, got {kv[0].shape[2]}"

    def test_no_nan_output(self, small_config: AceConfig, device: torch.device) -> None:
        """MHA output must be NaN-free on random input."""
        mha = MultiHeadAttention(small_config).to(device)
        x = torch.randn(4, 32, small_config.d_model, device=device)

        out, _ = mha(x)

        assert not torch.isnan(out).any(), "NaN detected in MHA output"
        assert not torch.isinf(out).any(), "Inf detected in MHA output"

    def test_gqa_n_groups(self, small_config: AceConfig) -> None:
        """GQA group count is n_heads // n_kv_heads."""
        mha = MultiHeadAttention(small_config)

        assert small_config.n_kv_heads < small_config.n_heads
        expected_groups = small_config.n_heads // small_config.n_kv_heads
        assert mha.n_groups == expected_groups

    def test_gqa_non_gqa_same_shape(self, device: torch.device) -> None:
        """When n_kv_heads == n_heads (MHA), output shape is unchanged."""
        cfg_mha = AceConfig(
            d_model=64,
            n_heads=4,
            n_kv_heads=4,  # full MHA, no GQA
            d_ff=256,
            n_layers=1,
            max_seq_len=64,
            vocab_size=256,
            mamba_layers=[],
            moe_layers=[],
            retnet_layers=[],
            dropout=0.0,
        )
        mha = MultiHeadAttention(cfg_mha).to(device)
        x = torch.randn(2, 16, 64, device=device)

        out, _ = mha(x)
        assert out.shape == (2, 16, 64)
        assert mha.n_groups == 1

    def test_custom_mask(self, small_config: AceConfig, device: torch.device) -> None:
        """MHA accepts and uses a custom attention mask without crashing."""
        mha = MultiHeadAttention(small_config).to(device)
        B, L = 2, 8
        x = torch.randn(B, L, small_config.d_model, device=device)

        # Bool mask: lower-triangular (causal)
        mask = (
            torch.tril(torch.ones(L, L, device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, L, L)

        out, _ = mha(x, mask=mask)
        assert out.shape == (B, L, small_config.d_model)
        assert not torch.isnan(out).any()

    def test_deterministic_with_no_dropout(
        self, small_config: AceConfig, device: torch.device
    ) -> None:
        """With dropout=0.0 and eval mode, two forward passes yield same output."""
        mha = MultiHeadAttention(small_config).to(device).eval()
        x = torch.randn(1, 8, small_config.d_model, device=device)

        out1, _ = mha(x)
        out2, _ = mha(x)

        torch.testing.assert_close(out1, out2)

    def test_cache_consistency(self, small_config: AceConfig, device: torch.device) -> None:
        """Full-sequence output matches last-token output from cached generation.

        Verifies that the final hidden state from processing all tokens at
        once is close to the hidden state obtained by processing the last
        token with a KV cache built from the preceding tokens.
        """
        mha = MultiHeadAttention(small_config).to(device).eval()
        torch.manual_seed(42)

        B, L = 1, 8
        D = small_config.d_model
        x = torch.randn(B, L, D, device=device)

        # Full-sequence forward
        full_out, _ = mha(x)  # (B, L, D)
        last_full = full_out[:, -1:, :]  # (B, 1, D)

        # Incremental: prefix then last token
        prefix = x[:, :-1, :]  # (B, L-1, D)
        _, kv = mha(prefix, use_cache=True)

        last_token = x[:, -1:, :]  # (B, 1, D)
        last_out, _ = mha(last_token, past_key_value=kv, use_cache=True)

        torch.testing.assert_close(
            last_full,
            last_out,
            atol=1e-5,
            rtol=1e-4,
            msg="Cache-based last-token output diverges from full-sequence output",
        )

    def test_invalid_kv_heads_raises(self) -> None:
        """AceConfig rejects n_heads % n_kv_heads != 0 at construction time."""
        with pytest.raises(ValueError, match="divisible"):
            AceConfig(
                d_model=64,
                n_heads=4,
                n_kv_heads=3,  # 4 % 3 != 0
                d_ff=256,
                n_layers=1,
                max_seq_len=32,
                vocab_size=256,
                mamba_layers=[],
                moe_layers=[],
                retnet_layers=[],
            )
