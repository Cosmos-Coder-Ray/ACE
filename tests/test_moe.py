"""Tests for ace.model.moe — Mixture-of-Experts module.

Covers:
    - Expert: SwiGLU FFN output shape, no NaNs, gradient flow.
    - TopKRouter: output shapes, valid expert indices, weight normalisation,
      deterministic behaviour.
    - compute_aux_loss: scalar output, non-negative, gradient flow,
      response to uniform vs. skewed routing.
    - MoEFeedForward: output shape matches input, aux_loss is scalar,
      gradient flow, all experts receive tokens.
    - SparseMoEBlock: residual connection, dimension mismatch rejection,
      gradient flow, output shape.
"""

from __future__ import annotations

import pytest
import torch

from ace.model.config import AceConfig
from ace.model.moe import (
    Expert,
    MoEFeedForward,
    RMSNorm,
    SparseMoEBlock,
    TopKRouter,
)

# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def tiny_config() -> AceConfig:
    """Minimal AceConfig for fast tests."""
    return AceConfig(
        d_model=64,
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


@pytest.fixture
def device() -> torch.device:
    """Use CPU for deterministic testing."""
    return torch.device("cpu")


@pytest.fixture
def batch_params() -> tuple[int, int]:
    """Return (B, L) for test inputs."""
    return 2, 16


# ══════════════════════════════════════════════════════════════════════════
# RMSNorm
# ══════════════════════════════════════════════════════════════════════════


class TestRMSNorm:
    """Tests for the RMSNorm module."""

    def test_output_shape(self, tiny_config: AceConfig, device: torch.device) -> None:
        """RMSNorm output shape matches input shape."""
        norm = RMSNorm(d_model=tiny_config.d_model, eps=tiny_config.norm_eps).to(device)
        x = torch.randn(2, 16, tiny_config.d_model, device=device)  # (B, L, D)
        out = norm(x)  # (B, L, D)
        assert out.shape == x.shape

    def test_no_nans(self, tiny_config: AceConfig, device: torch.device) -> None:
        """RMSNorm output contains no NaN values."""
        norm = RMSNorm(d_model=tiny_config.d_model).to(device)
        x = torch.randn(2, 16, tiny_config.d_model, device=device)  # (B, L, D)
        out = norm(x)  # (B, L, D)
        assert not torch.isnan(out).any()

    def test_rejects_zero_dim(self) -> None:
        """RMSNorm raises ValueError for non-positive d_model."""
        with pytest.raises(ValueError, match="d_model must be positive"):
            RMSNorm(d_model=0)

    def test_rejects_negative_dim(self) -> None:
        """RMSNorm raises ValueError for negative d_model."""
        with pytest.raises(ValueError, match="d_model must be positive"):
            RMSNorm(d_model=-5)


# ══════════════════════════════════════════════════════════════════════════
# Expert
# ══════════════════════════════════════════════════════════════════════════


class TestExpert:
    """Tests for the SwiGLU Expert FFN."""

    def test_output_shape(
        self,
        tiny_config: AceConfig,
        device: torch.device,
        batch_params: tuple[int, int],
    ) -> None:
        """Expert output shape matches input shape (B, L, D)."""
        B, L = batch_params
        expert = Expert(tiny_config).to(device)
        x = torch.randn(B, L, tiny_config.d_model, device=device)  # (B, L, D)
        out = expert(x)  # (B, L, D)
        assert out.shape == (B, L, tiny_config.d_model)

    def test_no_nans(
        self,
        tiny_config: AceConfig,
        device: torch.device,
        batch_params: tuple[int, int],
    ) -> None:
        """Expert output contains no NaN values."""
        B, L = batch_params
        expert = Expert(tiny_config).to(device)
        x = torch.randn(B, L, tiny_config.d_model, device=device)  # (B, L, D)
        out = expert(x)  # (B, L, D)
        assert not torch.isnan(out).any()

    def test_flat_input(self, tiny_config: AceConfig, device: torch.device) -> None:
        """Expert handles flat (T, D) input (tokens routed by MoE)."""
        expert = Expert(tiny_config).to(device)
        x = torch.randn(10, tiny_config.d_model, device=device)  # (T, D)
        out = expert(x)  # (T, D)
        assert out.shape == (10, tiny_config.d_model)

    def test_gradient_flow(
        self,
        tiny_config: AceConfig,
        device: torch.device,
        batch_params: tuple[int, int],
    ) -> None:
        """Gradients flow through the Expert."""
        B, L = batch_params
        expert = Expert(tiny_config).to(device)
        x = torch.randn(B, L, tiny_config.d_model, device=device, requires_grad=True)  # (B, L, D)
        out = expert(x)  # (B, L, D)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_three_projections(self, tiny_config: AceConfig) -> None:
        """Expert has exactly the three SwiGLU projections."""
        expert = Expert(tiny_config)
        assert hasattr(expert, "gate_proj")
        assert hasattr(expert, "up_proj")
        assert hasattr(expert, "down_proj")
        assert expert.gate_proj.in_features == tiny_config.d_model
        assert expert.gate_proj.out_features == tiny_config.d_ff
        assert expert.down_proj.in_features == tiny_config.d_ff
        assert expert.down_proj.out_features == tiny_config.d_model


# ══════════════════════════════════════════════════════════════════════════
# TopKRouter
# ══════════════════════════════════════════════════════════════════════════


class TestTopKRouter:
    """Tests for the TopKRouter module."""

    def test_output_shapes(
        self,
        tiny_config: AceConfig,
        device: torch.device,
        batch_params: tuple[int, int],
    ) -> None:
        """Router returns correct shapes for weights, indices, and logits."""
        B, L = batch_params
        router = TopKRouter(tiny_config).to(device)
        x = torch.randn(B, L, tiny_config.d_model, device=device)  # (B, L, D)
        weights, indices, logits = router(x)

        assert weights.shape == (B, L, tiny_config.top_k_experts)
        assert indices.shape == (B, L, tiny_config.top_k_experts)
        assert logits.shape == (B, L, tiny_config.num_experts)

    def test_indices_in_valid_range(
        self,
        tiny_config: AceConfig,
        device: torch.device,
        batch_params: tuple[int, int],
    ) -> None:
        """Expert indices are in [0, num_experts)."""
        B, L = batch_params
        router = TopKRouter(tiny_config).to(device)
        x = torch.randn(B, L, tiny_config.d_model, device=device)  # (B, L, D)
        _, indices, _ = router(x)  # (B, L, top_k)

        assert (indices >= 0).all(), "Found negative expert index"
        assert (
            indices < tiny_config.num_experts
        ).all(), f"Found expert index >= {tiny_config.num_experts}"

    def test_weights_sum_to_one(
        self,
        tiny_config: AceConfig,
        device: torch.device,
        batch_params: tuple[int, int],
    ) -> None:
        """Routing weights sum to 1.0 per token (softmax normalization)."""
        B, L = batch_params
        router = TopKRouter(tiny_config).to(device)
        x = torch.randn(B, L, tiny_config.d_model, device=device)  # (B, L, D)
        weights, _, _ = router(x)  # (B, L, top_k)

        weight_sums = weights.sum(dim=-1)  # (B, L)
        assert torch.allclose(
            weight_sums, torch.ones_like(weight_sums), atol=1e-5
        ), f"Weights do not sum to 1: {weight_sums}"

    def test_weights_nonnegative(
        self,
        tiny_config: AceConfig,
        device: torch.device,
        batch_params: tuple[int, int],
    ) -> None:
        """Routing weights are non-negative (from softmax)."""
        B, L = batch_params
        router = TopKRouter(tiny_config).to(device)
        x = torch.randn(B, L, tiny_config.d_model, device=device)  # (B, L, D)
        weights, _, _ = router(x)  # (B, L, top_k)
        assert (weights >= 0).all(), "Found negative routing weight"

    def test_top_k_selection(self, tiny_config: AceConfig, device: torch.device) -> None:
        """Selected indices correspond to the actual top-k logit values."""
        router = TopKRouter(tiny_config).to(device)
        x = torch.randn(1, 1, tiny_config.d_model, device=device)  # (1, 1, D)
        _, indices, logits = router(x)

        # Manually compute top-k
        _, expected_indices = torch.topk(logits, tiny_config.top_k_experts, dim=-1)  # (1, 1, top_k)
        assert torch.equal(indices, expected_indices)

    def test_rejects_invalid_config(self) -> None:
        """Router raises ValueError for invalid expert configurations."""
        # top_k > num_experts is caught by AceConfig.__post_init__
        with pytest.raises(ValueError):
            AceConfig(
                d_model=64,
                n_heads=4,
                n_kv_heads=2,
                d_ff=256,
                n_layers=2,
                vocab_size=256,
                num_experts=2,
                top_k_experts=5,
                mamba_layers=[],
                moe_layers=[],
                retnet_layers=[],
            )


# ══════════════════════════════════════════════════════════════════════════
# Auxiliary Loss
# ══════════════════════════════════════════════════════════════════════════


class TestAuxLoss:
    """Tests for the auxiliary load-balancing loss."""

    def test_is_scalar(
        self,
        tiny_config: AceConfig,
        device: torch.device,
        batch_params: tuple[int, int],
    ) -> None:
        """Auxiliary loss is a scalar tensor (0-dim)."""
        B, L = batch_params
        router = TopKRouter(tiny_config).to(device)
        x = torch.randn(B, L, tiny_config.d_model, device=device)  # (B, L, D)
        _, _, logits = router(x)  # (B, L, num_experts)
        aux_loss = router.compute_aux_loss(logits)  # scalar
        assert aux_loss.dim() == 0, f"Expected scalar, got dim={aux_loss.dim()}"

    def test_is_nonnegative(
        self,
        tiny_config: AceConfig,
        device: torch.device,
        batch_params: tuple[int, int],
    ) -> None:
        """Auxiliary loss is non-negative."""
        B, L = batch_params
        router = TopKRouter(tiny_config).to(device)
        x = torch.randn(B, L, tiny_config.d_model, device=device)  # (B, L, D)
        _, _, logits = router(x)  # (B, L, num_experts)
        aux_loss = router.compute_aux_loss(logits)  # scalar
        assert aux_loss.item() >= 0.0

    def test_no_nan(
        self,
        tiny_config: AceConfig,
        device: torch.device,
        batch_params: tuple[int, int],
    ) -> None:
        """Auxiliary loss contains no NaN."""
        B, L = batch_params
        router = TopKRouter(tiny_config).to(device)
        x = torch.randn(B, L, tiny_config.d_model, device=device)  # (B, L, D)
        _, _, logits = router(x)  # (B, L, num_experts)
        aux_loss = router.compute_aux_loss(logits)  # scalar
        assert not torch.isnan(aux_loss), "NaN in aux_loss"

    def test_gradient_flows_through_aux_loss(
        self,
        tiny_config: AceConfig,
        device: torch.device,
        batch_params: tuple[int, int],
    ) -> None:
        """Gradient flows through the auxiliary loss to the router gate."""
        B, L = batch_params
        router = TopKRouter(tiny_config).to(device)
        x = torch.randn(B, L, tiny_config.d_model, device=device)  # (B, L, D)
        _, _, logits = router(x)  # (B, L, num_experts)
        aux_loss = router.compute_aux_loss(logits)  # scalar
        aux_loss.backward()

        assert router.gate.weight.grad is not None, "No gradient on router gate weight"
        assert not torch.isnan(router.gate.weight.grad).any(), "NaN in router gate gradient"

    def test_uniform_routing_lower_loss(self, tiny_config: AceConfig, device: torch.device) -> None:
        """Uniform routing logits should produce lower aux_loss than skewed.

        This validates that the load-balancing loss penalises imbalance.
        """
        router = TopKRouter(tiny_config).to(device)

        # Uniform logits — all experts equally likely
        uniform_logits = torch.zeros(4, 32, tiny_config.num_experts, device=device)  # (B, L, E)

        # Skewed logits — one expert strongly preferred
        skewed_logits = torch.zeros(4, 32, tiny_config.num_experts, device=device)  # (B, L, E)
        skewed_logits[:, :, 0] = 100.0  # all tokens → expert 0

        loss_uniform = router.compute_aux_loss(uniform_logits)  # scalar
        loss_skewed = router.compute_aux_loss(skewed_logits)  # scalar

        assert loss_skewed > loss_uniform, (
            f"Skewed loss ({loss_skewed.item():.4f}) should be > "
            f"uniform loss ({loss_uniform.item():.4f})"
        )


# ══════════════════════════════════════════════════════════════════════════
# MoEFeedForward
# ══════════════════════════════════════════════════════════════════════════


class TestMoEFeedForward:
    """Tests for the MoEFeedForward module."""

    def test_output_shape(
        self,
        tiny_config: AceConfig,
        device: torch.device,
        batch_params: tuple[int, int],
    ) -> None:
        """MoEFeedForward output shape matches input shape."""
        B, L = batch_params
        moe = MoEFeedForward(tiny_config).to(device)
        x = torch.randn(B, L, tiny_config.d_model, device=device)  # (B, L, D)
        out, aux_loss = moe(x)  # (B, L, D), scalar
        assert out.shape == (B, L, tiny_config.d_model)

    def test_aux_loss_scalar(
        self,
        tiny_config: AceConfig,
        device: torch.device,
        batch_params: tuple[int, int],
    ) -> None:
        """MoEFeedForward returns a scalar aux_loss."""
        B, L = batch_params
        moe = MoEFeedForward(tiny_config).to(device)
        x = torch.randn(B, L, tiny_config.d_model, device=device)  # (B, L, D)
        _, aux_loss = moe(x)  # scalar
        assert aux_loss.dim() == 0

    def test_no_nans(
        self,
        tiny_config: AceConfig,
        device: torch.device,
        batch_params: tuple[int, int],
    ) -> None:
        """MoEFeedForward output is free of NaN values."""
        B, L = batch_params
        moe = MoEFeedForward(tiny_config).to(device)
        x = torch.randn(B, L, tiny_config.d_model, device=device)  # (B, L, D)
        out, aux_loss = moe(x)  # (B, L, D), scalar
        assert not torch.isnan(out).any()
        assert not torch.isnan(aux_loss)

    def test_gradient_flow(
        self,
        tiny_config: AceConfig,
        device: torch.device,
        batch_params: tuple[int, int],
    ) -> None:
        """Gradients flow through MoEFeedForward to input."""
        B, L = batch_params
        moe = MoEFeedForward(tiny_config).to(device)
        x = torch.randn(B, L, tiny_config.d_model, device=device, requires_grad=True)  # (B, L, D)
        out, aux_loss = moe(x)  # (B, L, D), scalar
        (out.sum() + aux_loss).backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_correct_num_experts(self, tiny_config: AceConfig) -> None:
        """MoEFeedForward has the correct number of experts."""
        moe = MoEFeedForward(tiny_config)
        assert len(moe.experts) == tiny_config.num_experts

    def test_experts_are_independent(self, tiny_config: AceConfig) -> None:
        """Each expert has its own independent parameters."""
        moe = MoEFeedForward(tiny_config)
        # Check that expert 0 and expert 1 have different parameter tensors
        for p0, p1 in zip(moe.experts[0].parameters(), moe.experts[1].parameters(), strict=False):
            assert p0.data_ptr() != p1.data_ptr(), "Experts should not share parameter storage"


# ══════════════════════════════════════════════════════════════════════════
# SparseMoEBlock
# ══════════════════════════════════════════════════════════════════════════


class TestSparseMoEBlock:
    """Tests for the SparseMoEBlock module."""

    def test_output_shape(
        self,
        tiny_config: AceConfig,
        device: torch.device,
        batch_params: tuple[int, int],
    ) -> None:
        """SparseMoEBlock output shape matches input shape."""
        B, L = batch_params
        block = SparseMoEBlock(tiny_config).to(device)
        x = torch.randn(B, L, tiny_config.d_model, device=device)  # (B, L, D)
        out, aux_loss = block(x)  # (B, L, D), scalar
        assert out.shape == (B, L, tiny_config.d_model)

    def test_aux_loss_scalar(
        self,
        tiny_config: AceConfig,
        device: torch.device,
        batch_params: tuple[int, int],
    ) -> None:
        """SparseMoEBlock returns a scalar aux_loss."""
        B, L = batch_params
        block = SparseMoEBlock(tiny_config).to(device)
        x = torch.randn(B, L, tiny_config.d_model, device=device)  # (B, L, D)
        _, aux_loss = block(x)  # scalar
        assert aux_loss.dim() == 0

    def test_residual_connection(
        self,
        tiny_config: AceConfig,
        device: torch.device,
        batch_params: tuple[int, int],
    ) -> None:
        """Residual connection: output differs from pure MoE FFN output.

        If we zero out all expert weights, the residual should ensure
        output ≈ input.
        """
        B, L = batch_params
        block = SparseMoEBlock(tiny_config).to(device)

        # Zero out all expert parameters so MoE output is ~zero
        with torch.no_grad():
            for expert in block.moe_ffn.experts:
                for param in expert.parameters():
                    param.zero_()

        x = torch.randn(B, L, tiny_config.d_model, device=device)  # (B, L, D)
        out, _ = block(x)  # (B, L, D)

        # With zeroed experts, output should be very close to input
        assert torch.allclose(
            out, x, atol=1e-5
        ), "With zeroed experts, residual should yield output ≈ input"

    def test_rejects_dimension_mismatch(
        self,
        tiny_config: AceConfig,
        device: torch.device,
        batch_params: tuple[int, int],
    ) -> None:
        """SparseMoEBlock raises ValueError for wrong input dimension."""
        B, L = batch_params
        block = SparseMoEBlock(tiny_config).to(device)
        bad_x = torch.randn(B, L, tiny_config.d_model * 2, device=device)  # (B, L, 2*D)
        with pytest.raises(ValueError, match="Expected last dimension"):
            block(bad_x)

    def test_gradient_flow(
        self,
        tiny_config: AceConfig,
        device: torch.device,
        batch_params: tuple[int, int],
    ) -> None:
        """Gradients flow through SparseMoEBlock to input."""
        B, L = batch_params
        block = SparseMoEBlock(tiny_config).to(device)
        x = torch.randn(B, L, tiny_config.d_model, device=device, requires_grad=True)  # (B, L, D)
        out, aux_loss = block(x)  # (B, L, D), scalar
        (out.sum() + aux_loss).backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_no_nans(
        self,
        tiny_config: AceConfig,
        device: torch.device,
        batch_params: tuple[int, int],
    ) -> None:
        """SparseMoEBlock output is free of NaN values."""
        B, L = batch_params
        block = SparseMoEBlock(tiny_config).to(device)
        x = torch.randn(B, L, tiny_config.d_model, device=device)  # (B, L, D)
        out, aux_loss = block(x)  # (B, L, D), scalar
        assert not torch.isnan(out).any()
        assert not torch.isnan(aux_loss)

    def test_has_norm_layer(self, tiny_config: AceConfig) -> None:
        """SparseMoEBlock contains a pre-norm RMSNorm."""
        block = SparseMoEBlock(tiny_config)
        assert isinstance(block.norm, RMSNorm)
        assert block.norm.d_model == tiny_config.d_model

    def test_single_token_batch(self, tiny_config: AceConfig, device: torch.device) -> None:
        """SparseMoEBlock handles B=1, L=1 edge case."""
        block = SparseMoEBlock(tiny_config).to(device)
        x = torch.randn(1, 1, tiny_config.d_model, device=device)  # (1, 1, D)
        out, aux_loss = block(x)  # (1, 1, D), scalar
        assert out.shape == (1, 1, tiny_config.d_model)
        assert aux_loss.dim() == 0

    def test_different_configs(self, device: torch.device) -> None:
        """SparseMoEBlock works with different expert/top-k configs."""
        cfg_4_1 = AceConfig(
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            d_ff=128,
            n_layers=2,
            max_seq_len=64,
            vocab_size=128,
            num_experts=4,
            top_k_experts=1,
            mamba_layers=[],
            moe_layers=[0, 1],
            retnet_layers=[],
        )
        block = SparseMoEBlock(cfg_4_1).to(device)
        x = torch.randn(2, 8, 32, device=device)  # (B, L, D)
        out, aux_loss = block(x)  # (B, L, D), scalar
        assert out.shape == (2, 8, 32)
        assert aux_loss.dim() == 0
