"""Tests for ace.model.backbone — AceBlock, AceModel, SwiGLUFFN.

Covers:
    - Forward output shapes
    - AceModelOutput fields
    - Auxiliary loss is non-zero when MoE layers present
    - Gradient flow through the full model
    - KV-cache consistency (single-step decode matches)
    - Weight tying (tied vs untied)
    - Parameter count sanity
    - from_config classmethod
    - Gradient checkpointing
    - Edge cases and validation
"""

from __future__ import annotations

import pytest
import torch

from ace.model.backbone import (
    AceBlock,
    AceModel,
    AceModelOutput,
    SwiGLUFFN,
)
from ace.model.config import AceConfig

# ═══════════════════════════════════════════════════════════════════════════
# SwiGLUFFN
# ═══════════════════════════════════════════════════════════════════════════


class TestSwiGLUFFN:
    """Tests for the dense SwiGLU feed-forward network."""

    def test_forward_shape(self, micro_config: AceConfig, device: torch.device) -> None:
        """Output shape must match input shape ``(B, L, D)``."""
        ffn = SwiGLUFFN(micro_config).to(device)
        x = torch.randn(2, 16, micro_config.d_model, device=device)
        out = ffn(x)
        assert out.shape == (2, 16, micro_config.d_model)

    def test_no_nan(self, micro_config: AceConfig, device: torch.device) -> None:
        """Output must be free of NaN values."""
        ffn = SwiGLUFFN(micro_config).to(device)
        x = torch.randn(2, 16, micro_config.d_model, device=device)
        out = ffn(x)
        assert not torch.isnan(out).any()

    def test_gradient_flow(self, micro_config: AceConfig, device: torch.device) -> None:
        """Gradients must flow back to the input."""
        ffn = SwiGLUFFN(micro_config).to(device)
        x = torch.randn(2, 16, micro_config.d_model, device=device, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ═══════════════════════════════════════════════════════════════════════════
# AceBlock
# ═══════════════════════════════════════════════════════════════════════════


class TestAceBlock:
    """Tests for individual AceBlock (attention/mamba + ffn/moe)."""

    def test_attention_dense_forward(self, micro_config: AceConfig, device: torch.device) -> None:
        """Attention + dense FFN block: correct output shape and zero aux_loss."""
        block = AceBlock(micro_config, layer_idx=0).to(device)
        x = torch.randn(2, 16, micro_config.d_model, device=device)
        h, aux, kv = block(x)

        assert h.shape == (2, 16, micro_config.d_model)
        assert aux.item() == 0.0, "Non-MoE block should have zero aux_loss"
        assert kv is None, "No cache without use_cache=True"

    def test_attention_moe_forward(self, micro_config: AceConfig, device: torch.device) -> None:
        """Attention + MoE FFN block: correct shape and non-zero aux_loss."""
        # layer_idx=2 is in moe_layers=[2, 3]
        block = AceBlock(micro_config, layer_idx=2).to(device)
        x = torch.randn(2, 16, micro_config.d_model, device=device)
        h, aux, kv = block(x)

        assert h.shape == (2, 16, micro_config.d_model)
        assert aux.item() > 0.0, "MoE block should have non-zero aux_loss"

    def test_kv_cache(self, micro_config: AceConfig, device: torch.device) -> None:
        """KV cache is returned when ``use_cache=True`` and grows correctly."""
        block = AceBlock(micro_config, layer_idx=0).to(device)
        B, L, D = 2, 16, micro_config.d_model

        x = torch.randn(B, L, D, device=device)
        h1, _, kv1 = block(x, use_cache=True)

        assert kv1 is not None
        assert kv1[0].shape[2] == L  # cached L keys

        # Single-step decode
        x_next = torch.randn(B, 1, D, device=device)
        h2, _, kv2 = block(x_next, past_key_value=kv1, use_cache=True)

        assert h2.shape == (B, 1, D)
        assert kv2 is not None
        assert kv2[0].shape[2] == L + 1  # L + 1 cached keys

    def test_invalid_layer_idx(self, micro_config: AceConfig) -> None:
        """AceBlock rejects out-of-range layer indices."""
        with pytest.raises(ValueError, match="layer_idx"):
            AceBlock(micro_config, layer_idx=-1)

        with pytest.raises(ValueError, match="layer_idx"):
            AceBlock(micro_config, layer_idx=micro_config.n_layers)

    def test_residual_connection(self, micro_config: AceConfig, device: torch.device) -> None:
        """Output should differ from input (residual + non-zero transform)."""
        block = AceBlock(micro_config, layer_idx=0).to(device)
        x = torch.randn(2, 16, micro_config.d_model, device=device)
        h, _, _ = block(x)
        # The block applies norm+attn+norm+ffn, so output ≠ input
        assert not torch.allclose(h, x, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# AceModel
# ═══════════════════════════════════════════════════════════════════════════


class TestAceModelForward:
    """Tests for AceModel forward pass shapes and outputs."""

    def test_forward_shape(self, micro_config: AceConfig, device: torch.device) -> None:
        """Logits and hidden states have correct shapes."""
        model = AceModel(micro_config).to(device)
        B, L = 2, 16
        ids = torch.randint(0, micro_config.vocab_size, (B, L), device=device)

        out = model(ids)

        assert isinstance(out, AceModelOutput)
        assert out.logits.shape == (B, L, micro_config.vocab_size)
        assert out.hidden_states.shape == (B, L, micro_config.d_model)

    def test_return_dict_false(self, micro_config: AceConfig, device: torch.device) -> None:
        """``return_dict=False`` yields a 4-tuple."""
        model = AceModel(micro_config).to(device)
        ids = torch.randint(0, micro_config.vocab_size, (2, 16), device=device)

        result = model(ids, return_dict=False)

        assert isinstance(result, tuple)
        assert len(result) == 4
        logits, past_kvs, hidden, aux = result
        assert logits.shape == (2, 16, micro_config.vocab_size)
        assert past_kvs is None  # use_cache=False
        assert hidden.shape == (2, 16, micro_config.d_model)

    def test_no_nan_in_logits(self, micro_config: AceConfig, device: torch.device) -> None:
        """Logits must be NaN-free."""
        model = AceModel(micro_config).to(device)
        ids = torch.randint(0, micro_config.vocab_size, (2, 16), device=device)
        out = model(ids)
        assert not torch.isnan(out.logits).any()


class TestAuxLoss:
    """Tests for MoE auxiliary load-balancing loss."""

    def test_aux_loss_nonzero(self, micro_config: AceConfig, device: torch.device) -> None:
        """Total aux_loss should be > 0 when MoE layers exist."""
        model = AceModel(micro_config).to(device)
        ids = torch.randint(0, micro_config.vocab_size, (2, 16), device=device)
        out = model(ids)
        assert out.total_aux_loss.item() > 0.0

    def test_aux_loss_zero_no_moe(self, device: torch.device) -> None:
        """Total aux_loss should be 0 when no MoE layers exist."""
        cfg = AceConfig(
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            d_ff=256,
            n_layers=2,
            max_seq_len=128,
            vocab_size=256,
            mamba_layers=[],
            moe_layers=[],  # no MoE
            retnet_layers=[],
            tie_embeddings=True,
        )
        model = AceModel(cfg).to(device)
        ids = torch.randint(0, cfg.vocab_size, (2, 8), device=device)
        out = model(ids)
        assert out.total_aux_loss.item() == 0.0

    def test_aux_loss_scalar(self, micro_config: AceConfig, device: torch.device) -> None:
        """Auxiliary loss must be a scalar (0-dim tensor)."""
        model = AceModel(micro_config).to(device)
        ids = torch.randint(0, micro_config.vocab_size, (2, 16), device=device)
        out = model(ids)
        assert out.total_aux_loss.dim() == 0


class TestGradientFlow:
    """Tests that gradients propagate correctly through AceModel."""

    def test_gradient_to_embeddings(self, micro_config: AceConfig, device: torch.device) -> None:
        """Gradients must reach the embedding table."""
        model = AceModel(micro_config).to(device)
        model.zero_grad()
        ids = torch.randint(0, micro_config.vocab_size, (2, 16), device=device)

        out = model(ids)
        loss = out.logits.sum() + out.total_aux_loss
        loss.backward()

        assert model.embed_tokens.weight.grad is not None
        assert model.embed_tokens.weight.grad.abs().sum() > 0

    def test_gradient_to_all_layers(self, micro_config: AceConfig, device: torch.device) -> None:
        """Every backbone layer should receive gradients."""
        model = AceModel(micro_config).to(device)
        model.zero_grad()
        ids = torch.randint(0, micro_config.vocab_size, (2, 16), device=device)

        out = model(ids)
        loss = out.logits.sum() + out.total_aux_loss
        loss.backward()

        for i, layer in enumerate(model.layers):
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters()
            )
            assert has_grad, f"Layer {i} received no gradients"

    def test_aux_loss_gradient(self, micro_config: AceConfig, device: torch.device) -> None:
        """Gradients from aux_loss should reach the MoE router."""
        model = AceModel(micro_config).to(device)
        model.zero_grad()
        ids = torch.randint(0, micro_config.vocab_size, (2, 16), device=device)

        out = model(ids)
        # Only backprop through aux_loss, not logits
        out.total_aux_loss.backward()

        # MoE layers (2, 3) should have router gradients
        for idx in micro_config.moe_layers:
            router_gate = model.layers[idx].ffn.router.gate
            assert (
                router_gate.weight.grad is not None
            ), f"MoE router at layer {idx} received no aux_loss gradient"


class TestKVCache:
    """Tests for KV-cache consistency during autoregressive generation."""

    def test_cache_returned(self, micro_config: AceConfig, device: torch.device) -> None:
        """``use_cache=True`` should return a list of per-layer caches."""
        model = AceModel(micro_config).to(device)
        ids = torch.randint(0, micro_config.vocab_size, (2, 16), device=device)

        out = model(ids, use_cache=True)

        assert out.past_key_values is not None
        assert len(out.past_key_values) == micro_config.n_layers

    def test_cache_none_when_disabled(self, micro_config: AceConfig, device: torch.device) -> None:
        """``use_cache=False`` should return ``None`` for past_key_values."""
        model = AceModel(micro_config).to(device)
        ids = torch.randint(0, micro_config.vocab_size, (2, 8), device=device)
        out = model(ids, use_cache=False)
        assert out.past_key_values is None

    def test_single_step_decode_shape(self, micro_config: AceConfig, device: torch.device) -> None:
        """Decoding one token with cache produces ``(B, 1, vocab)`` logits."""
        model = AceModel(micro_config).to(device).eval()
        B, L = 2, 16
        ids = torch.randint(0, micro_config.vocab_size, (B, L), device=device)

        with torch.no_grad():
            out_full = model(ids, use_cache=True)

        next_token = torch.randint(0, micro_config.vocab_size, (B, 1), device=device)

        with torch.no_grad():
            out_step = model(
                next_token,
                past_key_values=out_full.past_key_values,
                use_cache=True,
            )

        assert out_step.logits.shape == (B, 1, micro_config.vocab_size)

    def test_cache_consistency(self, micro_config: AceConfig, device: torch.device) -> None:
        """Last-token logit from cached decode should match full recompute.

        Runs the full sequence through the model, then separately
        caches the prefix and decodes the final token.  The logits for
        the last position should be close.
        """
        model = AceModel(micro_config).to(device).eval()
        B, L = 1, 8
        torch.manual_seed(42)
        ids = torch.randint(0, micro_config.vocab_size, (B, L), device=device)

        with torch.no_grad():
            # Full pass
            out_full = model(ids)
            logits_full_last = out_full.logits[:, -1, :]  # (B, vocab)

            # Prefix + single-step decode
            prefix = ids[:, :-1]  # (B, L-1)
            out_prefix = model(prefix, use_cache=True)

            last_token = ids[:, -1:]  # (B, 1)
            out_step = model(
                last_token,
                past_key_values=out_prefix.past_key_values,
                use_cache=True,
            )
            logits_step_last = out_step.logits[:, 0, :]  # (B, vocab)

        assert torch.allclose(logits_full_last, logits_step_last, atol=1e-4), (
            f"Cache inconsistency: max diff = "
            f"{(logits_full_last - logits_step_last).abs().max().item():.6f}"
        )


class TestWeightTying:
    """Tests for embedding–LM-head weight tying."""

    def test_tied_weights(self, micro_config: AceConfig, device: torch.device) -> None:
        """With ``tie_embeddings=True``, lm_head.weight IS embed_tokens.weight."""
        model = AceModel(micro_config).to(device)
        assert model.lm_head.weight is model.embed_tokens.weight

    def test_untied_weights(self, micro_config_untied: AceConfig, device: torch.device) -> None:
        """With ``tie_embeddings=False``, weights are separate tensors."""
        model = AceModel(micro_config_untied).to(device)
        assert model.lm_head.weight is not model.embed_tokens.weight

    def test_tied_params_fewer(
        self,
        micro_config: AceConfig,
        micro_config_untied: AceConfig,
        device: torch.device,
    ) -> None:
        """Tied model should have fewer parameters than untied."""
        model_tied = AceModel(micro_config).to(device)
        model_untied = AceModel(micro_config_untied).to(device)

        # Tied model saves vocab_size * d_model parameters
        expected_diff = micro_config.vocab_size * micro_config.d_model
        actual_diff = model_untied.num_parameters - model_tied.num_parameters
        assert (
            actual_diff == expected_diff
        ), f"Expected param diff {expected_diff:,}, got {actual_diff:,}"


class TestParameterCount:
    """Tests for parameter counting properties."""

    def test_num_parameters_positive(self, micro_config: AceConfig, device: torch.device) -> None:
        """Model should have a positive parameter count."""
        model = AceModel(micro_config).to(device)
        assert model.num_parameters > 0

    def test_non_embedding_count(self, micro_config: AceConfig, device: torch.device) -> None:
        """Non-embedding parameters should be less than total."""
        model = AceModel(micro_config).to(device)
        assert model.num_parameters_non_embedding < model.num_parameters

    def test_micro_param_count_sanity(self, micro_config: AceConfig, device: torch.device) -> None:
        """Micro config should have parameters in a reasonable range.

        With d_model=64, d_ff=256, 4 layers, 4 MoE experts on 2 layers,
        we expect roughly ~1-3M parameters.
        """
        model = AceModel(micro_config).to(device)
        n_params = model.num_parameters
        assert (
            100_000 < n_params < 10_000_000
        ), f"Micro config has {n_params:,} params — outside expected range"


class TestFromConfig:
    """Tests for the ``from_config`` classmethod."""

    def test_from_config_returns_model(self, micro_config: AceConfig) -> None:
        """``from_config`` should return an AceModel instance."""
        model = AceModel.from_config(micro_config)
        assert isinstance(model, AceModel)

    def test_from_config_same_params(self, micro_config: AceConfig) -> None:
        """Model from ``from_config`` should match direct construction."""
        model1 = AceModel(micro_config)
        model2 = AceModel.from_config(micro_config)
        assert model1.num_parameters == model2.num_parameters


class TestGradientCheckpointing:
    """Tests for gradient checkpointing mode."""

    def test_checkpoint_forward(self, device: torch.device) -> None:
        """Model with gradient checkpointing should produce valid output."""
        cfg = AceConfig(
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            d_ff=256,
            n_layers=4,
            max_seq_len=128,
            vocab_size=256,
            mamba_layers=[],
            moe_layers=[2, 3],
            retnet_layers=[],
            tie_embeddings=True,
            use_gradient_checkpointing=True,
        )
        model = AceModel(cfg).to(device).train()
        ids = torch.randint(0, cfg.vocab_size, (2, 16), device=device)

        out = model(ids, use_cache=False)
        assert out.logits.shape == (2, 16, cfg.vocab_size)
        assert not torch.isnan(out.logits).any()

    def test_checkpoint_gradient_flow(self, device: torch.device) -> None:
        """Gradients should flow when gradient checkpointing is enabled."""
        cfg = AceConfig(
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            d_ff=256,
            n_layers=4,
            max_seq_len=128,
            vocab_size=256,
            mamba_layers=[],
            moe_layers=[2, 3],
            retnet_layers=[],
            tie_embeddings=True,
            use_gradient_checkpointing=True,
        )
        model = AceModel(cfg).to(device).train()
        model.zero_grad()
        ids = torch.randint(0, cfg.vocab_size, (2, 16), device=device)

        out = model(ids, use_cache=False)
        loss = out.logits.sum() + out.total_aux_loss
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad

    def test_checkpoint_rejects_cache(self, device: torch.device) -> None:
        """``use_cache=True`` should raise when checkpointing is active."""
        cfg = AceConfig(
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
            tie_embeddings=True,
            use_gradient_checkpointing=True,
        )
        model = AceModel(cfg).to(device).train()
        ids = torch.randint(0, cfg.vocab_size, (2, 8), device=device)

        with pytest.raises(ValueError, match="use_cache"):
            model(ids, use_cache=True)


class TestWeightInitialisation:
    """Tests for weight initialisation strategy."""

    def test_residual_projections_scaled(
        self, micro_config: AceConfig, device: torch.device
    ) -> None:
        """Residual projections should have smaller std than other weights.

        After ``initialize_weights``, the ``o_proj`` and ``down_proj``
        layers should have their standard deviation scaled down by
        ``1/sqrt(2*n_layers)`` relative to non-residual projections.
        """
        model = AceModel(micro_config).to(device)

        residual_stds: list[float] = []
        other_stds: list[float] = []

        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue
            std = param.data.std().item()
            if any(
                name.endswith(s) for s in ("o_proj.weight", "down_proj.weight", "out_proj.weight")
            ):
                residual_stds.append(std)
            elif "weight" in name and "norm" not in name:
                other_stds.append(std)

        if residual_stds and other_stds:
            avg_residual = sum(residual_stds) / len(residual_stds)
            avg_other = sum(other_stds) / len(other_stds)
            # Residual projections should be scaled down
            assert avg_residual < avg_other, (
                f"Residual std ({avg_residual:.4f}) should be < " f"other std ({avg_other:.4f})"
            )
