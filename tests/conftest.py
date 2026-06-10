"""Shared pytest fixtures for ACE test suite.

Provides a ``micro_config`` fixture with a small AceConfig suitable for
fast CPU-only unit tests.  All test modules should use this fixture
instead of constructing their own configs to keep tests consistent.
"""

from __future__ import annotations

import pytest
import torch

from ace.model.config import AceConfig


@pytest.fixture
def micro_config() -> AceConfig:
    """A tiny AceConfig for fast CPU-only testing.

    Layout (4 layers):
        - Layer 0: Attention + dense SwiGLU FFN
        - Layer 1: Attention + dense SwiGLU FFN
        - Layer 2: Attention + MoE FFN
        - Layer 3: Attention + MoE FFN

    Mamba layers are excluded by default because ``mamba_ssm`` may not
    be installed.  Tests that need Mamba should create their own config
    with ``mamba_layers`` set and skip if ``mamba_ssm`` is unavailable.

    Returns:
        A minimal AceConfig instance.
    """
    return AceConfig(
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


@pytest.fixture
def micro_config_untied() -> AceConfig:
    """Micro config with ``tie_embeddings=False`` for weight-tying tests.

    Returns:
        An AceConfig with untied embedding/LM-head weights.
    """
    return AceConfig(
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
        tie_embeddings=False,
        use_gradient_checkpointing=False,
    )


@pytest.fixture
def device() -> torch.device:
    """Return CPU device for deterministic testing.

    Returns:
        ``torch.device("cpu")``.
    """
    return torch.device("cpu")
