"""Regression checks for causality, cached chunks and unsupported model settings."""

from dataclasses import replace

import pytest
import torch

from ace.model.attention import MultiHeadAttention
from ace.model.backbone import AceModel


def test_cached_chunk_matches_full_sequence(micro_config):
    torch.manual_seed(4)
    model = MultiHeadAttention(micro_config).eval()
    inputs = torch.randn(2, 9, micro_config.d_model)
    with torch.no_grad():
        full, _ = model(inputs)
        _, cache = model(inputs[:, :4], use_cache=True)
        chunk, _ = model(inputs[:, 4:], past_key_value=cache)
    torch.testing.assert_close(chunk, full[:, 4:], atol=1e-6, rtol=1e-5)


def test_config_counts_match_instantiated_model(micro_config):
    model = AceModel(micro_config)
    total, active = micro_config.parameter_counts()
    assert total == sum(p.numel() for p in model.parameters())
    assert 0 < active < total


@pytest.mark.parametrize("dtype", [torch.bool, torch.long, torch.float32])
def test_padding_mask_keeps_causality(micro_config, dtype):
    model = MultiHeadAttention(micro_config).eval()
    x = torch.randn(1, 6, micro_config.d_model)
    mask = (
        torch.zeros(1, 1, 1, 6) if dtype == torch.float32 else torch.ones(1, 1, 1, 6, dtype=dtype)
    )
    altered = x.clone()
    altered[:, 3:] += 100
    with torch.no_grad():
        before, _ = model(x, mask=mask)
        after, _ = model(altered, mask=mask)
    torch.testing.assert_close(before[:, :3], after[:, :3])


def test_hybrid_cache_fails_explicitly(micro_config):
    model = AceModel(replace(micro_config, mamba_layers=[1]))
    with pytest.raises(ValueError, match="recurrent-state"):
        model(torch.ones(1, 4, dtype=torch.long), use_cache=True)


@pytest.mark.parametrize(
    "change",
    [{"world_model_enabled": True}, {"transformer2_enabled": True}, {"retnet_layers": [0]}],
)
def test_unimplemented_features_do_not_silently_activate(micro_config, change):
    with pytest.raises(NotImplementedError):
        AceModel(replace(micro_config, **change))
