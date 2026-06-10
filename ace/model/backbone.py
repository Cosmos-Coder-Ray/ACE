"""ACE model backbone — assembles all components into the full AceModel.

Provides:
    - ``SwiGLUFFN`` — Standard SwiGLU feed-forward network for non-MoE layers.
    - ``AceBlock`` — Single transformer block with Mamba/Attention + MoE/FFN.
    - ``AceModelOutput`` — Typed output container.
    - ``AceModel`` — Full language model: Embed → N×AceBlock → Norm → LM Head.

Architecture per AceBlock::

    residual = x
    x = attn_norm(x)
    x = Mamba(x)  OR  MultiHeadAttention(x)
    x = residual + x

    residual = x
    x = ffn_norm(x)
    x, aux = MoEFeedForward(x)  OR  x = SwiGLUFFN(x)
    x = residual + x

The model supports:
    - **Mamba layers**: Layer indices in ``config.mamba_layers`` use the
      Mamba SSM instead of multi-head attention, providing O(L) sequence
      modelling for ACE's 32k context window.
    - **MoE layers**: Layer indices in ``config.moe_layers`` use sparse
      MoE feed-forward with top-k routing, yielding 1.5B total / 400M
      active parameters.
    - **Weight tying**: When ``config.tie_embeddings=True``, the LM head
      shares weights with the token embedding.
    - **Gradient checkpointing**: When ``config.use_gradient_checkpointing=True``
      and ``model.training``, backbone layers are wrapped in
      ``torch.utils.checkpoint`` to trade compute for memory.

All hyper-parameters come from :class:`AceConfig` — no magic numbers.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from ace.model.attention import MultiHeadAttention
from ace.model.config import AceConfig
from ace.model.mamba_block import Mamba, RMSNorm
from ace.model.moe import MoEFeedForward

logger = logging.getLogger(__name__)

# Parameter names whose weights sit on the residual stream and should
# receive scaled initialisation (GPT-2 style: 1/sqrt(2*n_layers)).
_RESIDUAL_PROJ_SUFFIXES: tuple[str, ...] = (
    "o_proj.weight",
    "down_proj.weight",
    "out_proj.weight",
)


# ═══════════════════════════════════════════════════════════════════════════
# AceModelOutput
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AceModelOutput:
    """Typed output container for :class:`AceModel` forward pass.

    Attributes:
        logits: Language-model logits of shape ``(B, L, vocab_size)``.
        past_key_values: Per-layer KV caches.  ``None`` for Mamba layers.
        hidden_states: Final hidden states *after* the final RMSNorm,
            shape ``(B, L, D)``.
        total_aux_loss: Summed MoE auxiliary load-balancing loss (scalar).
    """

    logits: torch.Tensor  # (B, L, vocab_size)
    past_key_values: Optional[list[Optional[tuple[torch.Tensor, torch.Tensor]]]]
    hidden_states: torch.Tensor  # (B, L, D)
    total_aux_loss: torch.Tensor  # scalar


# ═══════════════════════════════════════════════════════════════════════════
# SwiGLU Feed-Forward Network (dense, non-MoE)
# ═══════════════════════════════════════════════════════════════════════════


class SwiGLUFFN(nn.Module):
    """Standard SwiGLU feed-forward network for non-MoE layers.

    Architecture::

        gate = gate_proj(x)          # (*, d_ff)
        up   = up_proj(x)            # (*, d_ff)
        h    = SiLU(gate) * up       # (*, d_ff)   — SwiGLU activation
        h    = dropout(h)            # (*, d_ff)
        out  = down_proj(h)          # (*, D)

    Attributes:
        d_model: Input / output hidden dimension.
        d_ff: Intermediate (expansion) dimension.
    """

    def __init__(self, config: AceConfig) -> None:
        """Initialise SwiGLU FFN projections.

        Args:
            config: Global model configuration.  Uses ``d_model``,
                ``d_ff``, and ``dropout``.
        """
        super().__init__()

        self.d_model: int = config.d_model
        self.d_ff: int = config.d_ff

        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)  # (D) -> (d_ff)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)  # (D) -> (d_ff)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)  # (d_ff) -> (D)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU FFN.

        Args:
            x: Input tensor of shape ``(B, L, D)`` or ``(*, D)``.

        Returns:
            Output tensor with the same shape as *x*.
        """
        gate = self.gate_proj(x)  # (*, d_ff)
        up = self.up_proj(x)  # (*, d_ff)
        h = F.silu(gate) * up  # (*, d_ff) — SwiGLU activation
        h = self.dropout(h)  # (*, d_ff)
        out: torch.Tensor = self.down_proj(h)  # (*, D)
        return out  # (*, D)

    def extra_repr(self) -> str:
        """Show dimensions in repr."""
        return f"d_model={self.d_model}, d_ff={self.d_ff}"


# ═══════════════════════════════════════════════════════════════════════════
# AceBlock
# ═══════════════════════════════════════════════════════════════════════════


class AceBlock(nn.Module):
    """Single ACE backbone block.

    Architecture::

        # --- Attention / Mamba sub-layer ---
        residual = x                            # (B, L, D)
        x = attn_norm(x)                        # (B, L, D)
        x = Mamba(x)  OR  Attention(x)          # (B, L, D)
        x = residual + x                        # (B, L, D)

        # --- FFN / MoE sub-layer ---
        residual = x                            # (B, L, D)
        x = ffn_norm(x)                         # (B, L, D)
        x, aux = MoEFFN(x)  OR  x = FFN(x)     # (B, L, D)
        x = residual + x                        # (B, L, D)

    Layer routing:
        - ``layer_idx in config.mamba_layers`` → Mamba SSM (no KV cache).
        - Otherwise → :class:`MultiHeadAttention` (supports KV cache).
        - ``layer_idx in config.moe_layers`` → :class:`MoEFeedForward`.
        - Otherwise → :class:`SwiGLUFFN`.

    Attributes:
        layer_idx: Zero-based index of this block in the backbone.
        d_model: Hidden dimension.
        is_mamba: Whether this block uses a Mamba SSM.
        is_moe: Whether this block uses sparse MoE FFN.
    """

    def __init__(self, config: AceConfig, layer_idx: int) -> None:
        """Initialise a single AceBlock.

        Mamba layers use ``mamba_ssm.Mamba`` when available, otherwise
        fall back to :class:`~ace.utils.fallbacks.FallbackMamba`.

        Args:
            config: Global model configuration.
            layer_idx: Zero-based index of this layer in the backbone
                stack.  Controls whether Mamba or Attention is used,
                and whether MoE or dense FFN is used.

        Raises:
            ValueError: If *layer_idx* is negative or >= ``config.n_layers``.
        """
        super().__init__()

        if layer_idx < 0 or layer_idx >= config.n_layers:
            raise ValueError(f"layer_idx must be in [0, {config.n_layers}), " f"got {layer_idx}")

        self.layer_idx: int = layer_idx
        self.d_model: int = config.d_model
        self.is_mamba: bool = layer_idx in config.mamba_layers
        self.is_moe: bool = layer_idx in config.moe_layers

        # ── Attention / Mamba sub-layer ──────────────────────────────
        self.attn_norm = RMSNorm(d_model=config.d_model, eps=config.norm_eps)

        if self.is_mamba:
            self.attn: nn.Module = Mamba(
                d_model=config.d_model,
                d_state=16,
                d_conv=4,
                expand=2,
            )
        else:
            self.attn = MultiHeadAttention(config)

        # ── FFN / MoE sub-layer ───────────────────────────────────────
        self.ffn_norm = RMSNorm(d_model=config.d_model, eps=config.norm_eps)

        if self.is_moe:
            self.ffn: nn.Module = MoEFeedForward(config)
        else:
            self.ffn = SwiGLUFFN(config)

    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for a single AceBlock.

        Args:
            hidden_states: Input tensor of shape ``(B, L, D)``.
            attention_mask: Optional attention mask for non-Mamba layers.
                Shape ``(B, 1, L, L)`` or broadcastable.
            past_key_value: Optional KV cache from previous forward pass.
                Ignored for Mamba layers.
            use_cache: If *True*, return updated KV cache (attention
                layers only).

        Returns:
            Tuple of:
                - ``hidden_states``: Output tensor ``(B, L, D)``.
                - ``aux_loss``: Scalar MoE auxiliary loss (0 for non-MoE).
                - ``past_key_value``: Updated KV cache or ``None``.
        """
        # ── Attention / Mamba sub-layer ────────────────────────────────
        residual = hidden_states  # (B, L, D)
        hidden_states = self.attn_norm(hidden_states)  # (B, L, D)

        new_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None

        if self.is_mamba:
            hidden_states = self.attn(hidden_states)  # (B, L, D)
        else:
            hidden_states, new_cache = self.attn(
                hidden_states,
                mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )  # (B, L, D), optional cache

        hidden_states = residual + hidden_states  # (B, L, D)

        # ── FFN / MoE sub-layer ───────────────────────────────────────
        residual = hidden_states  # (B, L, D)
        hidden_states = self.ffn_norm(hidden_states)  # (B, L, D)

        aux_loss = torch.tensor(
            0.0, device=hidden_states.device, dtype=hidden_states.dtype
        )  # scalar

        if self.is_moe:
            hidden_states, aux_loss = self.ffn(hidden_states)  # (B, L, D), scalar
        else:
            hidden_states = self.ffn(hidden_states)  # (B, L, D)

        hidden_states = residual + hidden_states  # (B, L, D)

        return hidden_states, aux_loss, new_cache

    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        """Show layer configuration in repr."""
        return (
            f"layer_idx={self.layer_idx}, d_model={self.d_model}, "
            f"is_mamba={self.is_mamba}, is_moe={self.is_moe}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# AceModel
# ═══════════════════════════════════════════════════════════════════════════


class AceModel(nn.Module):
    """Full ACE language model.

    Architecture::

        input_ids  →  Embedding  →  Dropout
            →  N × AceBlock  →  RMSNorm  →  LM Head  →  logits

    Features:
        - **Hybrid backbone**: Mamba SSM + Multi-Head Attention blocks.
        - **Sparse MoE**: Conditional computation via top-k expert routing.
        - **Weight tying**: Embedding ↔ LM head when ``tie_embeddings=True``.
        - **Gradient checkpointing**: Memory-efficient training.
        - **Scaled initialisation**: Residual projections scaled by
          ``1 / sqrt(2 * n_layers)`` for stable training at init.

    Attributes:
        config: The :class:`AceConfig` driving all hyper-parameters.
        embed_tokens: Token embedding table.
        layers: ``nn.ModuleList`` of :class:`AceBlock` instances.
        norm: Final :class:`RMSNorm` before the LM head.
        lm_head: Linear projection to vocabulary logits.
    """

    def __init__(self, config: AceConfig) -> None:
        """Initialise the full AceModel from config.

        Mamba layers use ``mamba_ssm.Mamba`` when available, otherwise
        fall back to :class:`~ace.utils.fallbacks.FallbackMamba` so the
        model works on Windows CPU without GPU libraries.

        Args:
            config: Global model configuration.
        """
        super().__init__()

        self.config: AceConfig = config

        # ── Token embedding ───────────────────────────────────────────
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)  # (vocab_size, D)

        self.embed_dropout = nn.Dropout(config.dropout)

        # ── Backbone layers ───────────────────────────────────────────
        self.layers = nn.ModuleList([AceBlock(config, layer_idx=i) for i in range(config.n_layers)])

        # ── Final normalisation ───────────────────────────────────────
        self.norm = RMSNorm(d_model=config.d_model, eps=config.norm_eps)

        # ── LM head ──────────────────────────────────────────────────
        self.lm_head = nn.Linear(
            config.d_model, config.vocab_size, bias=False
        )  # (D) -> (vocab_size)

        # ── Weight tying ──────────────────────────────────────────────
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # ── Initialise weights ────────────────────────────────────────
        self.initialize_weights()

        logger.info(
            "AceModel initialised: %s total parameters " "(%s with weight tying)",
            f"{self.num_parameters:,}",
            "tied" if config.tie_embeddings else "untied",
        )

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        """Initialise a single sub-module.

        - ``nn.Linear``: Normal(0, 0.02), bias zeroed.
        - ``nn.Embedding``: Normal(0, 0.02).

        Mamba's internal SSM parameters (A_log, D, conv1d) are *not*
        ``nn.Linear`` or ``nn.Embedding``, so they retain their original
        initialisation from ``mamba_ssm``.

        Args:
            module: The sub-module to initialise.
        """
        std: float = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def initialize_weights(self) -> None:
        """Initialise all weights with scaled init for residual projections.

        Strategy:
            1. Apply ``Normal(0, 0.02)`` to all ``nn.Linear`` and
               ``nn.Embedding`` modules via :meth:`_init_weights`.
            2. Scale residual-stream projections (``o_proj``, ``down_proj``,
               ``out_proj``) by ``1 / sqrt(2 * n_layers)`` to keep the
               residual stream variance stable at initialisation (GPT-2
               style).
        """
        self.apply(self._init_weights)

        # Scale residual projections
        residual_scale: float = 1.0 / math.sqrt(2.0 * self.config.n_layers)

        for name, param in self.named_parameters():
            if any(name.endswith(suffix) for suffix in _RESIDUAL_PROJ_SUFFIXES):
                with torch.no_grad():
                    param.mul_(residual_scale)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[Optional[tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> (
        AceModelOutput
        | tuple[
            torch.Tensor,
            Optional[list[Optional[tuple[torch.Tensor, torch.Tensor]]]],
            torch.Tensor,
            torch.Tensor,
        ]
    ):
        """Run the full ACE language model forward pass.

        Args:
            input_ids: Token indices of shape ``(B, L)``.
            attention_mask: Optional attention mask of shape
                ``(B, 1, L, L)`` or broadcastable.  Passed through to
                attention layers; ignored by Mamba layers.
            past_key_values: Optional list of per-layer KV caches, one
                entry per backbone layer. ``None`` entries are valid
                (e.g. for Mamba layers).
            use_cache: If *True*, attention layers return updated KV
                caches.  Must be *False* when gradient checkpointing
                is enabled.
            return_dict: If *True* (default), return an
                :class:`AceModelOutput`.  Otherwise return a plain tuple.

        Returns:
            :class:`AceModelOutput` or a tuple of
            ``(logits, past_key_values, hidden_states, total_aux_loss)``.

        Raises:
            ValueError: If ``use_cache=True`` while gradient checkpointing
                is active during training.
        """
        if self.config.use_gradient_checkpointing and self.training and use_cache:
            raise ValueError(
                "use_cache=True is incompatible with gradient "
                "checkpointing during training. Set use_cache=False "
                "or disable gradient checkpointing."
            )

        B, L = input_ids.shape  # (B, L)

        # ── Token embeddings ──────────────────────────────────────────
        hidden_states = self.embed_tokens(input_ids)  # (B, L, D)
        hidden_states = self.embed_dropout(hidden_states)  # (B, L, D)

        # ── Backbone layers ───────────────────────────────────────────
        total_aux_loss = torch.tensor(
            0.0, device=hidden_states.device, dtype=hidden_states.dtype
        )  # scalar
        new_key_values: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = []

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            if self.config.use_gradient_checkpointing and self.training:
                hidden_states, aux_loss, new_kv = torch_checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    past_kv,
                    use_cache,
                    use_reentrant=False,
                )
            else:
                hidden_states, aux_loss, new_kv = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                )

            total_aux_loss = total_aux_loss + aux_loss  # scalar
            new_key_values.append(new_kv)

        # ── Final norm + LM head ──────────────────────────────────────
        hidden_states = self.norm(hidden_states)  # (B, L, D)
        logits = self.lm_head(hidden_states)  # (B, L, vocab_size)

        # ── Build output ──────────────────────────────────────────────
        output = AceModelOutput(
            logits=logits,
            past_key_values=new_key_values if use_cache else None,
            hidden_states=hidden_states,
            total_aux_loss=total_aux_loss,
        )

        if return_dict:
            return output

        return (
            output.logits,
            output.past_key_values,
            output.hidden_states,
            output.total_aux_loss,
        )

    # ------------------------------------------------------------------
    # Constructors & properties
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: AceConfig) -> "AceModel":
        """Construct an :class:`AceModel` from an :class:`AceConfig`.

        This is the preferred factory method — it makes the dependency
        on ``AceConfig`` explicit and keeps the constructor signature
        stable.

        Args:
            config: Global model configuration.

        Returns:
            A freshly initialised :class:`AceModel`.
        """
        return cls(config)

    @property
    def num_parameters(self) -> int:
        """Total number of learnable parameters (respects weight tying).

        Returns:
            Integer count of all ``requires_grad`` parameters.
        """
        return sum(p.numel() for p in self.parameters())

    @property
    def num_parameters_non_embedding(self) -> int:
        """Parameters excluding the token embedding table.

        Useful for reporting backbone-only parameter counts.

        Returns:
            Integer count of non-embedding parameters.
        """
        embed_params = self.embed_tokens.weight.numel()
        total = self.num_parameters
        # If tied, lm_head shares embed weight — already excluded
        if not self.config.tie_embeddings:
            return total - embed_params - self.lm_head.weight.numel()
        return total - embed_params

    def extra_repr(self) -> str:
        """Show key model metrics in repr."""
        return (
            f"n_layers={self.config.n_layers}, "
            f"d_model={self.config.d_model}, "
            f"vocab_size={self.config.vocab_size}, "
            f"params={self.num_parameters:,}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("backbone.py — smoke test")
    print("=" * 60)

    # ── Micro config (no Mamba — may not be installed) ────────────────
    cfg = AceConfig(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        d_ff=256,
        n_layers=4,
        max_seq_len=128,
        vocab_size=256,
        num_experts=4,
        top_k_experts=2,
        mamba_layers=[],  # skip Mamba for portability
        moe_layers=[2, 3],
        retnet_layers=[],
        dropout=0.0,
        tie_embeddings=True,
        use_gradient_checkpointing=False,
    )

    device = torch.device("cpu")
    dtype = torch.float32
    B, L = 2, 16

    # ── SwiGLUFFN ─────────────────────────────────────────────────────
    ffn = SwiGLUFFN(cfg).to(device).to(dtype)
    x_ffn = torch.randn(B, L, cfg.d_model, device=device, dtype=dtype)
    out_ffn = ffn(x_ffn)
    assert out_ffn.shape == (B, L, cfg.d_model), f"SwiGLUFFN shape: {out_ffn.shape}"
    assert not torch.isnan(out_ffn).any(), "NaN in SwiGLUFFN output"
    print("[PASS] SwiGLUFFN forward")

    # ── AceBlock (attention + dense FFN) ──────────────────────────────
    block_attn = AceBlock(cfg, layer_idx=0).to(device).to(dtype)
    x_block = torch.randn(B, L, cfg.d_model, device=device, dtype=dtype)
    h, aux, kv = block_attn(x_block)
    assert h.shape == (B, L, cfg.d_model), f"AceBlock attn shape: {h.shape}"
    assert aux.item() == 0.0, "Non-MoE block should have zero aux_loss"
    assert kv is None, "Cache should be None when use_cache=False"
    print("[PASS] AceBlock (attention + dense FFN)")

    # ── AceBlock (attention + MoE FFN) ────────────────────────────────
    block_moe = AceBlock(cfg, layer_idx=2).to(device).to(dtype)
    h_moe, aux_moe, _ = block_moe(x_block)
    assert h_moe.shape == (B, L, cfg.d_model), f"AceBlock MoE shape: {h_moe.shape}"
    assert aux_moe.item() > 0.0, "MoE block should have non-zero aux_loss"
    print(f"[PASS] AceBlock (attention + MoE), aux_loss={aux_moe.item():.6f}")

    # ── AceBlock with KV cache ────────────────────────────────────────
    h1, _, kv1 = block_attn(x_block, use_cache=True)
    assert kv1 is not None, "KV cache should be returned"
    assert kv1[0].shape[2] == L, f"Cache length: {kv1[0].shape[2]}"
    x_next = torch.randn(B, 1, cfg.d_model, device=device, dtype=dtype)
    h2, _, kv2 = block_attn(x_next, past_key_value=kv1, use_cache=True)
    assert h2.shape == (B, 1, cfg.d_model), f"Cached step shape: {h2.shape}"
    assert kv2 is not None and kv2[0].shape[2] == L + 1
    print("[PASS] AceBlock KV-cache")

    # ── Full AceModel ─────────────────────────────────────────────────
    model = AceModel(cfg).to(device).to(dtype)
    input_ids = torch.randint(0, cfg.vocab_size, (B, L), device=device)

    output = model(input_ids)
    assert isinstance(output, AceModelOutput)
    assert output.logits.shape == (B, L, cfg.vocab_size), f"Logits shape: {output.logits.shape}"
    assert output.hidden_states.shape == (
        B,
        L,
        cfg.d_model,
    ), f"Hidden shape: {output.hidden_states.shape}"
    assert (
        output.total_aux_loss.item() > 0.0
    ), "Total aux_loss should be non-zero (MoE layers present)"
    assert output.past_key_values is None, "past_key_values should be None when use_cache=False"
    assert not torch.isnan(output.logits).any(), "NaN in logits"
    print("[PASS] AceModel forward (return_dict=True)")

    # ── Tuple output ──────────────────────────────────────────────────
    tup = model(input_ids, return_dict=False)
    assert isinstance(tup, tuple) and len(tup) == 4
    print("[PASS] AceModel forward (return_dict=False)")

    # ── Weight tying ──────────────────────────────────────────────────
    assert (
        model.lm_head.weight is model.embed_tokens.weight
    ), "Weight tying failed: lm_head.weight should be embed_tokens.weight"
    print("[PASS] Weight tying")

    # ── KV cache through full model ───────────────────────────────────
    out_cache = model(input_ids, use_cache=True)
    assert out_cache.past_key_values is not None
    assert len(out_cache.past_key_values) == cfg.n_layers
    # Attention layers should have caches, MoE layer type doesn't matter
    input_next = torch.randint(0, cfg.vocab_size, (B, 1), device=device)
    out_step = model(
        input_next,
        past_key_values=out_cache.past_key_values,
        use_cache=True,
    )
    assert out_step.logits.shape == (B, 1, cfg.vocab_size)
    print("[PASS] AceModel KV-cache (full model)")

    # ── Gradient flow ─────────────────────────────────────────────────
    model.zero_grad()
    input_grad = torch.randint(0, cfg.vocab_size, (B, L), device=device)
    out_grad = model(input_grad)
    loss = out_grad.logits.sum() + out_grad.total_aux_loss
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "No gradients flowed through the model"
    print("[PASS] Gradient flow")

    # ── from_config classmethod ───────────────────────────────────────
    model2 = AceModel.from_config(cfg)
    assert isinstance(model2, AceModel)
    assert model2.num_parameters == model.num_parameters
    print("[PASS] from_config classmethod")

    # ── Parameter count ───────────────────────────────────────────────
    print(f"  Total parameters: {model.num_parameters:,}")
    print(f"  Non-embedding:    {model.num_parameters_non_embedding:,}")

    # ── layer_idx validation ──────────────────────────────────────────
    try:
        AceBlock(cfg, layer_idx=-1)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass

    try:
        AceBlock(cfg, layer_idx=cfg.n_layers)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
    print("[PASS] AceBlock rejects invalid layer_idx")

    print("=" * 60)
    print("[ALL PASS] backbone.py smoke test complete")
    print("=" * 60)
