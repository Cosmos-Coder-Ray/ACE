"""Attention primitives for the ACE model.

Provides:
    - ``RotaryPositionalEmbedding`` — precomputed cos/sin cache with a
      static ``apply_rotary_emb`` helper.
    - ``MultiHeadAttention`` — multi-head attention with Grouped-Query
      Attention (GQA), optional flash-attention, KV-cache for autoregressive
      inference, and causal masking.

All tensor reshaping uses ``einops.rearrange``.  Every tensor line carries a
shape comment in the form ``# (B, H, L, D)`` or similar.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ace.model.config import AceConfig
from ace.utils.env_detect import HAS_FLASH_ATTN
from ace.utils.fallbacks import get_attention_fn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flash-attention implementation: real flash_attn or pure-PyTorch fallback
# ---------------------------------------------------------------------------
_flash_attn_fn = get_attention_fn()


# ═══════════════════════════════════════════════════════════════════════════
# Rotary Positional Embedding
# ═══════════════════════════════════════════════════════════════════════════


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) with a precomputed cos/sin cache.

    Attributes:
        head_dim: Dimension per attention head (must be even).
        max_seq_len: Maximum sequence length for precomputation.
        theta: Base frequency for the sinusoidal schedule.
    """

    def __init__(self, config: AceConfig) -> None:
        """Initialise RoPE and precompute the cos/sin cache.

        Args:
            config: Global model configuration.

        Raises:
            ValueError: If ``head_dim`` is odd.
        """
        super().__init__()

        head_dim: int = config.head_dim
        max_seq_len: int = config.max_seq_len
        theta: float = config.rope_theta

        if head_dim % 2 != 0:
            raise ValueError(
                f"RotaryPositionalEmbedding requires an even head_dim, " f"got {head_dim}"
            )

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Inverse-frequency vector — (head_dim // 2,)
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )  # (head_dim // 2,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Declare buffer types for mypy (populated by _build_cache)
        self.inv_freq: torch.Tensor
        self.cos_cached: torch.Tensor
        self.sin_cached: torch.Tensor

        # Precompute cache up to max_seq_len
        self._build_cache(max_seq_len)

    # ------------------------------------------------------------------

    def _build_cache(self, seq_len: int) -> None:
        """Build and register the cos/sin cache buffers.

        Args:
            seq_len: Length of the sequence to precompute.
        """
        t = torch.arange(seq_len, device=self.inv_freq.device).float()  # (L,)
        freqs = torch.outer(t, self.inv_freq)  # (L, head_dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (L, head_dim)

        # Register as non-persistent buffers so they follow .to(device/dtype)
        cos_cached = emb.cos()  # (L, head_dim)
        sin_cached = emb.sin()  # (L, head_dim)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    # ------------------------------------------------------------------

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin tensors for positions ``[0, seq_len)``.

        If *seq_len* exceeds the cache, the cache is rebuilt on the fly.

        Args:
            seq_len: Current sequence length.

        Returns:
            Tuple of ``(cos, sin)`` each with shape ``(seq_len, head_dim)``.
        """
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)

        cos = self.cos_cached[:seq_len]  # (L, head_dim)
        sin = self.sin_cached[:seq_len]  # (L, head_dim)
        return cos, sin

    # ------------------------------------------------------------------

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the last dimension.

        Args:
            x: Tensor of shape ``(B, H, L, D)``.

        Returns:
            Rotated tensor with the same shape.
        """
        x1 = x[..., : x.shape[-1] // 2]  # (B, H, L, D//2)
        x2 = x[..., x.shape[-1] // 2 :]  # (B, H, L, D//2)
        return torch.cat([-x2, x1], dim=-1)  # (B, H, L, D)

    # ------------------------------------------------------------------

    @staticmethod
    def apply_rotary_emb(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary embeddings to a query **or** key tensor.

        Args:
            x: Tensor of shape ``(B, H, L, D)`` — queries or keys.
            cos: Cosine cache of shape ``(L, D)`` or broadcastable.
            sin: Sine cache of shape ``(L, D)`` or broadcastable.

        Returns:
            Tensor with rotary embeddings applied, same shape as *x*.
        """
        # Broadcast cos/sin to (1, 1, L, D) for (B, H, L, D) input
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, L, D)
        sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, L, D)

        rotated = x * cos + RotaryPositionalEmbedding._rotate_half(x) * sin  # (B, H, L, D)
        return rotated


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Head Attention with Grouped-Query Attention (GQA)
# ═══════════════════════════════════════════════════════════════════════════


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with Grouped-Query Attention (GQA).

    Supports:
        - GQA when ``n_kv_heads < n_heads`` (key/value heads are shared
          across groups of query heads).
        - Flash-attention (``flash_attn``) when the package is installed;
          otherwise falls back to
          ``torch.nn.functional.scaled_dot_product_attention``.
        - KV-cache for efficient autoregressive generation.
        - Causal masking.

    Attributes:
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value heads.
        head_dim: Dimension per head.
        n_groups: Number of query heads sharing each KV head.
    """

    def __init__(self, config: AceConfig) -> None:
        """Initialise projections, RoPE, and output projection.

        Args:
            config: Global model configuration.

        Raises:
            ValueError: If ``n_heads`` is not divisible by ``n_kv_heads``.
        """
        super().__init__()

        self.n_heads: int = config.n_heads
        self.n_kv_heads: int = config.n_kv_heads
        self.head_dim: int = config.head_dim
        self.d_model: int = config.d_model
        self.n_groups: int = self.n_heads // self.n_kv_heads

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by " f"n_kv_heads ({self.n_kv_heads})"
            )

        # ── Linear projections ────────────────────────────────────────
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

        # ── RoPE ──────────────────────────────────────────────────────
        self.rope = RotaryPositionalEmbedding(config)

        # ── Dropout ───────────────────────────────────────────────────
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Compute multi-head attention with optional KV-cache.

        Args:
            x: Input tensor of shape ``(B, L, D)``.
            mask: Optional attention mask of shape ``(B, 1, L, L)`` or
                broadcastable.  ``True``/``1`` = attend, ``False``/``0`` = mask.
                If *None* and no KV-cache, a causal mask is generated.
            past_key_value: Optional cached ``(K, V)`` each of shape
                ``(B, n_kv_heads, L_past, head_dim)``.
            use_cache: If *True*, return updated ``(K, V)`` cache.

        Returns:
            Tuple of ``(output, past_key_value)`` where *output* has shape
            ``(B, L, D)`` and *past_key_value* is ``None`` unless
            ``use_cache=True``.

        Raises:
            RuntimeError: If tensor shapes are inconsistent.
        """
        B, L, D = x.shape  # (B, L, D)

        # ── Project Q, K, V ───────────────────────────────────────────
        q = self.q_proj(x)  # (B, L, n_heads * head_dim)
        k = self.k_proj(x)  # (B, L, n_kv_heads * head_dim)
        v = self.v_proj(x)  # (B, L, n_kv_heads * head_dim)

        # Reshape to multi-head form using einops
        q = rearrange(
            q, "b l (h d) -> b h l d", h=self.n_heads, d=self.head_dim
        )  # (B, n_heads, L, head_dim)
        k = rearrange(
            k, "b l (h d) -> b h l d", h=self.n_kv_heads, d=self.head_dim
        )  # (B, n_kv_heads, L, head_dim)
        v = rearrange(
            v, "b l (h d) -> b h l d", h=self.n_kv_heads, d=self.head_dim
        )  # (B, n_kv_heads, L, head_dim)

        # ── Apply RoPE ────────────────────────────────────────────────
        # Determine position offset for KV-cache continuation
        past_len = past_key_value[0].shape[2] if past_key_value is not None else 0
        total_len = past_len + L

        cos, sin = self.rope(total_len)  # (total_len, head_dim) each

        # Slice cos/sin to only the new positions
        cos_q = cos[past_len:total_len]  # (L, head_dim)
        sin_q = sin[past_len:total_len]  # (L, head_dim)

        q = RotaryPositionalEmbedding.apply_rotary_emb(q, cos_q, sin_q)  # (B, n_heads, L, head_dim)
        k = RotaryPositionalEmbedding.apply_rotary_emb(
            k, cos_q, sin_q
        )  # (B, n_kv_heads, L, head_dim)

        # ── KV-cache: concatenate past keys/values ────────────────────
        if past_key_value is not None:
            past_k, past_v = past_key_value  # each (B, n_kv_heads, L_past, head_dim)
            k = torch.cat([past_k, k], dim=2)  # (B, n_kv_heads, total_len, head_dim)
            v = torch.cat([past_v, v], dim=2)  # (B, n_kv_heads, total_len, head_dim)

        new_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        if use_cache:
            new_cache = (k, v)

        # ── GQA: expand KV heads to match query heads ─────────────────
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)  # (B, n_heads, total_len, head_dim)
            v = v.repeat_interleave(self.n_groups, dim=1)  # (B, n_heads, total_len, head_dim)

        # Current shapes: q (B, n_heads, L, head_dim)
        #                  k (B, n_heads, total_len, head_dim)
        #                  v (B, n_heads, total_len, head_dim)

        kv_len = k.shape[2]  # total_len

        # ── Attention computation ─────────────────────────────────────
        attn_output = self._attention(
            q, k, v, mask=mask, B=B, L=L, kv_len=kv_len
        )  # (B, n_heads, L, head_dim)

        # ── Merge heads and project out ───────────────────────────────
        attn_output = rearrange(attn_output, "b h l d -> b l (h d)")  # (B, L, n_heads * head_dim)

        output = self.o_proj(attn_output)  # (B, L, D)
        output = self.resid_dropout(output)  # (B, L, D)

        return output, new_cache

    # ------------------------------------------------------------------

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        B: int,
        L: int,
        kv_len: int,
    ) -> torch.Tensor:
        """Dispatch to flash-attention or SDPA fallback.

        Args:
            q: Queries ``(B, n_heads, L, head_dim)``.
            k: Keys ``(B, n_heads, kv_len, head_dim)``.
            v: Values ``(B, n_heads, kv_len, head_dim)``.
            mask: Optional mask ``(B, 1, L, kv_len)`` or *None*.
            B: Batch size.
            L: Query sequence length.
            kv_len: Key/value sequence length (may include cache).

        Returns:
            Attention output ``(B, n_heads, L, head_dim)``.
        """
        is_causal = mask is None and kv_len == L  # safe to use built-in causal

        # ── Try flash-attention first (real or fallback) ─────────────────
        # Real flash_attn only works on CUDA with no custom mask and causal.
        # Fallback always works but uses SDPA internally.
        use_flash: bool = (
            HAS_FLASH_ATTN
            and mask is None
            and is_causal
            and q.is_cuda
        )

        if use_flash:
            # flash_attn_func expects (B, L, H, D) layout
            q_fa = rearrange(q, "b h l d -> b l h d")  # (B, L, n_heads, head_dim)
            k_fa = rearrange(k, "b h l d -> b l h d")  # (B, kv_len, n_heads, head_dim)
            v_fa = rearrange(v, "b h l d -> b l h d")  # (B, kv_len, n_heads, head_dim)

            out = _flash_attn_fn(
                q_fa,
                k_fa,
                v_fa,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                causal=True,
            )  # (B, L, n_heads, head_dim)

            result: torch.Tensor = rearrange(out, "b l h d -> b h l d")  # (B, n_heads, L, head_dim)
            return result

        # ── Fallback: F.scaled_dot_product_attention ───────────────────
        # Build causal mask if needed
        attn_mask: Optional[torch.Tensor] = None
        if mask is not None:
            # User-supplied mask: convert bool -> additive float mask
            # True = attend -> 0.0 ;  False = mask -> -inf
            if mask.dtype == torch.bool:
                attn_mask = torch.zeros_like(mask, dtype=q.dtype)  # (B, 1, L, kv_len)
                attn_mask = attn_mask.masked_fill(~mask, float("-inf"))  # (B, 1, L, kv_len)
            else:
                attn_mask = mask  # assume already additive

        out = F.scaled_dot_product_attention(
            q,  # (B, n_heads, L, head_dim)
            k,  # (B, n_heads, kv_len, head_dim)
            v,  # (B, n_heads, kv_len, head_dim)
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )  # (B, n_heads, L, head_dim)

        return out  # (B, n_heads, L, head_dim)


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("attention.py — smoke test")
    print("=" * 60)

    # Use a small config for the smoke test
    cfg = AceConfig(
        d_model=128,
        n_heads=8,
        n_kv_heads=4,
        d_ff=512,
        n_layers=2,
        max_seq_len=256,
        vocab_size=256,
        mamba_layers=[],
        moe_layers=[],
        retnet_layers=[],
        dropout=0.0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # ── RoPE test ─────────────────────────────────────────────────────
    rope = RotaryPositionalEmbedding(cfg).to(device)
    cos, sin = rope(32)
    assert cos.shape == (32, cfg.head_dim), f"cos shape: {cos.shape}"
    assert sin.shape == (32, cfg.head_dim), f"sin shape: {sin.shape}"

    x_rope = torch.randn(2, 8, 32, cfg.head_dim, device=device, dtype=dtype)
    x_rot = RotaryPositionalEmbedding.apply_rotary_emb(x_rope, cos, sin)
    assert x_rot.shape == x_rope.shape, f"rotated shape: {x_rot.shape}"
    assert not torch.isnan(x_rot).any(), "NaN in rotary output"
    print("[PASS] RotaryPositionalEmbedding")

    # ── MHA (no cache) ────────────────────────────────────────────────
    mha = MultiHeadAttention(cfg).to(device).to(dtype)
    x = torch.randn(2, 32, cfg.d_model, device=device, dtype=dtype)  # (B, L, D)
    out, cache = mha(x)
    assert out.shape == (2, 32, cfg.d_model), f"output shape: {out.shape}"
    assert cache is None
    assert not torch.isnan(out).any(), "NaN in MHA output"
    print("[PASS] MultiHeadAttention (no cache)")

    # ── MHA (with cache) ──────────────────────────────────────────────
    out1, kv1 = mha(x, use_cache=True)
    assert kv1 is not None
    assert kv1[0].shape[2] == 32  # L cached

    x_next = torch.randn(2, 1, cfg.d_model, device=device, dtype=dtype)
    out2, kv2 = mha(x_next, past_key_value=kv1, use_cache=True)
    assert out2.shape == (2, 1, cfg.d_model), f"cached output shape: {out2.shape}"
    assert kv2 is not None
    assert kv2[0].shape[2] == 33  # 32 + 1
    assert not torch.isnan(out2).any(), "NaN in cached MHA output"
    print("[PASS] MultiHeadAttention (KV-cache)")

    # ── GQA check ─────────────────────────────────────────────────────
    assert cfg.n_kv_heads < cfg.n_heads, "GQA requires n_kv_heads < n_heads"
    assert mha.n_groups == cfg.n_heads // cfg.n_kv_heads
    print(f"[PASS] GQA groups = {mha.n_groups}")

    print("=" * 60)
    print("[ALL PASS] attention.py smoke test complete")
    print("=" * 60)
