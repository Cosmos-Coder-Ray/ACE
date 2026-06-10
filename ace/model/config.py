"""AceConfig — central configuration dataclass for the ACE model.

Every hyper-parameter lives here.  No magic numbers in model code.

Usage:
    config = AceConfig()                       # defaults (~1.5B)
    config = AceConfig.from_yaml("ace.yaml")   # load from file
    config.to_yaml("snapshot.yaml")            # persist
    print(config.num_parameters)               # property
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


@dataclass
class AceConfig:
    """Master configuration for the ACE model.

    All architectural, training, and runtime parameters are defined here.
    Model files must **never** contain hard-coded constants -- they pull
    every value from an ``AceConfig`` instance.

    Attributes:
        d_model: Hidden dimension throughout the backbone.
        n_layers: Total number of backbone layers.
        n_heads: Number of attention heads (for standard attention layers).
        n_kv_heads: Number of key-value heads (GQA). Must divide n_heads.
        d_ff: Feed-forward intermediate dimension.
        vocab_size: BPE tokenizer vocabulary size.
        max_seq_len: Maximum sequence length (Mamba handles long ctx).
        num_experts: Total MoE experts per MoE layer.
        top_k_experts: Number of experts activated per token.
        mamba_layers: Layer indices that use Mamba SSM blocks.
        moe_layers: Layer indices that use Mixture-of-Experts FFN.
        retnet_layers: Layer indices that use RetNet blocks.
        transformer2_enabled: Whether Transformer-Squared adaptation is active.
        dropout: Global dropout rate.
        norm_eps: Layer-norm / RMSNorm epsilon.
        rope_theta: RoPE base frequency.
        world_model_enabled: Enable the world-model planning module.
        world_model_d_state: State dimension for the world model.
        world_model_n_steps: Planning horizon for the world model.
        tie_embeddings: Tie input embedding and LM head weights.
        use_gradient_checkpointing: Enable gradient checkpointing to save memory.
        dtype: Default torch dtype string (e.g. ``"bfloat16"``).
        seed: Global random seed.
    """

    # ── Core dimensions ───────────────────────────────────────────────
    d_model: int = 2048
    n_layers: int = 16
    n_heads: int = 16
    n_kv_heads: int = 8
    d_ff: int = 8192
    vocab_size: int = 65536
    max_seq_len: int = 32768

    # ── Mixture of Experts ────────────────────────────────────────────
    num_experts: int = 8
    top_k_experts: int = 2

    # ── Layer routing ─────────────────────────────────────────────────
    mamba_layers: list[int] = field(default_factory=lambda: [4, 5, 6, 7])
    moe_layers: list[int] = field(default_factory=lambda: list(range(4, 16)))
    retnet_layers: list[int] = field(default_factory=lambda: [8, 9, 10, 11])

    # ── Transformer² ──────────────────────────────────────────────────
    transformer2_enabled: bool = True

    # ── Regularisation ────────────────────────────────────────────────
    dropout: float = 0.0
    norm_eps: float = 1e-5

    # ── Positional encoding ───────────────────────────────────────────
    rope_theta: float = 500000.0

    # ── World model ───────────────────────────────────────────────────
    world_model_enabled: bool = True
    world_model_d_state: int = 512
    world_model_n_steps: int = 8

    # ── Weight tying & checkpointing ─────────────────────────────────
    tie_embeddings: bool = True
    use_gradient_checkpointing: bool = False

    # ── Runtime ───────────────────────────────────────────────────────
    dtype: str = "bfloat16"
    seed: int = 42

    # ── Derived properties ────────────────────────────────────────────

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head: ``d_model // n_heads``."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by " f"n_heads ({self.n_heads})"
            )
        return self.d_model // self.n_heads

    @property
    def num_parameters(self) -> int:
        """Rough estimate of **total** parameters (not active).

        Counts:
            - Embedding + output projection
            - Per-layer: attention + FFN (or MoE FFN)
            - World model (if enabled)

        Returns:
            Estimated total parameter count.
        """
        # Embedding (tied input/output)
        embed_params = self.vocab_size * self.d_model

        total_layer_params = 0
        for i in range(self.n_layers):
            # Attention projection: Q, K, V, O
            q_params = self.d_model * self.d_model  # Q
            kv_params = 2 * self.d_model * (self.n_kv_heads * self.head_dim)  # K, V
            o_params = self.d_model * self.d_model  # O
            attn_params = q_params + kv_params + o_params

            # FFN
            if i in self.moe_layers:
                # Each expert has an FFN, plus the router
                single_ffn = 3 * self.d_model * self.d_ff  # gate + up + down (SwiGLU)
                ffn_params = self.num_experts * single_ffn
                ffn_params += self.d_model * self.num_experts  # router
            else:
                ffn_params = 3 * self.d_model * self.d_ff  # SwiGLU

            # Mamba layers replace attention with SSM params — similar magnitude
            if i in self.mamba_layers:
                # Mamba: ~4 * d_model * d_model for in/out projections + SSM
                attn_params = 4 * self.d_model * self.d_model

            # Layer norms (small, but include)
            norm_params = 2 * self.d_model  # pre-attn + pre-ffn

            total_layer_params += attn_params + ffn_params + norm_params

        # World model
        wm_params = 0
        if self.world_model_enabled:
            wm_params = (
                self.d_model * self.world_model_d_state * 4  # state transitions
                + self.world_model_d_state * self.world_model_d_state * 2  # internal
            )

        # Final layer norm
        final_norm = self.d_model

        return embed_params + total_layer_params + wm_params + final_norm

    @property
    def num_active_parameters(self) -> int:
        """Estimate of parameters **active per token** (MoE top-k only).

        Returns:
            Estimated active parameter count.
        """
        embed_params = self.vocab_size * self.d_model
        total_layer_params = 0

        for i in range(self.n_layers):
            q_params = self.d_model * self.d_model
            kv_params = 2 * self.d_model * (self.n_kv_heads * self.head_dim)
            o_params = self.d_model * self.d_model
            attn_params = q_params + kv_params + o_params

            if i in self.moe_layers:
                single_ffn = 3 * self.d_model * self.d_ff
                ffn_params = self.top_k_experts * single_ffn  # only top-k active
                ffn_params += self.d_model * self.num_experts  # router always active
            else:
                ffn_params = 3 * self.d_model * self.d_ff

            if i in self.mamba_layers:
                attn_params = 4 * self.d_model * self.d_model

            norm_params = 2 * self.d_model
            total_layer_params += attn_params + ffn_params + norm_params

        wm_params = 0
        if self.world_model_enabled:
            wm_params = (
                self.d_model * self.world_model_d_state * 4
                + self.world_model_d_state * self.world_model_d_state * 2
            )

        return embed_params + total_layer_params + wm_params + self.d_model

    # ── Serialisation ─────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AceConfig":
        """Load config from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            An ``AceConfig`` instance populated from the file.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the YAML contains unknown fields.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with filepath.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        # Validate keys
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(raw.keys()) - valid_keys
        if unknown:
            raise ValueError(
                f"Unknown config keys: {unknown}. " f"Valid keys: {sorted(valid_keys)}"
            )

        return cls(**raw)

    def to_yaml(self, path: str | Path) -> Path:
        """Persist config to a YAML file.

        Args:
            path: Destination file path.

        Returns:
            The resolved :class:`pathlib.Path` that was written.
        """
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with filepath.open("w", encoding="utf-8") as f:
            yaml.dump(
                asdict(self),
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
        return filepath

    def __post_init__(self) -> None:
        """Validate invariants after initialisation."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        if self.top_k_experts > self.num_experts:
            raise ValueError(
                f"top_k_experts ({self.top_k_experts}) must be <= "
                f"num_experts ({self.num_experts})"
            )
        for idx in self.mamba_layers:
            if idx >= self.n_layers:
                raise ValueError(f"mamba_layer index {idx} >= n_layers ({self.n_layers})")
        for idx in self.moe_layers:
            if idx >= self.n_layers:
                raise ValueError(f"moe_layer index {idx} >= n_layers ({self.n_layers})")
        for idx in self.retnet_layers:
            if idx >= self.n_layers:
                raise ValueError(f"retnet_layer index {idx} >= n_layers ({self.n_layers})")

    def summary(self) -> str:
        """Return a human-readable summary string.

        Returns:
            Multi-line summary with key metrics.
        """
        total = self.num_parameters
        active = self.num_active_parameters
        sep = "-" * 50
        return (
            f"ACE Config Summary\n"
            f"{sep}\n"
            f"  d_model        : {self.d_model}\n"
            f"  n_layers       : {self.n_layers}\n"
            f"  n_heads        : {self.n_heads} (kv: {self.n_kv_heads})\n"
            f"  d_ff           : {self.d_ff}\n"
            f"  vocab_size     : {self.vocab_size:,}\n"
            f"  max_seq_len    : {self.max_seq_len:,}\n"
            f"  MoE            : {self.num_experts} experts, top-{self.top_k_experts}\n"
            f"  Mamba layers   : {self.mamba_layers}\n"
            f"  MoE layers     : {self.moe_layers}\n"
            f"  RetNet layers  : {self.retnet_layers}\n"
            f"  World model    : {'ON' if self.world_model_enabled else 'OFF'}\n"
            f"  Total params   : {total:,} ({total / 1e9:.2f}B)\n"
            f"  Active params  : {active:,} ({active / 1e9:.2f}B)\n"
            f"{sep}"
        )


# ──────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    # Test defaults
    cfg = AceConfig()
    print(cfg.summary())
    assert cfg.head_dim == cfg.d_model // cfg.n_heads
    assert cfg.num_parameters > 0
    assert cfg.num_active_parameters < cfg.num_parameters

    # Test YAML round-trip
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test_config.yaml"
        cfg.to_yaml(p)
        cfg2 = AceConfig.from_yaml(p)
        assert cfg == cfg2, "YAML round-trip mismatch"

    # Test validation
    try:
        AceConfig(d_model=100, n_heads=3)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass

    print("[PASS] config.py smoke test passed")
