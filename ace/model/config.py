"""AceConfig — central configuration dataclass for the ACE model.

Every hyper-parameter lives here.  No magic numbers in model code.

Usage:
    config = AceConfig()                       # small dense baseline
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
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2
    d_ff: int = 768
    vocab_size: int = 4096
    max_seq_len: int = 512

    # ── Mixture of Experts ────────────────────────────────────────────
    num_experts: int = 4
    top_k_experts: int = 2

    # ── Layer routing ─────────────────────────────────────────────────
    mamba_layers: list[int] = field(default_factory=list)
    moe_layers: list[int] = field(default_factory=list)
    retnet_layers: list[int] = field(default_factory=list)

    # ── Transformer² ──────────────────────────────────────────────────
    transformer2_enabled: bool = False

    # ── Regularisation ────────────────────────────────────────────────
    dropout: float = 0.0
    norm_eps: float = 1e-5

    # ── Positional encoding ───────────────────────────────────────────
    rope_theta: float = 500000.0

    # ── World model ───────────────────────────────────────────────────
    world_model_enabled: bool = False
    world_model_d_state: int = 512
    world_model_n_steps: int = 8

    # ── Weight tying & checkpointing ─────────────────────────────────
    tie_embeddings: bool = True
    use_gradient_checkpointing: bool = False

    # ── Runtime ───────────────────────────────────────────────────────
    dtype: str = "float32"
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

    def parameter_counts(self) -> tuple[int, int]:
        """Count actual instantiated parameters without allocating their storage.

        The selected backend determines Mamba's tensors. Active count subtracts
        unselected experts; it is not a FLOP or inference-memory measurement.
        """
        import torch

        from ace.model.backbone import AceModel
        from ace.model.moe import MoEFeedForward

        with torch.device("meta"):
            model = AceModel(self)
        total = model.num_parameters
        inactive = sum(
            parameter.numel()
            for layer in model.layers
            if isinstance(layer.ffn, MoEFeedForward)
            for expert in list(layer.ffn.experts)[self.top_k_experts :]
            for parameter in expert.parameters()
        )
        return total, total - inactive

    @property
    def num_parameters(self) -> int:
        """Actual total parameters for this configuration and installed backend."""
        return self.parameter_counts()[0]

    @property
    def num_active_parameters(self) -> int:
        """Approximate parameter participation per token, not compute cost."""
        return self.parameter_counts()[1]

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
        for name in (
            "d_model",
            "n_layers",
            "n_heads",
            "n_kv_heads",
            "d_ff",
            "vocab_size",
            "max_seq_len",
            "num_experts",
            "top_k_experts",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        for name in ("mamba_layers", "moe_layers", "retnet_layers"):
            indices = getattr(self, name)
            if any(i < 0 for i in indices) or len(indices) != len(set(indices)):
                raise ValueError(f"{name} must contain unique nonnegative indices")
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
        total, active = self.parameter_counts()
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
    assert cfg.num_active_parameters <= cfg.num_parameters

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
