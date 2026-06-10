"""ACE utilities — registry, logging, shared helpers, environment detection, fallbacks."""

from ace.utils.env_detect import (
    DEVICE,
    DTYPE,
    HAS_CAUSAL_CONV1D,
    HAS_CUDA,
    HAS_FLASH_ATTN,
    HAS_MAMBA,
    HAS_MEGABLOCKS,
    HAS_TRITON,
    IS_WINDOWS,
    get_environment_summary,
    log_environment,
    require_cuda,
)
from ace.utils.fallbacks import (
    FallbackCausalConv1d,
    FallbackMamba,
    FallbackMoELayer,
    fallback_flash_attention,
    get_attention_fn,
    get_mamba_class,
    get_moe_class,
)
from ace.utils.logging_utils import get_logger
from ace.utils.registry import get, register

__all__ = [
    # Registry
    "register",
    "get",
    # Logging
    "get_logger",
    # Environment detection — flags
    "HAS_CUDA",
    "HAS_FLASH_ATTN",
    "HAS_MAMBA",
    "HAS_CAUSAL_CONV1D",
    "HAS_MEGABLOCKS",
    "HAS_TRITON",
    "IS_WINDOWS",
    # Environment detection — constants
    "DEVICE",
    "DTYPE",
    # Environment detection — functions
    "get_environment_summary",
    "log_environment",
    "require_cuda",
    # Fallback implementations
    "FallbackMamba",
    "fallback_flash_attention",
    "FallbackCausalConv1d",
    "FallbackMoELayer",
    # Fallback factory functions
    "get_mamba_class",
    "get_attention_fn",
    "get_moe_class",
]

