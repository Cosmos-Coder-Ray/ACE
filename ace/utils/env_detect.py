"""Environment detection and capability flags for ACE.

This module is the **single source of truth** for what hardware accelerators
and optional libraries are available at runtime.  Every other ACE module
should import capability flags from here rather than attempting its own
``try/except`` import blocks.

Detected capabilities:
    - CUDA availability and device count
    - FlashAttention (``flash_attn``)
    - Mamba SSM (``mamba_ssm``)
    - Causal Conv1D (``causal_conv1d``)
    - MegaBlocks MoE (``megablocks``)
    - Triton JIT kernels (``triton``)
    - Operating system (Windows vs. POSIX)

Usage::

    from ace.utils.env_detect import (
        HAS_CUDA, HAS_FLASH_ATTN, HAS_MAMBA, DEVICE, DTYPE,
    )
    from ace.utils.env_detect import log_environment

    log_environment()  # emit a one-line summary at startup
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import torch

# ── Module logger ─────────────────────────────────────────────────────
logger: logging.Logger = logging.getLogger(__name__)

# ── CUDA ──────────────────────────────────────────────────────────────
HAS_CUDA: bool = torch.cuda.is_available()
"""``True`` when at least one CUDA device is visible."""

CUDA_DEVICE_COUNT: int = torch.cuda.device_count() if HAS_CUDA else 0
"""Number of visible CUDA devices (0 when CPU-only)."""

# ── FlashAttention ────────────────────────────────────────────────────
HAS_FLASH_ATTN: bool = False
"""``True`` when the ``flash_attn`` package is importable."""
try:
    import flash_attn as _flash_attn  # noqa: F401

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

# ── Mamba SSM ─────────────────────────────────────────────────────────
HAS_MAMBA: bool = False
"""``True`` when the ``mamba_ssm`` package is importable."""
try:
    import mamba_ssm as _mamba_ssm  # noqa: F401

    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

# ── Causal Conv1D ─────────────────────────────────────────────────────
HAS_CAUSAL_CONV1D: bool = False
"""``True`` when the ``causal_conv1d`` package is importable."""
try:
    import causal_conv1d as _causal_conv1d  # noqa: F401

    HAS_CAUSAL_CONV1D = True
except ImportError:
    HAS_CAUSAL_CONV1D = False

# ── MegaBlocks (MoE) ─────────────────────────────────────────────────
HAS_MEGABLOCKS: bool = False
"""``True`` when the ``megablocks`` package is importable."""
try:
    import megablocks as _megablocks  # noqa: F401

    HAS_MEGABLOCKS = True
except ImportError:
    HAS_MEGABLOCKS = False

# ── Triton ────────────────────────────────────────────────────────────
HAS_TRITON: bool = False
"""``True`` when the ``triton`` package is importable."""
try:
    import triton as _triton  # noqa: F401

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# ── OS detection ──────────────────────────────────────────────────────
IS_WINDOWS: bool = sys.platform == "win32"
"""``True`` when running on Windows."""

# ── Derived constants ─────────────────────────────────────────────────
DEVICE: str = "cuda" if HAS_CUDA else "cpu"
"""Default device string (``'cuda'`` or ``'cpu'``)."""

DTYPE: torch.dtype = torch.bfloat16 if HAS_CUDA else torch.float32
"""Default training dtype.

``bfloat16`` is used on CUDA (native hardware support on Ampere+).
``float32`` is used on CPU because ``bfloat16`` matmuls are not
hardware-accelerated and can silently produce lower-quality gradients.
"""


# ── Public helpers ────────────────────────────────────────────────────


def get_environment_summary() -> dict[str, Any]:
    """Collect all capability flags into a single dictionary.

    Returns:
        Dictionary mapping flag names to their boolean / string values.
        Suitable for JSON serialisation or logging.
    """
    return {
        "has_cuda": HAS_CUDA,
        "cuda_device_count": CUDA_DEVICE_COUNT,
        "has_flash_attn": HAS_FLASH_ATTN,
        "has_mamba": HAS_MAMBA,
        "has_causal_conv1d": HAS_CAUSAL_CONV1D,
        "has_megablocks": HAS_MEGABLOCKS,
        "has_triton": HAS_TRITON,
        "is_windows": IS_WINDOWS,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "torch_version": torch.__version__,
        "python_version": sys.version,
    }


def log_environment() -> None:
    """Log a one-line capability summary at INFO level.

    Output looks like::

        [ACE] CUDA: YES (2 GPUs) | FlashAttn: YES | Mamba: YES |
              MoE: megablocks | Triton: YES | Device: cuda | Dtype: bfloat16

    When an optional accelerator is missing, the log line indicates
    which fallback path will be used so operators can diagnose
    performance issues at a glance.
    """
    cuda_status: str = (
        f"YES ({CUDA_DEVICE_COUNT} GPU{'s' if CUDA_DEVICE_COUNT != 1 else ''})"
        if HAS_CUDA
        else "NO"
    )
    flash_status: str = "YES" if HAS_FLASH_ATTN else "NO (using PyTorch SDPA)"
    mamba_status: str = "YES" if HAS_MAMBA else "NO (using pure-PyTorch SSM)"
    moe_status: str = "megablocks" if HAS_MEGABLOCKS else "fallback"
    triton_status: str = "YES" if HAS_TRITON else "NO"
    conv1d_status: str = "YES" if HAS_CAUSAL_CONV1D else "NO"
    os_label: str = "Windows" if IS_WINDOWS else "POSIX"

    summary_line: str = (
        f"[ACE] CUDA: {cuda_status} | "
        f"FlashAttn: {flash_status} | "
        f"Mamba: {mamba_status} | "
        f"CausalConv1D: {conv1d_status} | "
        f"MoE: {moe_status} | "
        f"Triton: {triton_status} | "
        f"Device: {DEVICE} | "
        f"Dtype: {DTYPE} | "
        f"OS: {os_label}"
    )
    logger.info(summary_line)


def require_cuda(operation: str = "this operation") -> None:
    """Raise if CUDA is not available.

    A convenience guard for code paths that *cannot* fall back to CPU
    (e.g. custom CUDA kernels).

    Args:
        operation: Human-readable name inserted into the error message.

    Raises:
        RuntimeError: When ``HAS_CUDA`` is ``False``.
    """
    if not HAS_CUDA:
        raise RuntimeError(
            f"CUDA is required for {operation}, but no CUDA device was detected. "
            f"torch.cuda.is_available() returned False."
        )


# ──────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Environment summary dict is well-formed
    summary: dict[str, Any] = get_environment_summary()
    assert isinstance(summary, dict), "get_environment_summary must return a dict"
    expected_keys: set[str] = {
        "has_cuda",
        "cuda_device_count",
        "has_flash_attn",
        "has_mamba",
        "has_causal_conv1d",
        "has_megablocks",
        "has_triton",
        "is_windows",
        "device",
        "dtype",
        "torch_version",
        "python_version",
    }
    missing_keys: set[str] = expected_keys - set(summary.keys())
    assert not missing_keys, f"Missing keys in summary: {missing_keys}"

    # 2. Boolean flags are actually bools
    for flag_name in (
        "has_cuda",
        "has_flash_attn",
        "has_mamba",
        "has_causal_conv1d",
        "has_megablocks",
        "has_triton",
        "is_windows",
    ):
        assert isinstance(summary[flag_name], bool), f"{flag_name} must be bool"

    # 3. DEVICE / DTYPE consistency
    if HAS_CUDA:
        assert DEVICE == "cuda", "DEVICE should be 'cuda' when CUDA is available"
        assert DTYPE == torch.bfloat16, "DTYPE should be bfloat16 on CUDA"
    else:
        assert DEVICE == "cpu", "DEVICE should be 'cpu' when CUDA is absent"
        assert DTYPE == torch.float32, "DTYPE should be float32 on CPU"

    # 4. IS_WINDOWS consistency
    assert IS_WINDOWS == (sys.platform == "win32"), "IS_WINDOWS mismatch"

    # 5. log_environment runs without error
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    log_environment()

    # 6. require_cuda behaviour
    if not HAS_CUDA:
        try:
            require_cuda("smoke test")
            raise AssertionError("require_cuda should have raised RuntimeError")
        except RuntimeError:
            pass  # expected

    print("[PASS] env_detect.py smoke test passed")
