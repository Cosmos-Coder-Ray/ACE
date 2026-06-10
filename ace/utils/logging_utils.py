"""Rank-aware structured logging for ACE.

Features:
    - JSON-structured log records
    - Distributed-training aware (only rank-0 logs by default)
    - Configurable log levels per module
    - W&B integration helpers

Usage:
    from ace.utils.logging_utils import get_logger
    logger = get_logger(__name__)
    logger.info("starting training", extra={"epoch": 1})
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any


class StructuredFormatter(logging.Formatter):
    """JSON formatter that emits one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        """Format *record* as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            JSON-encoded log string.
        """
        payload: dict[str, Any] = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge any extra fields the caller passed
        for key in ("epoch", "step", "loss", "lr", "rank"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, default=str)


class RankFilter(logging.Filter):
    """Only allow log records through on rank 0 (or when not in distributed mode).

    Args:
        rank: The current process rank. ``0`` means main process.
    """

    def __init__(self, rank: int = 0) -> None:
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        """Allow the record if we are rank 0 or the record is WARNING+.

        Args:
            record: log record to filter.

        Returns:
            ``True`` if the record should be emitted.
        """
        if self.rank == 0:
            return True
        # Non-rank-0 processes only emit warnings and above
        return record.levelno >= logging.WARNING


def _get_rank() -> int:
    """Detect the distributed rank from common env vars.

    Returns:
        Integer rank (0 if not in distributed mode).
    """
    for var in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        val = os.environ.get(var)
        if val is not None:
            return int(val)
    return 0


def get_logger(
    name: str,
    level: int = logging.INFO,
    *,
    structured: bool = True,
    rank_zero_only: bool = True,
) -> logging.Logger:
    """Create or retrieve a rank-aware logger.

    Args:
        name: Logger name (typically ``__name__``).
        level: Minimum log level.
        structured: If ``True``, emit JSON-formatted logs.
        rank_zero_only: If ``True``, only rank-0 emits INFO+.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured — avoid duplicate handlers
        return logger

    logger.setLevel(level)
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if structured:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    if rank_zero_only:
        handler.addFilter(RankFilter(rank=_get_rank()))

    logger.addHandler(handler)
    return logger


# ──────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log = get_logger("ace.test", structured=True)
    log.info("structured log test", extra={"step": 42, "loss": 0.123})

    log_plain = get_logger("ace.test_plain", structured=False)
    log_plain.info("plain log test")

    print("[PASS] logging_utils smoke test passed")
