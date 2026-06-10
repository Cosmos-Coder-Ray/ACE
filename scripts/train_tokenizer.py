#!/usr/bin/env python3
"""CLI script to train the ACE BPE tokenizer.

Trains on user-provided text files (e.g. The Stack, ArXiv), validates
on 10 samples, saves to ``data/tokenizer/``, and reports a compression
ratio versus raw UTF-8 bytes.

Usage::

    python scripts/train_tokenizer.py \
        --files data/raw/stack.txt data/raw/arxiv.txt \
        --vocab-size 65536 \
        --min-frequency 2 \
        --output data/tokenizer

    # Quick demo with auto-generated sample data:
    python scripts/train_tokenizer.py --demo
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path

# Ensure the project root is on sys.path when running as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ace.tokenizer.ace_tokenizer import AceTokenizer  # noqa: E402
from ace.tokenizer.special_tokens import SPECIAL_TOKENS_DICT  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_tokenizer")


# -- Validation samples ------------------------------------------------

VALIDATION_SAMPLES: list[str] = [
    # Plain English
    "The quick brown fox jumps over the lazy dog.",
    # Python code
    "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
    # Import statement
    "import torch\nfrom torch import nn\nimport numpy as np",
    # Class definition
    "class AceModel(nn.Module):\n    def __init__(self, config):\n        super().__init__()\n",
    # Mixed case (case-sensitivity check)
    "camelCase snake_case PascalCase SCREAMING_SNAKE",
    # Special characters
    "x = (a + b) * c / d - e ** f % g",
    # Docstring
    '""""""A docstring with triple quotes.""""""\n',
    # Unicode
    "# Comments with unicode: \u00e9\u00e0\u00fc\u00f1 \u03b1\u03b2\u03b3\u03b4",
    # JSON-like
    '{"key": "value", "list": [1, 2, 3], "nested": {"a": true}}',
    # f-string
    'result = f"The answer is {answer}"',
]


def compute_compression_ratio(tokenizer: AceTokenizer, texts: list[str]) -> float:
    """Compute compression ratio: raw bytes / token count.

    Args:
        tokenizer: Trained tokenizer instance.
        texts: List of text samples.

    Returns:
        Average bytes per token (higher = better compression).
    """
    total_bytes = 0
    total_tokens = 0
    for text in texts:
        raw_bytes = len(text.encode("utf-8"))
        token_ids = tokenizer.encode(text)
        total_bytes += raw_bytes
        total_tokens += len(token_ids)

    if total_tokens == 0:
        return 0.0
    return total_bytes / total_tokens


def validate_tokenizer(tokenizer: AceTokenizer) -> bool:
    """Run validation checks on the trained tokenizer.

    Args:
        tokenizer: Trained tokenizer instance.

    Returns:
        True if all checks pass.
    """
    all_ok = True

    # 1. Check all special tokens are present
    logger.info("Checking special tokens...")
    for name, token_str in SPECIAL_TOKENS_DICT.items():
        tid = tokenizer.token_to_id(token_str)
        if tid is None:
            logger.error("  FAIL: Special token %s (%s) not in vocab", name, token_str)
            all_ok = False
        else:
            logger.info("  OK: %s -> id %d", name, tid)

    # 2. Roundtrip validation on 10 samples
    logger.info("Running roundtrip validation on %d samples...", len(VALIDATION_SAMPLES))
    for i, sample in enumerate(VALIDATION_SAMPLES):
        ids = tokenizer.encode(sample)
        decoded = tokenizer.decode(ids)
        if decoded != sample:
            logger.error(
                "  FAIL sample %d: \n    original : %r\n    decoded  : %r",
                i,
                sample,
                decoded,
            )
            all_ok = False
        else:
            logger.info(
                "  OK sample %d: %d chars -> %d tokens",
                i,
                len(sample),
                len(ids),
            )

    # 3. Compression ratio
    ratio = compute_compression_ratio(tokenizer, VALIDATION_SAMPLES)
    logger.info("Compression ratio: %.2f bytes/token", ratio)

    return all_ok


def generate_demo_corpus(path: Path) -> Path:
    """Generate a small demo corpus for testing.

    Args:
        path: Directory to write the corpus file.

    Returns:
        Path to the generated corpus file.
    """
    corpus_path = path / "demo_corpus.txt"
    lines: list[str] = []

    # Python code samples
    lines.extend(
        [
            "import torch",
            "import torch.nn as nn",
            "import numpy as np",
            "from pathlib import Path",
            "from typing import Optional, List, Dict, Tuple",
            "",
            "class TransformerBlock(nn.Module):",
            "    def __init__(self, d_model: int, n_heads: int):",
            "        super().__init__()",
            "        self.attention = nn.MultiheadAttention(d_model, n_heads)",
            "        self.norm1 = nn.LayerNorm(d_model)",
            "        self.norm2 = nn.LayerNorm(d_model)",
            "        self.ffn = nn.Sequential(",
            "            nn.Linear(d_model, 4 * d_model),",
            "            nn.GELU(),",
            "            nn.Linear(4 * d_model, d_model),",
            "        )",
            "",
            "    def forward(self, x: torch.Tensor) -> torch.Tensor:",
            '        """Forward pass with pre-norm and residual."""',
            "        residual = x",
            "        x = self.norm1(x)",
            "        x = self.attention(x, x, x)[0]",
            "        x = x + residual",
            "        residual = x",
            "        x = self.norm2(x)",
            "        x = self.ffn(x)",
            "        return x + residual",
            "",
            "def train_step(model, optimizer, batch):",
            "    optimizer.zero_grad()",
            "    loss = model(batch).loss",
            "    loss.backward()",
            "    optimizer.step()",
            "    return loss.item()",
            "",
        ]
    )

    # Repeat for more data
    full_text = "\n".join(lines) * 200

    corpus_path.write_text(full_text, encoding="utf-8")
    logger.info("Generated demo corpus: %s (%.1f KB)", corpus_path, len(full_text) / 1024)
    return corpus_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train the ACE BPE tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--files",
        nargs="+",
        type=str,
        help="Paths to training text files",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=65536,
        help="Target vocabulary size (default: 65536)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum merge frequency (default: 2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/tokenizer",
        help="Output directory for saved tokenizer (default: data/tokenizer)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with auto-generated sample data",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point for the tokenizer training CLI.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    args = parse_args()

    if args.demo:
        logger.info("Running in DEMO mode with generated corpus")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_path = generate_demo_corpus(tmp_path)
            return _run_training(
                files=[str(corpus_path)],
                vocab_size=1000,  # small for demo
                min_frequency=1,
                output_dir=args.output,
            )
    else:
        if not args.files:
            logger.error("No training files specified. Use --files or --demo.")
            return 1
        return _run_training(
            files=args.files,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            output_dir=args.output,
        )


def _run_training(
    files: list[str],
    vocab_size: int,
    min_frequency: int,
    output_dir: str,
) -> int:
    """Execute the training pipeline.

    Args:
        files: Paths to training text files.
        vocab_size: Target vocabulary size.
        min_frequency: Minimum merge frequency.
        output_dir: Output directory for saved tokenizer.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    logger.info("=" * 60)
    logger.info("ACE Tokenizer Training")
    logger.info("=" * 60)
    logger.info("  Files        : %d", len(files))
    logger.info("  Vocab size   : %d", vocab_size)
    logger.info("  Min frequency: %d", min_frequency)
    logger.info("  Output       : %s", output_dir)
    logger.info("=" * 60)

    # Train
    t0 = time.perf_counter()
    tokenizer = AceTokenizer.train(
        files=[Path(f) for f in files],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )
    elapsed = time.perf_counter() - t0
    logger.info("Training completed in %.1f seconds", elapsed)
    logger.info("Final vocab size: %d", tokenizer.vocab_size)

    # Validate
    logger.info("-" * 60)
    ok = validate_tokenizer(tokenizer)

    # Save
    logger.info("-" * 60)
    out_path = Path(output_dir)
    tokenizer.save(out_path)
    logger.info("Tokenizer saved to: %s", out_path.resolve())

    # Report
    logger.info("-" * 60)
    ratio = compute_compression_ratio(tokenizer, VALIDATION_SAMPLES)
    logger.info("COMPRESSION RATIO: %.2f bytes/token", ratio)

    # Compute ratio against raw bytes of training files
    total_raw = sum(Path(f).stat().st_size for f in files if Path(f).exists())
    if total_raw > 0:
        logger.info("Training data size: %.1f MB", total_raw / 1024 / 1024)

    logger.info("=" * 60)
    if ok:
        logger.info("ALL VALIDATION CHECKS PASSED")
        return 0
    else:
        logger.error("SOME VALIDATION CHECKS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
