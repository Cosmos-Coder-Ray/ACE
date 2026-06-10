"""AceTokenizer — custom BPE tokenizer for the ACE model.

Wraps the HuggingFace ``tokenizers`` library to provide a byte-level BPE
tokenizer with NFC normalisation, 65 536 vocab, and ACE-specific special
tokens.  No lowercasing is applied because source code is case-sensitive.

Usage::

    tokenizer = AceTokenizer.load(Path("data/tokenizer"))
    ids = tokenizer.encode("def hello():")
    text = tokenizer.decode(ids)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from ace.tokenizer.special_tokens import SPECIAL_TOKENS_DICT, SPECIAL_TOKENS_LIST

logger = logging.getLogger(__name__)


class AceTokenizer:
    """Custom BPE tokenizer for the ACE model.

    Attributes:
        _tokenizer: Underlying HuggingFace ``Tokenizer`` instance.
    """

    # -- Construction ---------------------------------------------------

    def __init__(self, tokenizer: Tokenizer) -> None:
        """Initialise from an already-configured ``Tokenizer``.

        Args:
            tokenizer: A HuggingFace ``Tokenizer`` instance with BPE model,
                NFC normalisation, and ByteLevel pre-tokenisation already set.

        Raises:
            TypeError: If *tokenizer* is not a ``Tokenizer`` instance.
        """
        if not isinstance(tokenizer, Tokenizer):
            raise TypeError(f"Expected tokenizers.Tokenizer, got {type(tokenizer).__name__}")
        self._tokenizer: Tokenizer = tokenizer

    @classmethod
    def _build_base(cls) -> Tokenizer:
        """Create a blank BPE tokenizer with ACE defaults.

        Returns:
            A ``Tokenizer`` with BPE model, NFC normaliser, ByteLevel
            pre-tokeniser (no prefix space), and ByteLevel decoder.
        """
        tok_obj = Tokenizer(BPE(unk_token=None))
        tok_obj.normalizer = NFC()
        tok_obj.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tok_obj.decoder = ByteLevelDecoder()
        return tok_obj

    # -- Training -------------------------------------------------------

    @classmethod
    def train(
        cls,
        files: list[str | Path],
        vocab_size: int = 65_536,
        min_frequency: int = 2,
    ) -> "AceTokenizer":
        """Train a new BPE tokenizer from scratch.

        Args:
            files: Paths to plain-text training files.
            vocab_size: Target vocabulary size (must be positive).
            min_frequency: Minimum merge frequency (must be >= 1).

        Returns:
            A fully trained ``AceTokenizer``.

        Raises:
            ValueError: If *vocab_size* <= 0 or *min_frequency* < 1.
            FileNotFoundError: If any training file does not exist.
        """
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if min_frequency < 1:
            raise ValueError(f"min_frequency must be >= 1, got {min_frequency}")

        # Validate files exist
        resolved: list[str] = []
        for f in files:
            p = Path(f)
            if not p.exists():
                raise FileNotFoundError(f"Training file not found: {p}")
            resolved.append(str(p))

        logger.info(
            "Training BPE tokenizer — vocab_size=%d, min_freq=%d, files=%d",
            vocab_size,
            min_frequency,
            len(resolved),
        )

        base = cls._build_base()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS_LIST,
            show_progress=True,
        )
        base.train(resolved, trainer)

        # Re-add special tokens with special=True for safe encoding
        for token_str in SPECIAL_TOKENS_LIST:
            base.add_special_tokens([AddedToken(token_str, special=True)])

        # NOTE: Padding is NOT enabled here. It is the responsibility of
        # the data collator / pipeline to pad sequences, giving callers
        # full control over pad direction and max length.

        logger.info("Training complete — final vocab size: %d", base.get_vocab_size())
        return cls(base)

    # -- Encoding -------------------------------------------------------

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:
        """Encode a single string to token IDs.

        Args:
            text: Input text to encode.
            add_special_tokens: Whether to wrap with BOS/EOS.

        Returns:
            List of integer token IDs.

        Raises:
            TypeError: If *text* is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        if add_special_tokens:
            text = SPECIAL_TOKENS_DICT["BOS"] + text + SPECIAL_TOKENS_DICT["EOS"]

        encoding = self._tokenizer.encode(text)
        result: list[int] = encoding.ids  # list[int]
        return result

    def encode_batch(
        self,
        texts: list[str],
        add_special_tokens: bool = False,
    ) -> list[list[int]]:
        """Encode a batch of strings to token IDs.

        Args:
            texts: List of input strings.
            add_special_tokens: Whether to wrap each with BOS/EOS.

        Returns:
            List of token-ID lists, one per input string.

        Raises:
            TypeError: If *texts* is not a list or contains non-strings.
        """
        if not isinstance(texts, list):
            raise TypeError(f"Expected list, got {type(texts).__name__}")

        prepared: list[str] = []
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(f"Expected str at index {i}, got {type(t).__name__}")
            if add_special_tokens:
                t = SPECIAL_TOKENS_DICT["BOS"] + t + SPECIAL_TOKENS_DICT["EOS"]
            prepared.append(t)

        encodings = self._tokenizer.encode_batch(prepared)
        return [enc.ids for enc in encodings]

    # -- Decoding -------------------------------------------------------

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = False,
    ) -> str:
        """Decode token IDs back to a string.

        Args:
            ids: List of integer token IDs.
            skip_special_tokens: Whether to strip special tokens from output.

        Returns:
            Decoded string.

        Raises:
            TypeError: If *ids* is not a list.
        """
        if not isinstance(ids, list):
            raise TypeError(f"Expected list[int], got {type(ids).__name__}")
        result: str = self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        return result

    def decode_batch(
        self,
        batch_ids: list[list[int]],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        """Decode a batch of token-ID lists back to strings.

        Args:
            batch_ids: List of token-ID lists.
            skip_special_tokens: Whether to strip special tokens from output.

        Returns:
            List of decoded strings.

        Raises:
            TypeError: If *batch_ids* is not a list of lists.
        """
        if not isinstance(batch_ids, list):
            raise TypeError(f"Expected list[list[int]], got {type(batch_ids).__name__}")
        result: list[str] = self._tokenizer.decode_batch(
            batch_ids, skip_special_tokens=skip_special_tokens
        )
        return result

    # -- Properties -----------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens.

        Returns:
            Number of tokens in the vocabulary.
        """
        size: int = self._tokenizer.get_vocab_size()
        return size

    @property
    def special_token_ids(self) -> dict[str, int]:
        """Map logical special-token name to its integer ID.

        Returns:
            Dictionary of token-name -> token-ID for every special token.

        Raises:
            RuntimeError: If any expected special token is missing from vocab.
        """
        result: dict[str, int] = {}
        for name, token_str in SPECIAL_TOKENS_DICT.items():
            tid = self._tokenizer.token_to_id(token_str)
            if tid is None:
                raise RuntimeError(f"Special token {name!r} ({token_str!r}) not in vocabulary")
            result[name] = tid
        return result

    @property
    def pad_token_id(self) -> int:
        """Integer ID of the PAD token.

        Returns:
            PAD token ID.

        Raises:
            RuntimeError: If PAD token is not in vocabulary.
        """
        tid = self._tokenizer.token_to_id(SPECIAL_TOKENS_DICT["PAD"])
        if tid is None:
            raise RuntimeError("PAD token not in vocabulary")
        return int(tid)

    @property
    def bos_token_id(self) -> int:
        """Integer ID of the BOS token.

        Returns:
            BOS token ID.

        Raises:
            RuntimeError: If BOS token is not in vocabulary.
        """
        tid = self._tokenizer.token_to_id(SPECIAL_TOKENS_DICT["BOS"])
        if tid is None:
            raise RuntimeError("BOS token not in vocabulary")
        return int(tid)

    @property
    def eos_token_id(self) -> int:
        """Integer ID of the EOS token.

        Returns:
            EOS token ID.

        Raises:
            RuntimeError: If EOS token is not in vocabulary.
        """
        tid = self._tokenizer.token_to_id(SPECIAL_TOKENS_DICT["EOS"])
        if tid is None:
            raise RuntimeError("EOS token not in vocabulary")
        return int(tid)

    # -- Persistence ----------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """Save tokenizer to a directory.

        Creates ``tokenizer.json`` and ``special_tokens_map.json`` in
        *directory*.

        Args:
            directory: Target directory (created if needed).

        Returns:
            The resolved directory path.

        Raises:
            OSError: If the directory cannot be created or written to.
        """
        dirpath = Path(directory)
        dirpath.mkdir(parents=True, exist_ok=True)

        tokenizer_path = dirpath / "tokenizer.json"
        self._tokenizer.save(str(tokenizer_path))

        # Also persist the special-token mapping for reconstruction
        special_map_path = dirpath / "special_tokens_map.json"
        with special_map_path.open("w", encoding="utf-8") as f:
            json.dump(SPECIAL_TOKENS_DICT, f, indent=2, ensure_ascii=False)

        logger.info("Saved tokenizer to %s", dirpath)
        return dirpath

    @classmethod
    def load(cls, directory: str | Path) -> "AceTokenizer":
        """Load a previously saved tokenizer from a directory.

        Args:
            directory: Directory containing ``tokenizer.json``.

        Returns:
            A fully initialised ``AceTokenizer``.

        Raises:
            FileNotFoundError: If *directory* or ``tokenizer.json`` does not exist.
        """
        dirpath = Path(directory)
        tokenizer_path = dirpath / "tokenizer.json"

        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

        base = Tokenizer.from_file(str(tokenizer_path))
        logger.info(
            "Loaded tokenizer from %s — vocab size: %d",
            dirpath,
            base.get_vocab_size(),
        )
        return cls(base)

    # -- Utilities ------------------------------------------------------

    def token_to_id(self, token: str) -> Optional[int]:
        """Look up the integer ID for a token string.

        Args:
            token: The token string to look up.

        Returns:
            The token ID, or ``None`` if not in vocabulary.
        """
        result: int | None = self._tokenizer.token_to_id(token)
        return result

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Look up the string for a token ID.

        Args:
            token_id: The integer ID to look up.

        Returns:
            The token string, or ``None`` if out of range.
        """
        result: str | None = self._tokenizer.id_to_token(token_id)
        return result

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return (
            f"AceTokenizer(vocab_size={self.vocab_size}, "
            f"special_tokens={len(SPECIAL_TOKENS_LIST)})"
        )

    def __len__(self) -> int:
        """Return the vocabulary size."""
        return self.vocab_size


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    print("Building minimal tokenizer for smoke test...")

    # Create a tiny training corpus
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text(
            "def hello_world():\n"
            '    print("Hello, World!")\n'
            "    return 42\n"
            "\n"
            "class Foo:\n"
            "    def bar(self, x: int) -> int:\n"
            "        return x * 2\n"
            "\n"
            "# This is a comment\n"
            "import torch\n"
            "import numpy as np\n" * 100,  # repeat for enough data
            encoding="utf-8",
        )

        # Train with small vocab
        tok_obj = AceTokenizer.train(
            files=[corpus_file],
            vocab_size=500,
            min_frequency=1,
        )
        print(f"  Trained: {tok_obj}")
        assert tok_obj.vocab_size >= 12, "Vocab too small for special tokens"

        # Roundtrip test
        test_text = "def foo(): return 42"
        ids = tok_obj.encode(test_text)
        decoded = tok_obj.decode(ids)
        assert decoded == test_text, f"Roundtrip failed: {test_text!r} -> {ids} -> {decoded!r}"
        print(f"  Roundtrip OK: {test_text!r}")

        # Batch encoding — compare decoded text (ByteLevel offsets may
        # differ between batch/single but decoded output must match)
        batch_texts = ["hello world", "import os"]
        batch_ids = tok_obj.encode_batch(batch_texts)
        single_ids = [tok_obj.encode(t) for t in batch_texts]
        for i, (bt, bi, si) in enumerate(zip(batch_texts, batch_ids, single_ids, strict=False)):
            dec_batch = tok_obj.decode(bi)
            dec_single = tok_obj.decode(si)
            assert dec_batch == dec_single == bt, (
                f"Batch/single mismatch at {i}: "
                f"batch={dec_batch!r}, single={dec_single!r}, orig={bt!r}"
            )
        print("  Batch encoding OK")

        # Special tokens
        stids = tok_obj.special_token_ids
        assert len(stids) == 12, f"Expected 12 special token IDs, got {len(stids)}"
        assert all(isinstance(v, int) for v in stids.values())
        print(f"  Special tokens OK: {stids}")

        # Save / load roundtrip
        save_dir = tmp_path / "saved_tokenizer"
        tok_obj.save(save_dir)
        tok_loaded = AceTokenizer.load(save_dir)
        ids2 = tok_loaded.encode(test_text)
        assert ids == ids2, "Save/load roundtrip failed"
        print("  Save/load OK")

    print("[PASS] ace_tokenizer.py smoke test passed")
