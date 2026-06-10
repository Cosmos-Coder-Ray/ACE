"""Comprehensive tests for the ACE tokenizer system.

Tests:
    - Roundtrip ASCII encode/decode
    - Roundtrip code encode/decode
    - All special tokens present in vocabulary
    - Save/load produces identical outputs
    - Batch encode == single encode (by decoded text)
    - Special token IDs are consistent
    - Error handling for invalid inputs
    - encode with add_special_tokens wraps BOS/EOS
    - vocab_size property
    - Unicode roundtrip
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ace.tokenizer.ace_tokenizer import AceTokenizer
from ace.tokenizer.special_tokens import (
    BOS,
    EOS,
    PAD,
    SPECIAL_TOKENS_DICT,
    SPECIAL_TOKENS_LIST,
    THINK_END,
    THINK_START,
    TOOL_CALL,
    TOOL_RESULT,
)

# -- Fixtures -----------------------------------------------------------


@pytest.fixture(scope="module")
def trained_tokenizer() -> AceTokenizer:
    """Train a small tokenizer once for the entire test module.

    Returns:
        A trained AceTokenizer with a small vocabulary.
    """
    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / "corpus.txt"
        # Build a corpus with diverse content for reliable roundtrips
        lines: list[str] = []

        # Python code
        lines.extend(
            [
                "import torch",
                "import torch.nn as nn",
                "import numpy as np",
                "from pathlib import Path",
                "from typing import Optional, List, Dict, Tuple, Any",
                "",
                "class TransformerBlock(nn.Module):",
                "    def __init__(self, d_model: int, n_heads: int) -> None:",
                "        super().__init__()",
                "        self.attention = nn.MultiheadAttention(d_model, n_heads)",
                "        self.norm = nn.LayerNorm(d_model)",
                "",
                "    def forward(self, x: torch.Tensor) -> torch.Tensor:",
                '        """Forward pass with residual."""',
                "        return x + self.attention(self.norm(x), self.norm(x), self.norm(x))[0]",
                "",
                "def fibonacci(n: int) -> int:",
                "    if n <= 1:",
                "        return n",
                "    return fibonacci(n - 1) + fibonacci(n - 2)",
                "",
                "# Mathematical operations",
                "x = (a + b) * c / d - e ** f % g",
                "result = {key: value for key, value in items.items()}",
                "",
                "# JSON-like data structures",
                '{"name": "ACE", "version": 1, "features": ["BPE", "Mamba"]}',
                "",
                "# Unicode comments: hello world abcdef 12345",
                "# Special chars: !@#$^&*()[]{}|\\;:,.<>?/~`",
                "",
                "# f-strings and formatting",
                'msg = f"The answer is {answer} and {2+2}"',
                'path = f"/home/{user}/data/{name}.txt"',
                "",
                "# The quick brown fox jumps over the lazy dog.",
                "# ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz 0123456789",
                "",
                "camelCase = snake_case = PascalCase = SCREAMING_SNAKE = True",
                "",
                "# Tab-indented block:",
                "\tif True:",
                '\t\tprint("Hello!")',
                "",
            ]
        )

        full_text = "\n".join(lines) * 500  # repeat for enough data
        corpus_path.write_text(full_text, encoding="utf-8")

        tok_obj = AceTokenizer.train(
            files=[corpus_path],
            vocab_size=2000,
            min_frequency=1,
        )
        return tok_obj


@pytest.fixture(scope="module")
def save_dir(trained_tokenizer: AceTokenizer) -> Path:
    """Save tokenizer and return the directory path.

    Persisted for the module so load tests can use it.

    Args:
        trained_tokenizer: The trained tokenizer fixture.

    Returns:
        Path to the saved tokenizer directory.
    """
    tmp = tempfile.mkdtemp()
    save_path = Path(tmp) / "tok_save"
    trained_tokenizer.save(save_path)
    return save_path


# -- Special tokens module tests ----------------------------------------


class TestSpecialTokens:
    """Tests for ace.tokenizer.special_tokens module."""

    def test_special_tokens_count(self) -> None:
        """There are exactly 12 special tokens."""
        assert len(SPECIAL_TOKENS_LIST) == 12

    def test_special_tokens_unique(self) -> None:
        """All special tokens are unique strings."""
        assert len(set(SPECIAL_TOKENS_LIST)) == 12

    def test_special_tokens_sentinel_format(self) -> None:
        """All tokens use the <|...|> sentinel format."""
        for token in SPECIAL_TOKENS_LIST:
            assert token.startswith("<|") and token.endswith(
                "|>"
            ), f"Token {token!r} does not match sentinel format"

    def test_special_tokens_dict_keys(self) -> None:
        """SPECIAL_TOKENS_DICT has the expected keys."""
        expected_keys = {
            "PAD",
            "BOS",
            "EOS",
            "SYSTEM",
            "USER",
            "ASSISTANT",
            "CODE_START",
            "CODE_END",
            "THINK_START",
            "THINK_END",
            "TOOL_CALL",
            "TOOL_RESULT",
        }
        assert set(SPECIAL_TOKENS_DICT.keys()) == expected_keys

    def test_specific_token_values(self) -> None:
        """Key tokens have the exact expected values."""
        assert PAD == "<|pad|>"
        assert BOS == "<|ace_start|>"
        assert EOS == "<|ace_end|>"
        assert THINK_START == "<|think|>"
        assert THINK_END == "<|think_end|>"
        assert TOOL_CALL == "<|tool_call|>"
        assert TOOL_RESULT == "<|tool_result|>"

    def test_dict_matches_list(self) -> None:
        """SPECIAL_TOKENS_LIST matches SPECIAL_TOKENS_DICT.values()."""
        assert list(SPECIAL_TOKENS_DICT.values()) == SPECIAL_TOKENS_LIST


# -- Roundtrip tests -----------------------------------------------------


class TestRoundtripASCII:
    """Roundtrip encode->decode tests for plain ASCII text."""

    @pytest.mark.parametrize(
        "text",
        [
            "hello world",
            "The quick brown fox jumps over the lazy dog.",
            "UPPERCASE lowercase MiXeD",
            "1234567890",
            "!@#$%^&*()",
            "",
        ],
    )
    def test_ascii_roundtrip(self, trained_tokenizer: AceTokenizer, text: str) -> None:
        """Encoding then decoding ASCII text recovers the original."""
        ids = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(ids)
        assert decoded == text

    def test_whitespace_roundtrip(self, trained_tokenizer: AceTokenizer) -> None:
        """Whitespace-heavy text round-trips correctly."""
        text = "  hello   world  \n\t  foo  "
        ids = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(ids)
        assert decoded == text

    def test_punctuation_roundtrip(self, trained_tokenizer: AceTokenizer) -> None:
        """Punctuation-heavy text round-trips correctly."""
        text = "x = (a + b) * c / d"
        ids = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(ids)
        assert decoded == text


class TestRoundtripCode:
    """Roundtrip tests for source code strings."""

    @pytest.mark.parametrize(
        "code",
        [
            "def foo(): return 42",
            "class Foo(Bar):\n    pass",
            "x = [i for i in range(10)]",
            "import torch\nfrom torch import nn",
            "    # indented comment",
            'print("hello")',
        ],
    )
    def test_code_roundtrip(self, trained_tokenizer: AceTokenizer, code: str) -> None:
        """Encoding then decoding code recovers the original."""
        ids = trained_tokenizer.encode(code)
        decoded = trained_tokenizer.decode(ids)
        assert decoded == code

    def test_multiline_code_roundtrip(self, trained_tokenizer: AceTokenizer) -> None:
        """Multi-line code block round-trips correctly."""
        code = (
            "def forward(self, x):\n"
            "    residual = x\n"
            "    x = self.norm(x)\n"
            "    return x + residual\n"
        )
        ids = trained_tokenizer.encode(code)
        decoded = trained_tokenizer.decode(ids)
        assert decoded == code


# -- Special token presence tests ----------------------------------------


class TestSpecialTokensInVocab:
    """Verify all special tokens exist in the trained vocabulary."""

    def test_all_special_tokens_present(self, trained_tokenizer: AceTokenizer) -> None:
        """Every special token has a non-None ID in the vocabulary."""
        for name, token_str in SPECIAL_TOKENS_DICT.items():
            tid = trained_tokenizer.token_to_id(token_str)
            assert tid is not None, f"Special token {name} ({token_str}) missing from vocab"

    def test_special_token_ids_property(self, trained_tokenizer: AceTokenizer) -> None:
        """The special_token_ids property returns all 12 IDs."""
        stids = trained_tokenizer.special_token_ids
        assert len(stids) == 12
        assert all(isinstance(v, int) for v in stids.values())
        assert all(v >= 0 for v in stids.values())

    def test_special_token_ids_are_unique(self, trained_tokenizer: AceTokenizer) -> None:
        """Each special token has a distinct integer ID."""
        stids = trained_tokenizer.special_token_ids
        ids_list = list(stids.values())
        assert len(set(ids_list)) == len(ids_list)

    def test_pad_bos_eos_convenience_properties(self, trained_tokenizer: AceTokenizer) -> None:
        """pad_token_id, bos_token_id, eos_token_id match special_token_ids."""
        stids = trained_tokenizer.special_token_ids
        assert trained_tokenizer.pad_token_id == stids["PAD"]
        assert trained_tokenizer.bos_token_id == stids["BOS"]
        assert trained_tokenizer.eos_token_id == stids["EOS"]


# -- Save / Load tests --------------------------------------------------


class TestSaveLoad:
    """Verify save/load produces identical tokenizer outputs."""

    def test_save_creates_files(self, save_dir: Path) -> None:
        """Save creates tokenizer.json and special_tokens_map.json."""
        assert (save_dir / "tokenizer.json").exists()
        assert (save_dir / "special_tokens_map.json").exists()

    def test_load_roundtrip(
        self,
        trained_tokenizer: AceTokenizer,
        save_dir: Path,
    ) -> None:
        """Loading a saved tokenizer produces identical encode outputs."""
        loaded = AceTokenizer.load(save_dir)
        test_texts = [
            "def foo(): return 42",
            "import torch",
            "hello world",
        ]
        for text in test_texts:
            orig_ids = trained_tokenizer.encode(text)
            loaded_ids = loaded.encode(text)
            assert orig_ids == loaded_ids, f"Mismatch for {text!r}: {orig_ids} != {loaded_ids}"

    def test_load_vocab_size_matches(
        self,
        trained_tokenizer: AceTokenizer,
        save_dir: Path,
    ) -> None:
        """Loaded tokenizer has the same vocab_size."""
        loaded = AceTokenizer.load(save_dir)
        assert loaded.vocab_size == trained_tokenizer.vocab_size

    def test_load_special_tokens_match(
        self,
        trained_tokenizer: AceTokenizer,
        save_dir: Path,
    ) -> None:
        """Loaded tokenizer has the same special token IDs."""
        loaded = AceTokenizer.load(save_dir)
        assert loaded.special_token_ids == trained_tokenizer.special_token_ids

    def test_load_nonexistent_raises(self) -> None:
        """Loading from a nonexistent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            AceTokenizer.load(Path("/nonexistent/path/tok"))


# -- Batch encoding tests -----------------------------------------------


class TestBatchEncoding:
    """Verify batch encoding consistency with single encoding."""

    def test_batch_matches_single_decoded(self, trained_tokenizer: AceTokenizer) -> None:
        """Batch encode produces same decoded text as single encode."""
        texts = ["hello world", "import torch", "def foo(): pass"]
        batch_ids = trained_tokenizer.encode_batch(texts)
        for text, b_ids in zip(texts, batch_ids, strict=False):
            single_ids = trained_tokenizer.encode(text)
            dec_batch = trained_tokenizer.decode(b_ids)
            dec_single = trained_tokenizer.decode(single_ids)
            assert dec_batch == dec_single == text

    def test_batch_empty_list(self, trained_tokenizer: AceTokenizer) -> None:
        """Batch encode with empty list returns empty list."""
        assert trained_tokenizer.encode_batch([]) == []

    def test_batch_single_element(self, trained_tokenizer: AceTokenizer) -> None:
        """Batch encode with one element matches single encode."""
        text = "hello"
        batch_ids = trained_tokenizer.encode_batch([text])
        single_ids = trained_tokenizer.encode(text)
        # Decoded text must match
        assert trained_tokenizer.decode(batch_ids[0]) == trained_tokenizer.decode(single_ids)

    def test_decode_batch(self, trained_tokenizer: AceTokenizer) -> None:
        """decode_batch returns list of decoded strings."""
        texts = ["hello", "world"]
        batch_ids = trained_tokenizer.encode_batch(texts)
        decoded = trained_tokenizer.decode_batch(batch_ids)
        assert decoded == texts


# -- add_special_tokens tests -------------------------------------------


class TestAddSpecialTokens:
    """Tests for encode with add_special_tokens=True."""

    def test_adds_bos_eos(self, trained_tokenizer: AceTokenizer) -> None:
        """encode(add_special_tokens=True) wraps text with BOS and EOS."""
        text = "hello"
        ids = trained_tokenizer.encode(text, add_special_tokens=True)
        bos_id = trained_tokenizer.bos_token_id
        eos_id = trained_tokenizer.eos_token_id
        assert ids[0] == bos_id
        assert ids[-1] == eos_id

    def test_decode_skip_special_tokens(self, trained_tokenizer: AceTokenizer) -> None:
        """decode(skip_special_tokens=True) strips BOS/EOS."""
        text = "hello world"
        ids = trained_tokenizer.encode(text, add_special_tokens=True)
        decoded = trained_tokenizer.decode(ids, skip_special_tokens=True)
        assert decoded == text


# -- Property tests -----------------------------------------------------


class TestProperties:
    """Tests for tokenizer properties."""

    def test_vocab_size_positive(self, trained_tokenizer: AceTokenizer) -> None:
        """vocab_size is a positive integer."""
        assert trained_tokenizer.vocab_size > 0

    def test_len_equals_vocab_size(self, trained_tokenizer: AceTokenizer) -> None:
        """len(tokenizer) equals vocab_size."""
        assert len(trained_tokenizer) == trained_tokenizer.vocab_size

    def test_repr(self, trained_tokenizer: AceTokenizer) -> None:
        """repr includes vocab_size and special token count."""
        r = repr(trained_tokenizer)
        assert "AceTokenizer" in r
        assert "vocab_size=" in r
        assert "special_tokens=12" in r


# -- Error handling tests ------------------------------------------------


class TestErrorHandling:
    """Tests for input validation and error handling."""

    def test_encode_non_string_raises(self, trained_tokenizer: AceTokenizer) -> None:
        """encode raises TypeError for non-string input."""
        with pytest.raises(TypeError, match="Expected str"):
            trained_tokenizer.encode(42)  # type: ignore[arg-type]

    def test_encode_batch_non_list_raises(self, trained_tokenizer: AceTokenizer) -> None:
        """encode_batch raises TypeError for non-list input."""
        with pytest.raises(TypeError, match="Expected list"):
            trained_tokenizer.encode_batch("hello")  # type: ignore[arg-type]

    def test_encode_batch_non_string_element_raises(self, trained_tokenizer: AceTokenizer) -> None:
        """encode_batch raises TypeError if element is not a string."""
        with pytest.raises(TypeError, match="Expected str at index"):
            trained_tokenizer.encode_batch(["hello", 42])  # type: ignore[list-item]

    def test_decode_non_list_raises(self, trained_tokenizer: AceTokenizer) -> None:
        """decode raises TypeError for non-list input."""
        with pytest.raises(TypeError, match="Expected list"):
            trained_tokenizer.decode("hello")  # type: ignore[arg-type]

    def test_constructor_non_tokenizer_raises(self) -> None:
        """AceTokenizer constructor raises TypeError for wrong type."""
        with pytest.raises(TypeError, match="Expected tokenizers.Tokenizer"):
            AceTokenizer("not a tokenizer")  # type: ignore[arg-type]

    def test_train_invalid_vocab_size(self) -> None:
        """train raises ValueError for non-positive vocab_size."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            AceTokenizer.train(files=[], vocab_size=0)

    def test_train_invalid_min_frequency(self) -> None:
        """train raises ValueError for min_frequency < 1."""
        with pytest.raises(ValueError, match="min_frequency must be >= 1"):
            AceTokenizer.train(files=[], vocab_size=100, min_frequency=0)

    def test_train_missing_file(self) -> None:
        """train raises FileNotFoundError for missing training file."""
        with pytest.raises(FileNotFoundError, match="Training file not found"):
            AceTokenizer.train(
                files=[Path("/nonexistent/file.txt")],
                vocab_size=100,
            )


# -- Utility method tests -----------------------------------------------


class TestUtilities:
    """Tests for token_to_id and id_to_token."""

    def test_token_to_id_known(self, trained_tokenizer: AceTokenizer) -> None:
        """token_to_id returns an int for a known token."""
        pad_id = trained_tokenizer.token_to_id(PAD)
        assert isinstance(pad_id, int)
        assert pad_id >= 0

    def test_token_to_id_unknown(self, trained_tokenizer: AceTokenizer) -> None:
        """token_to_id returns None for an unknown token."""
        result = trained_tokenizer.token_to_id("<|nonexistent_token_xyz|>")
        assert result is None

    def test_id_to_token_roundtrip(self, trained_tokenizer: AceTokenizer) -> None:
        """token_to_id and id_to_token are inverses for special tokens."""
        for name, token_str in SPECIAL_TOKENS_DICT.items():
            tid = trained_tokenizer.token_to_id(token_str)
            assert tid is not None
            recovered = trained_tokenizer.id_to_token(tid)
            assert (
                recovered == token_str
            ), f"Roundtrip failed for {name}: {token_str} -> {tid} -> {recovered}"
