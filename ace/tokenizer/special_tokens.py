"""ACE special token definitions.

Every sentinel token used by the ACE tokenizer is declared here.
Tokens use the ``<|...|>`` format so they are never confused with
natural-language or code substrings.

Exports:
    SPECIAL_TOKENS_DICT: ``dict[str, str]`` mapping logical name -> token string.
    SPECIAL_TOKENS_LIST: ``list[str]`` of all token strings, insertion-ordered.

Usage::

    from ace.tokenizer.special_tokens import SPECIAL_TOKENS_DICT, SPECIAL_TOKENS_LIST

    pad_token = SPECIAL_TOKENS_DICT["PAD"]       # "<|pad|>"
    all_tokens = SPECIAL_TOKENS_LIST              # ["<|pad|>", ...]
"""

from __future__ import annotations

# -- Individual token constants ----------------------------------------
PAD: str = "<|pad|>"
BOS: str = "<|ace_start|>"
EOS: str = "<|ace_end|>"

SYSTEM: str = "<|system|>"
USER: str = "<|user|>"
ASSISTANT: str = "<|assistant|>"

CODE_START: str = "<|code_start|>"
CODE_END: str = "<|code_end|>"

THINK_START: str = "<|think|>"
THINK_END: str = "<|think_end|>"

TOOL_CALL: str = "<|tool_call|>"
TOOL_RESULT: str = "<|tool_result|>"


# -- Aggregated exports ------------------------------------------------
SPECIAL_TOKENS_DICT: dict[str, str] = {
    "PAD": PAD,
    "BOS": BOS,
    "EOS": EOS,
    "SYSTEM": SYSTEM,
    "USER": USER,
    "ASSISTANT": ASSISTANT,
    "CODE_START": CODE_START,
    "CODE_END": CODE_END,
    "THINK_START": THINK_START,
    "THINK_END": THINK_END,
    "TOOL_CALL": TOOL_CALL,
    "TOOL_RESULT": TOOL_RESULT,
}
"""Mapping from logical token name to its string representation."""

SPECIAL_TOKENS_LIST: list[str] = list(SPECIAL_TOKENS_DICT.values())
"""Ordered list of all special token strings."""


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    assert (
        len(SPECIAL_TOKENS_LIST) == 12
    ), f"Expected 12 special tokens, got {len(SPECIAL_TOKENS_LIST)}"
    assert len(set(SPECIAL_TOKENS_LIST)) == len(
        SPECIAL_TOKENS_LIST
    ), "Duplicate special tokens detected"
    for name, token in SPECIAL_TOKENS_DICT.items():
        assert token.startswith("<|") and token.endswith(
            "|>"
        ), f"Token {name!r} = {token!r} does not match sentinel format"
    assert SPECIAL_TOKENS_DICT["BOS"] == "<|ace_start|>"
    assert SPECIAL_TOKENS_DICT["EOS"] == "<|ace_end|>"
    assert SPECIAL_TOKENS_DICT["THINK_START"] == "<|think|>"
    print(f"[PASS] special_tokens.py smoke test — {len(SPECIAL_TOKENS_LIST)} tokens")
