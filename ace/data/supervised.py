"""Assistant-only and fill-in-the-middle examples for native ACE SFT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ace.tokenizer.ace_tokenizer import AceTokenizer
from ace.tokenizer.special_tokens import SPECIAL_TOKENS_DICT


def _token_id(tokenizer: AceTokenizer, name: str) -> int:
    token_id = tokenizer.token_to_id(SPECIAL_TOKENS_DICT[name])
    if token_id is None:
        raise ValueError(f"Tokenizer does not contain the {name} token")
    return token_id


def encode_supervised_record(
    record: dict[str, Any], tokenizer: AceTokenizer, seq_len: int
) -> dict[str, torch.Tensor]:
    """Encode one instruction or FIM record with loss only on its target."""
    kind = record.get("kind")
    ids: list[int] = [tokenizer.bos_token_id]
    labels: list[int] = [-100]

    if kind == "instruction":
        system = record.get("system", "You are a coding assistant.")
        user = record.get("user")
        assistant = record.get("assistant")
        if not isinstance(system, str) or not isinstance(user, str) or not isinstance(
            assistant, str
        ):
            raise ValueError("Instruction records require string system, user, and assistant fields")
        ids.extend([_token_id(tokenizer, "SYSTEM"), *tokenizer.encode(system)])
        labels.extend([-100] * (1 + len(tokenizer.encode(system))))
        ids.extend([_token_id(tokenizer, "USER"), *tokenizer.encode(user)])
        labels.extend([-100] * (1 + len(tokenizer.encode(user))))
        ids.append(_token_id(tokenizer, "ASSISTANT"))
        labels.append(-100)
        target = tokenizer.encode(assistant)
        ids.extend(target)
        labels.extend(target)
        ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)
    elif kind == "fim":
        prefix, suffix, middle = (record.get(key) for key in ("prefix", "suffix", "middle"))
        if not all(isinstance(value, str) for value in (prefix, suffix, middle)) or not middle:
            raise ValueError("FIM records require string prefix, suffix, and nonempty middle fields")
        prefix_ids = tokenizer.encode(prefix)
        suffix_ids = tokenizer.encode(suffix)
        middle_ids = tokenizer.encode(middle)
        ids.extend([_token_id(tokenizer, "FIM_PREFIX"), *prefix_ids])
        labels.extend([-100] * (1 + len(prefix_ids)))
        ids.extend([_token_id(tokenizer, "FIM_SUFFIX"), *suffix_ids])
        labels.extend([-100] * (1 + len(suffix_ids)))
        ids.append(_token_id(tokenizer, "FIM_MIDDLE"))
        labels.append(-100)
        ids.extend(middle_ids)
        labels.extend(middle_ids)
        ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)
    else:
        raise ValueError(f"Unsupported supervised record kind: {kind!r}")

    ids = ids[:seq_len]
    labels = labels[:seq_len]
    if len(ids) < 2 or all(label == -100 for label in labels[1:]):
        raise ValueError("Supervised record has no target tokens within the configured sequence length")
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels)}


class SupervisedDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, path: str | Path, tokenizer: AceTokenizer, seq_len: int) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.records: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{self.path}:{line_number}: expected a JSON object")
                self.records.append(record)
        if not self.records:
            raise ValueError(f"No supervised records in {self.path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < len(self.records):
            raise IndexError(index)
        return encode_supervised_record(self.records[index], self.tokenizer, self.seq_len)
