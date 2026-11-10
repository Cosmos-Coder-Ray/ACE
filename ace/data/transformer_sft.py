"""Supervised examples encoded with a Hugging Face tokenizer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def _token_list(token_ids: Any) -> list[int]:
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def encode_transformer_record(
    record: dict[str, Any], tokenizer: Any, seq_len: int
) -> dict[str, torch.Tensor]:
    kind = record.get("kind")
    if kind == "instruction":
        system = record.get("system", "You are a coding assistant.")
        user = record.get("user")
        assistant = record.get("assistant")
        if not all(isinstance(value, str) for value in (system, user, assistant)):
            raise ValueError("Instruction records require string system, user, and assistant fields")
        prompt_ids = _token_list(
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        full_ids = _token_list(
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("Tokenizer chat template does not preserve the assistant prompt prefix")
        input_ids = full_ids
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    elif kind == "fim":
        prefix, suffix, middle = (record.get(key) for key in ("prefix", "suffix", "middle"))
        if not all(isinstance(value, str) for value in (prefix, suffix, middle)) or not middle:
            raise ValueError("FIM records require string prefix, suffix, and nonempty middle fields")
        fim_ids: dict[str, int] = {}
        for token in ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id == tokenizer.unk_token_id:
                raise ValueError(f"Model tokenizer does not contain {token}")
            fim_ids[token] = int(token_id)
        prefix_ids = _token_list(tokenizer.encode(prefix, add_special_tokens=False))
        suffix_ids = _token_list(tokenizer.encode(suffix, add_special_tokens=False))
        middle_ids = _token_list(tokenizer.encode(middle, add_special_tokens=False))
        input_ids = [
            fim_ids["<|fim_prefix|>"],
            *prefix_ids,
            fim_ids["<|fim_suffix|>"],
            *suffix_ids,
            fim_ids["<|fim_middle|>"],
            *middle_ids,
        ]
        labels = [-100] * (len(input_ids) - len(middle_ids)) + middle_ids
        if tokenizer.eos_token_id is not None:
            input_ids.append(int(tokenizer.eos_token_id))
            labels.append(int(tokenizer.eos_token_id))
    else:
        raise ValueError(f"Unsupported supervised record kind: {kind!r}")

    input_ids = input_ids[:seq_len]
    labels = labels[:seq_len]
    if len(input_ids) < 2 or all(label == -100 for label in labels[1:]):
        raise ValueError("Supervised record has no target tokens within the configured sequence length")
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels)}


class TransformersSFTDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, path: str | Path, tokenizer: Any, seq_len: int) -> None:
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
        return encode_transformer_record(self.records[index], self.tokenizer, self.seq_len)


class TransformersSFTCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        width = max(int(example["input_ids"].numel()) for example in examples)
        input_ids = torch.full((len(examples), width), self.pad_token_id, dtype=torch.long)
        labels = torch.full_like(input_ids, -100)
        attention_mask = torch.zeros_like(input_ids)
        for row, example in enumerate(examples):
            length = int(example["input_ids"].numel())
            input_ids[row, :length] = example["input_ids"]
            labels[row, :length] = example["labels"]
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
