"""Right padding with explicit attention and next-token loss masks."""

from dataclasses import dataclass

import torch


@dataclass
class CausalCollator:
    pad_token_id: int

    def __call__(self, examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        width = max(len(x["input_ids"]) for x in examples)
        ids = torch.full((len(examples), width), self.pad_token_id, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for row, example in enumerate(examples):
            tokens = example["input_ids"]
            ids[row, : len(tokens)] = tokens
            labels[row, : len(tokens)] = example.get("labels", tokens)
            mask[row, : len(tokens)] = True
        return {"input_ids": ids, "labels": labels, "attention_mask": mask[:, None, None, :]}
