"""Memory-mapped packed token streams; the corpus does not live in RAM."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTokenDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, path: str | Path, seq_len: int, vocab_size: int) -> None:
        path = Path(path)
        if seq_len < 2 or path.stat().st_size < 8 or path.stat().st_size % 4:
            raise ValueError("Expected a uint32 stream with >=2 tokens and seq_len >=2")
        self.tokens = np.memmap(path, dtype="<u4", mode="r")
        self.seq_len = seq_len
        # Validate in bounded chunks, including the last partial chunk.
        for start in range(0, len(self.tokens), 1_000_000):
            if int(self.tokens[start : start + 1_000_000].max()) >= vocab_size:
                raise ValueError("Dataset token IDs exceed the model vocabulary")

    def __len__(self) -> int:
        return (len(self.tokens) - 2) // (self.seq_len - 1) + 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        # One-token overlap preserves every next-token target across blocks.
        start = index * (self.seq_len - 1)
        ids = torch.from_numpy(self.tokens[start : start + self.seq_len].astype(np.int64))
        return {"input_ids": ids}
