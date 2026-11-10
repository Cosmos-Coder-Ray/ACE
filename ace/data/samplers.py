"""Deterministic sample order with an explicit cursor for interrupted sessions."""

from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class ResumableRandomSampler(Sampler[int]):
    """Use with num_workers=0 so prefetching cannot advance the saved cursor."""

    def __init__(self, size: int, seed: int = 42) -> None:
        if size < 1:
            raise ValueError("Sampler requires a nonempty dataset")
        self.size, self.seed, self.epoch, self.position = size, seed, 0, 0

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[int]:
        if self.position == self.size:
            self.epoch += 1
            self.position = 0
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=generator).tolist()
        while self.position < self.size:
            index = order[self.position]
            self.position += 1
            yield index

    def state_dict(self) -> dict[str, int]:
        return {
            "size": self.size,
            "seed": self.seed,
            "epoch": self.epoch,
            "position": self.position,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        if state["size"] != self.size or state["seed"] != self.seed:
            raise ValueError("Cannot resume sampler with a different dataset or seed")
        if not 0 <= state["position"] <= self.size or state["epoch"] < 0:
            raise ValueError("Invalid sampler cursor")
        self.epoch, self.position = state["epoch"], state["position"]
