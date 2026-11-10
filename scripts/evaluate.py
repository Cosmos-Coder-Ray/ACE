"""Evaluate token-weighted held-out language loss. Does not execute generated code."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ace.data.collator import CausalCollator
from ace.data.dataset import PackedTokenDataset
from ace.data.preprocessing import file_sha256
from ace.inference.bundle import load_bundle
from ace.model.backbone import AceModel


@torch.inference_mode()
def evaluate_model(model: AceModel, loader: DataLoader[dict[str, torch.Tensor]]) -> dict[str, Any]:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    loss_sum, count, correct = 0.0, 0, 0
    try:
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = (
                model(batch["input_ids"], attention_mask=batch["attention_mask"])
                .logits[:, :-1]
                .float()
            )
            labels = batch["labels"][:, 1:]
            valid = labels != -100
            loss_sum += F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            ).item()
            count += int(valid.sum())
            correct += int(((logits.argmax(-1) == labels) & valid).sum())
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("Evaluation has no target tokens")
    mean = loss_sum / count
    return {
        "tokens": count,
        "cross_entropy": mean,
        "perplexity": math.exp(mean) if mean < 700 else None,
        "next_token_accuracy": correct / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model, tokenizer, metadata = load_bundle(args.model)
    data_manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    if data_manifest["tokenizer_sha256"] != metadata["tokenizer_sha256"]:
        raise ValueError("Evaluation data uses a different tokenizer")
    if (
        file_sha256(args.data / "validation.bin")
        != data_manifest["streams"]["validation"]["sha256"]
    ):
        raise ValueError("Validation stream fingerprint mismatch")
    seq_len = args.seq_len or model.config.max_seq_len
    if not 2 <= seq_len <= model.config.max_seq_len:
        raise ValueError("Invalid evaluation context length")
    dataset = PackedTokenDataset(args.data / "validation.bin", seq_len, tokenizer.vocab_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=CausalCollator(tokenizer.pad_token_id)
    )
    metrics = evaluate_model(model, loader)
    metrics.update(
        model_sha256=file_sha256(args.model / "manifest.json"),
        data_sha256=file_sha256(args.data / "manifest.json"),
        seq_len=seq_len,
    )
    text = json.dumps(metrics, indent=2, allow_nan=False)
    if args.output:
        if args.output.exists():
            raise FileExistsError("Choose a new evaluation output file")
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
