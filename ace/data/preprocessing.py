"""Deterministic, offline corpus preparation for small ACE experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ace.model.config import AceConfig
from ace.tokenizer.ace_tokenizer import AceTokenizer


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_documents(paths: list[Path]) -> Iterator[str]:
    """Read explicit UTF-8 files; JSONL records must have a nonempty text field."""
    for path in paths:
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    text = row.get("text") if isinstance(row, dict) else None
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(f"{path}:{line_number}: expected nonempty text")
                    yield text
        else:
            # newline='' preserves CRLF and source formatting.
            with path.open(encoding="utf-8", newline="") as stream:
                text = stream.read()
            if text.strip():
                yield text


def prepare_corpus(
    paths: list[Path],
    output: Path,
    config: AceConfig,
    *,
    validation_fraction: float = 0.1,
    seed: int = 42,
    source_license: str,
    tokenizer_path: Path | None = None,
) -> dict[str, Any]:
    """Split documents before tokenizer training, deduplicate, then write token streams.

    Exact duplicate detection keeps only document hashes in RAM. Near-duplicate
    and repository-level splitting are separate requirements for release corpora.
    Existing output directories are rejected to protect previous experiments.
    """
    if not paths or not 0 < validation_fraction < 1 or not source_license.strip():
        raise ValueError(
            "Provide input files, a license description, and a split fraction in (0, 1)"
        )
    output.mkdir(parents=True, exist_ok=False)
    seen: set[str] = set()
    counts = {"train": 0, "validation": 0}
    duplicates = 0
    with (
        (output / "train.jsonl").open("w", encoding="utf-8") as train,
        (output / "validation.jsonl").open("w", encoding="utf-8") as validation,
        (output / "tokenizer_corpus.txt").open("w", encoding="utf-8", newline="") as corpus,
    ):
        for text in read_documents(paths):
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                duplicates += 1
                continue
            seen.add(digest)
            split_hash = hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest()
            split = (
                "validation" if int(split_hash[:16], 16) / 2**64 < validation_fraction else "train"
            )
            stream = validation if split == "validation" else train
            stream.write(json.dumps({"text": text, "sha256": digest}, ensure_ascii=False) + "\n")
            counts[split] += 1
            if split == "train":
                corpus.write(text + "\n")
    if not all(counts.values()):
        raise ValueError(
            "Both splits need documents. Add documents or adjust the seed/split fraction."
        )

    tokenizer = (
        AceTokenizer.load(tokenizer_path)
        if tokenizer_path is not None
        else AceTokenizer.train([output / "tokenizer_corpus.txt"], vocab_size=config.vocab_size)
    )
    tokenizer.save(output / "tokenizer")
    actual_config = replace(config, vocab_size=tokenizer.vocab_size)
    actual_config.to_yaml(output / "config.yaml")
    streams = {}
    for split in counts:
        token_count = 0
        path = output / f"{split}.bin"
        with path.open("wb") as binary_stream:
            for text in read_documents([output / f"{split}.jsonl"]):
                ids = tokenizer.encode(text, add_special_tokens=True)
                np.asarray(ids, dtype="<u4").tofile(binary_stream)
                token_count += len(ids)
        streams[split] = {"tokens": token_count, "sha256": file_sha256(path)}

    manifest = {
        "format_version": 1,
        "dtype": "<u4",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "documents": counts,
        "exact_duplicates_removed": duplicates,
        "streams": streams,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_sha256": file_sha256(output / "tokenizer" / "tokenizer.json"),
        "source_license": source_license,
        "sources": [{"path": str(p.resolve()), "sha256": file_sha256(p)} for p in paths],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
