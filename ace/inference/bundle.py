"""Portable ACE bundles with tokenizer identity, backend identity and hashed weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors.torch import load_file, save_file

from ace.data.preprocessing import file_sha256
from ace.model.backbone import AceModel
from ace.model.config import AceConfig
from ace.model.mamba_block import Mamba
from ace.tokenizer.ace_tokenizer import AceTokenizer


def backend_name(config: AceConfig) -> str:
    return f"{Mamba.__module__}.{Mamba.__name__}" if config.mamba_layers else "pytorch-attention"


def write_weights(
    tensors: Iterable[tuple[str, torch.Tensor]],
    directory: Path,
    shard_bytes: int = 64 * 1024**2,
) -> tuple[dict[str, str], dict[str, str]]:
    """Bound output buffering to one shard plus the largest individual tensor."""
    mapping: dict[str, str] = {}
    hashes: dict[str, str] = {}
    shard: dict[str, torch.Tensor] = {}
    size = 0

    def flush() -> None:
        if not shard:
            return
        filename = f"weights-{len(hashes):05d}.safetensors"
        save_file(shard, str(directory / filename))
        hashes[filename] = file_sha256(directory / filename)
        mapping.update(dict.fromkeys(shard, filename))
        shard.clear()

    for name, tensor in tensors:
        count = tensor.numel() * tensor.element_size()
        if shard and size + count > shard_bytes:
            flush()
            size = 0
        shard[name] = tensor.detach().cpu().contiguous().clone()
        size += count
    flush()
    return mapping, hashes


def save_bundle(
    model: AceModel,
    tokenizer: AceTokenizer,
    directory: Path,
    *,
    lineage_id: str,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Model and tokenizer vocabularies must match exactly")
    directory.mkdir(parents=True, exist_ok=False)
    tokenizer.save(directory / "tokenizer")
    model.config.to_yaml(directory / "config.yaml")
    from dataclasses import asdict

    weights = (
        (name, value)
        for name, value in model.state_dict().items()
        if not (model.config.tie_embeddings and name == "lm_head.weight")
    )
    mapping, hashes = write_weights(weights, directory)
    metadata = {
        "format_version": 1,
        "config": asdict(model.config),
        "backend": backend_name(model.config),
        "lineage_id": lineage_id,
        "parameters": model.num_parameters,
        "tokenizer_sha256": file_sha256(directory / "tokenizer" / "tokenizer.json"),
        "tensor_files": mapping,
        "weight_files": hashes,
        "provenance": provenance or {},
    }
    (directory / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def read_manifest(directory: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported bundle version")
    if file_sha256(directory / "tokenizer" / "tokenizer.json") != metadata["tokenizer_sha256"]:
        raise ValueError("Tokenizer fingerprint mismatch")
    for filename, digest in metadata["weight_files"].items():
        if Path(filename).name != filename or "/" in filename or "\\" in filename:
            raise ValueError("Weight filenames must stay inside the bundle")
        if file_sha256(directory / filename) != digest:
            raise ValueError(f"Weight fingerprint mismatch: {filename}")
    if set(metadata["tensor_files"].values()) != set(metadata["weight_files"]):
        raise ValueError("Invalid shard mapping")
    return metadata


def load_bundle(
    directory: Path, device: str = "cpu"
) -> tuple[AceModel, AceTokenizer, dict[str, Any]]:
    metadata = read_manifest(directory)
    config = AceConfig(**metadata["config"])
    if backend_name(config) != metadata["backend"]:
        raise ValueError("Mamba backend differs from the saved model; install its original backend")
    tokenizer = AceTokenizer.load(directory / "tokenizer")
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("Model/tokenizer vocabulary mismatch")
    model = AceModel(config)
    state = {}
    for filename in metadata["weight_files"]:
        state.update(load_file(str(directory / filename)))
    if config.tie_embeddings:
        state["lm_head.weight"] = state["embed_tokens.weight"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), tokenizer, metadata
