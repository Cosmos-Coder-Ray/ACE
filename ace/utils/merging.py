"""CPU interpolation of compatible ACE fine-tunes; no backward pass or new parameters."""

from __future__ import annotations

import argparse
import json
import math
import uuid
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Iterator, cast

import torch
from safetensors import safe_open

from ace.inference.bundle import read_manifest, write_weights
from ace.model.backbone import AceModel
from ace.model.config import AceConfig
from ace.tokenizer.ace_tokenizer import AceTokenizer

# safetensors supplies tensor APIs but its context-manager constructor is untyped.
open_safetensors = cast(Callable[..., Any], safe_open)


def merge_bundles(left: Path, right: Path, output: Path, alpha: float = 0.5) -> dict[str, Any]:
    """Interpolate weights and biases from a common initialization or base model.

    alpha=0 preserves left; alpha=1 preserves right. Config, tensor layouts,
    tokenizer IDs and backend must agree. Metadata lineage is a provenance check,
    not a guarantee of preserved skill: evaluate both parents and the result.
    """
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    a, b = read_manifest(left), read_manifest(right)
    for field in ("config", "backend", "tokenizer_sha256", "lineage_id"):
        if not a.get(field) or a[field] != b.get(field):
            raise ValueError(f"Incompatible {field}; arbitrary architectures cannot be merged")
    if set(a["tensor_files"]) != set(b["tensor_files"]):
        raise ValueError("Tensor names differ")

    with ExitStack() as stack:
        sources = [
            {
                filename: stack.enter_context(
                    open_safetensors(str(path / filename), framework="pt", device="cpu")
                )
                for filename in meta["weight_files"]
            }
            for path, meta in ((left, a), (right, b))
        ]
        # Validate headers before creating an output directory or doing arithmetic.
        for name in a["tensor_files"]:
            x = sources[0][a["tensor_files"][name]].get_slice(name)
            y = sources[1][b["tensor_files"][name]].get_slice(name)
            if x.get_shape() != y.get_shape() or x.get_dtype() != y.get_dtype():
                raise ValueError(f"Incompatible tensor: {name}")
        output.mkdir(parents=True, exist_ok=False)

        def tensors() -> Iterator[tuple[str, torch.Tensor]]:
            for name in a["tensor_files"]:
                x = sources[0][a["tensor_files"][name]].get_tensor(name)
                y = sources[1][b["tensor_files"][name]].get_tensor(name)
                if not x.is_floating_point() or name.endswith("inv_freq"):
                    if not torch.equal(x, y):
                        raise ValueError(f"Non-parameter buffer differs: {name}")
                    yield name, x
                else:
                    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                        raise ValueError(f"Nonfinite input weights: {name}")
                    yield name, torch.lerp(x.float(), y.float(), alpha).to(x.dtype)

        mapping, hashes = write_weights(tensors(), output)
    AceTokenizer.load(left / "tokenizer").save(output / "tokenizer")
    AceConfig(**a["config"]).to_yaml(output / "config.yaml")
    metadata = dict(a)
    metadata.update(
        tensor_files=mapping,
        weight_files=hashes,
        provenance={
            "operation": "linear_merge",
            "alpha": alpha,
            "quality_verified": False,
            "parents": [file_digest(left), file_digest(right)],
        },
    )
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def upcycle_bundle(
    source: Path,
    output: Path,
    layers: list[int],
    *,
    num_experts: int = 4,
    top_k: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    """Copy dense FFNs into experts without training or inventing new skills.

    Identical experts preserve the dense FFN output because ACE normalizes its
    selected routing weights to sum to one. Further training is needed for
    specialization. Disk size and parameter count increase immediately.
    """
    original = read_manifest(source)
    old_config = AceConfig(**original["config"])
    if old_config.moe_layers or not layers:
        raise ValueError("Upcycling expects a dense base and at least one target layer")
    config = replace(old_config, moe_layers=layers, num_experts=num_experts, top_k_experts=top_k)
    with torch.device("meta"):
        parameters = AceModel(config).num_parameters
    selected = {f"layers.{index}.ffn.": index for index in layers}
    output.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        readers = {
            filename: stack.enter_context(
                open_safetensors(str(source / filename), framework="pt", device="cpu")
            )
            for filename in original["weight_files"]
        }

        def tensors() -> Iterator[tuple[str, torch.Tensor]]:
            for name, filename in original["tensor_files"].items():
                value = readers[filename].get_tensor(name)
                prefix = next((p for p in selected if name.startswith(p)), None)
                if prefix is None:
                    yield name, value
                else:
                    for expert in range(num_experts):
                        yield f"{prefix}experts.{expert}.{name[len(prefix):]}", value
            generator = torch.Generator().manual_seed(seed)
            for prefix in selected:
                # Break routing ties; duplicated experts still produce the same output.
                yield prefix + "router.gate.weight", torch.randn(
                    num_experts,
                    config.d_model,
                    generator=generator,
                ) * 0.01

        mapping, hashes = write_weights(tensors(), output)
    AceTokenizer.load(source / "tokenizer").save(output / "tokenizer")
    config.to_yaml(output / "config.yaml")
    metadata = dict(original)
    metadata.update(
        config=asdict(config),
        parameters=parameters,
        lineage_id=uuid.uuid4().hex,
        tensor_files=mapping,
        weight_files=hashes,
        provenance={
            "operation": "dense_to_moe",
            "parent": file_digest(source),
            "seed": seed,
            "quality_verified": False,
            "new_skills_trained": False,
        },
    )
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def file_digest(path: Path) -> str:
    from ace.data.preprocessing import file_sha256

    return file_sha256(path / "manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    metadata = merge_bundles(args.left, args.right, args.output, args.alpha)
    print(json.dumps({"parameters": metadata["parameters"], "quality_verified": False}, indent=2))


if __name__ == "__main__":
    main()
