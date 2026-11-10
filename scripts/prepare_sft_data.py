"""Build deterministic native ACE FIM examples and combine instruction records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ace.data.preprocessing import file_sha256
from ace.data.repositories import ALLOWED_LICENSES


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def _fim_record(code: dict[str, Any], seed: int) -> dict[str, Any] | None:
    text = code.get("text")
    if not isinstance(text, str):
        raise ValueError("Code corpus records require a text field")
    lines = text.splitlines(keepends=True)
    if len(lines) < 3:
        return None
    identity = f"{seed}:{code.get('repository', '')}:{code.get('path', '')}:{text[:256]}"
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    start = 1 + int.from_bytes(digest[:4], "little") % (len(lines) - 2)
    width = 1 + int.from_bytes(digest[4:8], "little") % min(8, len(lines) - start)
    stop = min(len(lines) - 1, start + width)
    return {
        "kind": "fim",
        "language": code.get("language", "unknown"),
        "repository": code.get("repository"),
        "repository_url": code.get("repository_url"),
        "revision": code.get("revision"),
        "path": code.get("path"),
        "license": code.get("license"),
        "license_file_sha256": code.get("license_file_sha256"),
        "source_sha256": code.get("sha256"),
        "prefix": "".join(lines[:start]),
        "middle": "".join(lines[start:stop]),
        "suffix": "".join(lines[stop:]),
    }


def prepare_sft_data(
    code_data: Path,
    output: Path,
    *,
    instruction_data: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    code_manifest_path = code_data / "manifest.json"
    code_manifest = json.loads(code_manifest_path.read_text(encoding="utf-8"))
    if code_manifest.get("format_version") != 2:
        raise ValueError("Code corpus manifest version 2 is required")
    for split in ("train", "validation"):
        expected = code_manifest["streams"][split].get("jsonl_sha256")
        if not expected or file_sha256(code_data / f"{split}.jsonl") != expected:
            raise ValueError(f"{split} corpus JSONL fingerprint mismatch")
    sources = code_manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Code corpus manifest is missing source provenance")
    from ace.data.repositories import _validate_license

    code_repo_splits: dict[str, str] = {}
    for source in sources:
        _validate_license(source["license"])
        if len(source.get("resolved_commit", "")) != 40:
            raise ValueError("Every repository source must have a resolved commit")
        snapshot = code_data / source["license_snapshot"]
        if not snapshot.is_file() or file_sha256(snapshot) != source["license_file_sha256"]:
            raise ValueError(f"License snapshot fingerprint mismatch for {source['name']}")
        code_repo_splits[source["name"]] = source["split"]
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for split in splits:
        path = code_data / f"{split}.jsonl"
        for code in _read_jsonl(path):
            record = _fim_record(code, seed)
            if record is not None:
                splits[split].append(record)

    if instruction_data:
        instruction_splits: dict[str, str] = {}
        for record in _read_jsonl(instruction_data):
            split = record.get("split")
            if split not in splits:
                raise ValueError("Instruction records require split='train' or 'validation'")
            if record.get("kind", "instruction") != "instruction":
                raise ValueError("Instruction data may contain instruction records only")
            repository = record.get("repository")
            source = record.get("source")
            license_id = record.get("license")
            if not isinstance(repository, str) or not repository:
                raise ValueError("Instruction records require a repository/project identifier")
            if not isinstance(source, str) or not source:
                raise ValueError("Instruction records require source provenance")
            if license_id not in ALLOWED_LICENSES:
                raise ValueError("Instruction records require an allowlisted SPDX license")
            prior_split = instruction_splits.setdefault(repository, split)
            if prior_split != split:
                raise ValueError(f"Instruction project appears in both splits: {repository}")
            if repository in code_repo_splits and code_repo_splits[repository] != split:
                raise ValueError(f"Instruction project conflicts with code corpus split: {repository}")
            record["kind"] = "instruction"
            splits[split].append(record)

    if not all(splits.values()):
        raise ValueError("Both splits need supervised records")
    output.mkdir(parents=True)
    license_dir = code_data / "licenses"
    if license_dir.is_dir():
        import shutil

        shutil.copytree(license_dir, output / "licenses")
    import shutil

    shutil.copyfile(code_manifest_path, output / "source_code_manifest.json")
    output_counts = {}
    record_sources = {"repositories": sources, "instruction_sources": []}
    for split, records in splits.items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        output_counts[split] = len(records)
        instruction_rows = [
            record for selected in splits.values() for record in selected
            if record["kind"] == "instruction"
        ]
        record_sources["instruction_sources"] = sorted(
            {
                (record["source"], record["license"])
                for record in instruction_rows
            }
        )
        record_sources["instruction_sources"] = [
            {"source": source, "license": license_id}
            for source, license_id in record_sources["instruction_sources"]
        ]
    manifest = {
        "format_version": 1,
        "seed": seed,
        "source_code_manifest": {
            "path": "source_code_manifest.json",
            "sha256": file_sha256(output / "source_code_manifest.json"),
        },
        "instruction_data": {
            "path": str(instruction_data.resolve()),
            "sha256": file_sha256(instruction_data),
            "sources": sorted({record["source"] for records in splits.values() for record in records if record["kind"] == "instruction"}),
            "licenses": sorted({record["license"] for records in splits.values() for record in records if record["kind"] == "instruction"}),
        }
        if instruction_data
        else None,
        "records": output_counts,
        "sources": record_sources,
        "licenses": sorted(
            {source["license"] for source in sources}
            | {record["license"] for records in splits.values() for record in records if record["kind"] == "instruction"}
        ),
        "record_files": {
            split: file_sha256(output / f"{split}.jsonl") for split in output_counts
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-data", type=Path, required=True)
    parser.add_argument("--instruction-data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare_sft_data(args.code_data, args.output, instruction_data=args.instruction_data, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
