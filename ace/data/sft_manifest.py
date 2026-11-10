"""Verify prepared supervised data before it reaches a training loop."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ace.data.preprocessing import file_sha256
from ace.data.repositories import _validate_license


def read_sft_manifest(data_dir: Path) -> dict[str, Any]:
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported supervised data manifest")
    for split in ("train", "validation"):
        filename = f"{split}.jsonl"
        expected = manifest.get("record_files", {}).get(split)
        if not expected or file_sha256(data_dir / filename) != expected:
            raise ValueError(f"{split} supervised data fingerprint mismatch")
    source_record = manifest.get("source_code_manifest", {})
    source_path = data_dir / source_record.get("path", "")
    if not source_path.is_file() or file_sha256(source_path) != source_record.get("sha256"):
        raise ValueError("Source corpus manifest fingerprint mismatch")
    sources = manifest.get("sources", {}).get("repositories")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Supervised data is missing repository provenance")
    for source in sources:
        _validate_license(source["license"])
        if not re.fullmatch(r"[0-9a-f]{40}", source.get("resolved_commit", "")):
            raise ValueError("Every supervised source must have a pinned commit")
        license_path = source.get("license_snapshot")
        if not isinstance(license_path, str):
            raise ValueError("Supervised source is missing its license snapshot")
        license_file = data_dir / license_path
        if not license_file.is_file() or file_sha256(license_file) != source["license_file_sha256"]:
            raise ValueError(f"License snapshot fingerprint mismatch for {source['name']}")
    instruction_sources = manifest.get("sources", {}).get("instruction_sources", [])
    if not isinstance(instruction_sources, list):
        raise ValueError("Invalid instruction source provenance")
    for source in instruction_sources:
        _validate_license(source["license"])
    allowed = manifest.get("licenses")
    if not isinstance(allowed, list) or not allowed:
        raise ValueError("Supervised data manifest has no license allowlist record")
    for license_id in allowed:
        _validate_license(license_id)
    return manifest
