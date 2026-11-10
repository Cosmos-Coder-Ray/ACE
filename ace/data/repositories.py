"""Prepare code corpora from revision-pinned, permissively licensed repositories."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from ace.data.preprocessing import file_sha256
from ace.model.config import AceConfig
from ace.tokenizer.ace_tokenizer import AceTokenizer

ALLOWED_LICENSES = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC"}
LANGUAGE_EXTENSIONS = {
    "python": {".py"},
    "javascript": {".js", ".mjs", ".cjs"},
    "typescript": {".ts", ".tsx", ".mts", ".cts"},
    "java": {".java"},
    "rust": {".rs"},
    "c": {".c", ".h"},
    "cpp": {".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".hxx"},
}
SKIP_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "vendor",
    "third_party",
    "third-party",
    "dist",
    "build",
    "target",
    "out",
    "coverage",
    "generated",
    "__pycache__",
    "obj",
    "bin",
}


def _validate_license(expression: str) -> None:
    terms = re.split(r"\s+(?:OR|AND)\s+", expression.strip())
    if not terms or any(term not in ALLOWED_LICENSES for term in terms):
        raise ValueError(f"License is outside the permissive allowlist: {expression!r}")


def _validate_relative_path(value: str, name: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a repository-relative path")
    return path


def _clone_source(source: dict[str, Any], destination: Path) -> str:
    url = source.get("url")
    if not isinstance(url, str) or not url:
        raise ValueError("Every source requires a repository URL")
    if "://" in url:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"https", "ssh"}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Repository URLs must use HTTPS/SSH without embedded credentials")
    revision = source.get("revision")
    command = ["git", "clone", "--filter=blob:none", "--no-tags", "--no-checkout", "--", url]
    command.append(str(destination))
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=900)
        if revision:
            subprocess.run(
                ["git", "-C", str(destination), "fetch", "--depth=1", "origin", str(revision)],
                check=True,
                capture_output=True,
                text=True,
                timeout=900,
            )
            checkout = "FETCH_HEAD"
        else:
            checkout = "HEAD"
        subprocess.run(
            ["git", "-C", str(destination), "checkout", "--detach", checkout],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        result = subprocess.run(
            ["git", "-C", str(destination), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Timed out retrieving {source.get('name', url)}") from exc
    except subprocess.CalledProcessError as exc:
        # Avoid echoing the URL or environment details that could contain credentials.
        raise RuntimeError(f"Could not retrieve source {source.get('name', url)}") from exc
    return result.stdout.strip()


def _matches(relative_path: str, patterns: list[str]) -> bool:
    path = PurePosixPath(relative_path)
    return any(fnmatch.fnmatchcase(relative_path, pattern) or path.match(pattern) for pattern in patterns)


def _collect_files(
    root: Path, source: dict[str, Any], commit: str, max_file_bytes: int
) -> list[dict[str, Any]]:
    name = source.get("name")
    language = source.get("language")
    split = source.get("split")
    license_expression = source.get("license")
    if not isinstance(name, str) or not name or language not in LANGUAGE_EXTENSIONS:
        raise ValueError("Each source requires a name and supported language")
    if split not in {"train", "validation"}:
        raise ValueError("Each repository split must be 'train' or 'validation'")
    if not isinstance(license_expression, str):
        raise ValueError("Each source requires an SPDX license expression")
    _validate_license(license_expression)
    license_file = _validate_relative_path(source.get("license_file", "LICENSE"), "license_file")
    license_path = root / license_file
    if not license_path.is_file():
        raise ValueError(f"Missing declared license file in {name}: {license_file}")
    license_hash = file_sha256(license_path)
    include = source.get("include", [])
    exclude = source.get("exclude", [])
    if not isinstance(include, list) or not all(isinstance(item, str) for item in include):
        raise ValueError(f"Source include must be a list of path patterns: {name}")
    if not isinstance(exclude, list) or not all(isinstance(item, str) for item in exclude):
        raise ValueError(f"Source exclude must be a list of path patterns: {name}")

    collected: list[dict[str, Any]] = []
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories[:] = [
            directory
            for directory in directories
            if directory not in SKIP_DIRECTORIES and not directory.startswith(".")
        ]
        for filename in filenames:
            path = current_path / filename
            if path.is_symlink() or path.suffix.lower() not in LANGUAGE_EXTENSIONS[language]:
                continue
            relative_path = path.relative_to(root).as_posix()
            if include and not _matches(relative_path, include):
                continue
            if _matches(relative_path, exclude):
                continue
            if path.stat().st_size == 0 or path.stat().st_size > max_file_bytes:
                continue
            try:
                raw = path.read_bytes()
                text = raw.decode("utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if not text.strip():
                continue
            digest = hashlib.sha256(raw).hexdigest()
            collected.append(
                {
                    "sha256": digest,
                    "_local_path": str(path),
                    "path": relative_path,
                    "repository": name,
                    "repository_url": source["url"],
                    "revision": commit,
                    "language": language,
                    "split": split,
                    "license": license_expression,
                    "license_file_sha256": license_hash,
                }
            )
    return collected


def prepare_repository_corpus(
    source_manifest: Path,
    output: Path,
    config_path: Path,
    *,
    max_file_bytes: int = 1_000_000,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if max_file_bytes < 1:
        raise ValueError("max_file_bytes must be positive")
    manifest_input = json.loads(source_manifest.read_text(encoding="utf-8"))
    sources = manifest_input.get("sources") if isinstance(manifest_input, dict) else None
    if not isinstance(sources, list) or not sources:
        raise ValueError("Source manifest must contain a nonempty sources list")
    config = AceConfig.from_yaml(config_path)

    source_records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    duplicate_count = 0
    repository_splits: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="ace-corpus-") as temp_dir:
        temp_root = Path(temp_dir)
        for index, source in enumerate(sources):
            if not isinstance(source, dict):
                raise ValueError("Source entries must be JSON objects")
            name = source.get("name", f"source-{index}")
            source_url = source.get("url")
            if isinstance(source_url, str):
                repository_key = re.sub(r"\.git$", "", source_url.rstrip("/")).lower()
                old_split = repository_splits.get(repository_key)
                if old_split is not None and old_split != source.get("split"):
                    raise ValueError("A repository cannot appear in multiple data splits")
                repository_splits[repository_key] = source.get("split")
            repo_path = temp_root / f"repo-{index}"
            commit = _clone_source(source, repo_path)
            files = _collect_files(repo_path, source, commit, max_file_bytes)
            license_relative = _validate_relative_path(
                source.get("license_file", "LICENSE"), "license_file"
            )
            license_snapshot = temp_root / f"license-{index}.txt"
            shutil.copyfile(repo_path / license_relative, license_snapshot)
            source_records.append(
                {
                    "name": name,
                    "url": source["url"],
                    "requested_revision": source.get("revision"),
                    "resolved_commit": commit,
                    "language": source["language"],
                    "split": source["split"],
                    "license": source["license"],
                    "license_file": source.get("license_file", "LICENSE"),
                    "license_file_sha256": files[0]["license_file_sha256"]
                    if files
                    else file_sha256(repo_path / license_relative),
                    "_license_snapshot": str(license_snapshot),
                    "files_selected": len(files),
                }
            )
            candidates.extend(files)

        # Evaluation copies win exact cross-split duplicates so no training
        # example can reveal a held-out file verbatim.
        priority = {"train": 0, "validation": 1}
        unique: dict[str, dict[str, Any]] = {}
        for record in candidates:
            old = unique.get(record["sha256"])
            if old is None:
                unique[record["sha256"]] = record
                continue
            duplicate_count += 1
            if priority[record["split"]] > priority[old["split"]]:
                unique[record["sha256"]] = record

        selected = list(unique.values())
        counts = {split: 0 for split in ("train", "validation")}
        language_counts = {split: {} for split in counts}
        with tempfile.TemporaryDirectory(prefix="ace-tokenizer-") as tokenizer_dir:
            tokenizer_corpus = Path(tokenizer_dir) / "train.txt"
            for split in counts:
                split_records = [record for record in selected if record["split"] == split]
                if not split_records:
                    raise ValueError(f"No unique documents assigned to {split}")
                counts[split] = len(split_records)
                language_counts[split] = {}
                for record in split_records:
                    language = record["language"]
                    language_counts[split][language] = (
                        language_counts[split].get(language, 0) + 1
                    )

            with tokenizer_corpus.open("w", encoding="utf-8", newline="") as stream:
                for record in selected:
                    if record["split"] == "train":
                        text = Path(record["_local_path"]).read_text(encoding="utf-8")
                        stream.write(text + "\n")
            tokenizer = AceTokenizer.train([tokenizer_corpus], vocab_size=config.vocab_size)
            actual_config = replace(config, vocab_size=tokenizer.vocab_size)

            output.mkdir(parents=True, exist_ok=False)
            licenses_dir = output / "licenses"
            licenses_dir.mkdir()
            for index, source_record in enumerate(source_records):
                filename = (
                    f"{index:03d}-{re.sub(r'[^A-Za-z0-9._-]', '_', source_record['name'])}.txt"
                )
                shutil.copyfile(source_record.pop("_license_snapshot"), licenses_dir / filename)
                source_record["license_snapshot"] = f"licenses/{filename}"
            tokenizer.save(output / "tokenizer")
            actual_config.to_yaml(output / "config.yaml")
            shutil.copyfile(tokenizer_corpus, output / "tokenizer_corpus.txt")
            stream_info: dict[str, Any] = {}
            for split in counts:
                stream_path = output / f"{split}.bin"
                token_count = 0
                jsonl_path = output / f"{split}.jsonl"
                split_records = [record for record in selected if record["split"] == split]
                with stream_path.open("wb") as binary, jsonl_path.open(
                    "w", encoding="utf-8"
                ) as jsonl:
                    for record in split_records:
                        text = Path(record.pop("_local_path")).read_text(encoding="utf-8")
                        ids = tokenizer.encode(text, add_special_tokens=True)
                        np.asarray(ids, dtype="<u4").tofile(binary)
                        token_count += len(ids)
                        jsonl.write(
                            json.dumps({**record, "text": text}, ensure_ascii=False) + "\n"
                        )
                stream_info[split] = {
                    "tokens": token_count,
                    "sha256": file_sha256(stream_path),
                    "jsonl_sha256": file_sha256(jsonl_path),
                }

    manifest = {
        "format_version": 2,
        "dtype": "<u4",
        "documents": counts,
        "exact_duplicates_removed": duplicate_count,
        "language_documents": language_counts,
        "streams": stream_info,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_sha256": file_sha256(output / "tokenizer" / "tokenizer.json"),
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": file_sha256(source_manifest),
        "source_license": "Per-source; see sources",
        "sources": source_records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
