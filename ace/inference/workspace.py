"""Bounded, read-only retrieval from a local project workspace."""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path

SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "vendor",
    "third_party", "dist", "build", "target", "out", "coverage", "__pycache__",
    ".next", ".cache", "checkpoints", "weights", "models",
}
SECRET_NAMES = {".env", ".npmrc", ".pypirc", "credentials", "secrets.json"}
SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}


def _ignored(relative: str, directory: bool = False) -> bool:
    parts = Path(relative).parts
    if any(part in SKIP_DIRS for part in parts):
        return True
    name = parts[-1].lower() if parts else ""
    if name in SECRET_NAMES or name.startswith(".env"):
        return True
    if Path(name).suffix.lower() in SECRET_SUFFIXES or name.startswith("id_rsa"):
        return True
    if directory and name.startswith("."):
        return True
    return False


def _terms(text: str) -> set[str]:
    return {term.lower() for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)}


def _git_ignored(root: Path, relative_paths: list[str]) -> set[str]:
    git = shutil.which("git")
    if not git or not relative_paths:
        return set()
    try:
        result = subprocess.run(
            [git, "-C", str(root), "check-ignore", "--no-index", "-z", "--stdin"],
            input="\0".join(relative_paths).encode("utf-8"),
            capture_output=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    return set(result.stdout.decode("utf-8", errors="replace").split("\0"))


def workspace_file(workspace: Path, relative: str, *, max_bytes: int = 256_000) -> str:
    root = workspace.resolve(strict=True)
    candidate = (root / relative).resolve(strict=True)
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError("File must resolve to a regular file inside the workspace")
    normalized = candidate.relative_to(root).as_posix()
    if (
        _ignored(normalized)
        or normalized in _git_ignored(root, [normalized])
        or candidate.stat().st_size > max_bytes
    ):
        raise ValueError("File is ignored or exceeds the workspace file-size limit")
    try:
        return candidate.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        raise ValueError("File is not readable UTF-8 text") from exc


def retrieve_files(
    workspace: Path,
    query: str,
    *,
    max_files: int = 8,
    max_file_bytes: int = 64_000,
    max_total_bytes: int = 256_000,
) -> list[tuple[str, str]]:
    root = workspace.resolve(strict=True)
    query_terms = _terms(query)
    candidates: list[tuple[int, str, Path]] = []
    scanned = 0
    scanned_bytes = 0
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories[:] = [
            entry for entry in directories
            if not _ignored((current_path / entry).relative_to(root).as_posix(), True)
        ]
        for filename in filenames:
            path = current_path / filename
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or _ignored(relative) or path.stat().st_size > max_file_bytes:
                continue
            if path.suffix.lower() not in {
                ".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".java",
                ".rs", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".go", ".md",
                ".toml", ".yaml", ".yml", ".json",
            }:
                continue
            if scanned >= 1000 or scanned_bytes + path.stat().st_size > 64 * 1024 * 1024:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            scanned += 1
            scanned_bytes += path.stat().st_size
            name_terms = _terms(relative)
            score = len(query_terms & name_terms) * 4 + len(query_terms & _terms(content))
            if score:
                candidates.append((score, relative, path))
    ignored = _git_ignored(root, [item[1] for item in candidates])
    candidates = [item for item in candidates if item[1] not in ignored]
    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected: list[tuple[str, str]] = []
    used = 0
    for _, relative, path in candidates:
        raw = path.read_bytes()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if used + len(raw) > max_total_bytes:
            continue
        selected.append((relative, content))
        used += len(raw)
        if len(selected) == max_files:
            break
    return selected
