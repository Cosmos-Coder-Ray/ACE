"""Run coding fixtures in a resource-bounded, network-disabled container."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import uuid
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


LANGUAGE_IMAGES = {
    "python": "python:3.12-slim",
    "javascript": "node:22-slim",
    "typescript": "denoland/deno:2.1.4",
    "java": "eclipse-temurin:21-jdk",
    "rust": "rust:1.85-slim",
    "c": "gcc:14.2.0",
    "cpp": "gcc:14.2.0",
}


@dataclass(frozen=True)
class SandboxLimits:
    cpus: float = 2.0
    memory: str = "4g"
    timeout_seconds: int = 300

    def __post_init__(self) -> None:
        if not 0 < self.cpus <= 64 or not 0 < self.timeout_seconds <= 3600:
            raise ValueError("CPU limit must be <= 64 and timeout must be <= 3600 seconds")
        if self.memory not in {"1g", "2g", "4g", "8g", "16g"}:
            raise ValueError("memory must be one of 1g, 2g, 4g, 8g, or 16g")


def _copy_fixture(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Fixture directory does not exist: {source}")
    skipped = {
        ".git", ".venv", "venv", "node_modules", "vendor", "target", "dist", "build",
        "coverage", "__pycache__", "checkpoints", "weights", "models",
    }
    for current, directories, filenames in os.walk(source, followlinks=False):
        current_path = Path(current)
        kept_directories = [name for name in directories if name not in skipped]
        if any((current_path / name).is_symlink() for name in kept_directories + filenames):
            raise ValueError("Symlinks are not allowed in coding fixtures")
        directories[:] = kept_directories
    total_bytes = 0
    for current, directories, filenames in os.walk(source, followlinks=False):
        directories[:] = [name for name in directories if name not in {
            ".git", ".venv", "venv", "node_modules", "vendor", "target", "dist", "build",
            "coverage", "__pycache__", "checkpoints", "weights", "models",
        }]
        current_path = Path(current)
        for filename in filenames:
            path = current_path / filename
            if filename.startswith(".env") or path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}:
                continue
            total_bytes += path.stat().st_size
            if total_bytes > 256 * 1024 * 1024:
                raise ValueError("Workspace scratch copy exceeds the 256 MiB size limit")
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            ".git", ".venv", "venv", "node_modules", "vendor", "target", "dist", "build",
            "coverage", "__pycache__", "checkpoints", "weights", "models", ".env*",
            "*.pem", "*.key", "*.p12", "*.pfx", "id_rsa*",
        ),
        symlinks=False,
    )
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlinks are not allowed in coding fixtures")


def _missing_dependency(output: str) -> str | None:
    indicators = (
        ("No module named ", "Python test dependency is not installed in the container"),
        ("Cannot find module", "JavaScript/TypeScript dependency is not installed in the container"),
        ("ModuleNotFoundError", "A test dependency is not installed in the container"),
        ("command not found", "A test command or compiler is not installed in the container"),
        ("not found", "A test command or compiler is not installed in the container"),
        ("can't find crate", "A Rust test dependency is not available in the container"),
    )
    for needle, reason in indicators:
        if needle.lower() in output.lower():
            return reason
    return None


def run_in_sandbox(
    fixture: Path,
    language: str,
    command: Sequence[str],
    *,
    limits: SandboxLimits | None = None,
    image: str | None = None,
    changes: dict[str, str] | None = None,
) -> dict[str, object]:
    if language not in LANGUAGE_IMAGES:
        raise ValueError(f"Unsupported sandbox language: {language}")
    if not command or not all(isinstance(part, str) and part for part in command):
        raise ValueError("Sandbox command must be a nonempty argument list")
    if any(part in {"/", ".."} for part in command):
        raise ValueError("Sandbox command contains a forbidden path component")
    if not shutil.which("docker"):
        return {
            "status": "unavailable",
            "passed": False,
            "reason": "Docker is not installed or is not on PATH",
            "exit_code": None,
            "duration_seconds": 0.0,
        }

    selected_limits = limits or SandboxLimits()
    selected_image = image or LANGUAGE_IMAGES[language]
    if not isinstance(selected_image, str) or len(selected_image) > 256 or not re.fullmatch(
        r"[A-Za-z0-9._/:@-]+", selected_image
    ):
        raise ValueError("Container image must be a valid image reference")
    with tempfile.TemporaryDirectory(prefix="ace-test-") as temp_dir:
        scratch = Path(temp_dir) / "workspace"
        _copy_fixture(fixture, scratch)
        for relative, contents in (changes or {}).items():
            target = Path(relative)
            if target.is_absolute() or ".." in target.parts:
                raise ValueError("Changed files must stay inside the fixture")
            output = (scratch / target).resolve()
            if not output.is_relative_to(scratch.resolve()) or not output.is_file():
                raise ValueError("Changed file must already exist in the fixture")
            output.write_text(contents, encoding="utf-8")
        container_name = f"ace-eval-{uuid.uuid4().hex[:16]}"
        docker_command = [
            "docker", "run", "--rm", "--name", container_name, "--network=none", "--cpus", str(selected_limits.cpus),
            "--memory", selected_limits.memory,
            "--memory-swap", selected_limits.memory,
            "--pids-limit", "256", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--read-only", "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m", "--user", "1000:1000",
            "--mount", f"type=bind,source={scratch.resolve()},target=/workspace,rw",
            "--workdir", "/workspace", selected_image, *command,
        ]
        started = time.monotonic()
        try:
            result = subprocess.run(
                docker_command,
                capture_output=True,
                text=True,
                timeout=selected_limits.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            subprocess.run(["docker", "kill", container_name], capture_output=True, timeout=10, check=False)
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=10, check=False)
            return {
                "status": "timeout", "passed": False,
                "reason": f"Exceeded {selected_limits.timeout_seconds} seconds",
                "exit_code": None,
                "stdout": (exc.stdout or "")[-4000:], "stderr": (exc.stderr or "")[-4000:],
                "duration_seconds": time.monotonic() - started,
            }
        except OSError as exc:
            return {
                "status": "unavailable", "passed": False, "reason": str(exc),
                "exit_code": None, "duration_seconds": time.monotonic() - started,
            }
        missing_dependency = _missing_dependency(result.stdout + "\n" + result.stderr)
        status = "passed" if result.returncode == 0 else ("unavailable" if missing_dependency else "failed")
        return {
            "status": status,
            "passed": result.returncode == 0,
            "reason": missing_dependency,
            "exit_code": result.returncode,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
            "duration_seconds": time.monotonic() - started,
            "image": selected_image,
            "limits": {
                "cpus": selected_limits.cpus,
                "memory": selected_limits.memory,
                "timeout_seconds": selected_limits.timeout_seconds,
                "network": "disabled",
            },
        }
