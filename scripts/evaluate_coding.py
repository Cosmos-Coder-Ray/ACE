"""Evaluate completion and bug-fix tasks in a network-disabled Docker sandbox."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import psutil

from ace.data.preprocessing import file_sha256
from ace.inference.runtime import ModelRuntime
from ace.inference.sandbox import LANGUAGE_IMAGES, SandboxLimits, run_in_sandbox

LANGUAGES = ("python", "javascript", "typescript", "java", "rust", "c", "cpp")
TASK_KINDS = ("completion", "bugfix")
COMPLETION_MARKER = "{{ACE_COMPLETION}}"


def _read_tasks(path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            task = json.loads(line)
            if not isinstance(task, dict):
                raise ValueError(f"{path}:{line_number}: task must be a JSON object")
            tasks.append(task)
    if not tasks:
        raise ValueError("Evaluation suite is empty")
    seen: set[str] = set()
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for task in tasks:
        task_id = task.get("id")
        language = task.get("language")
        kind = task.get("kind")
        if not isinstance(task_id, str) or not task_id or task_id in seen:
            raise ValueError("Tasks must have unique nonempty IDs")
        seen.add(task_id)
        if language not in LANGUAGES or kind not in TASK_KINDS:
            raise ValueError(f"{task_id}: unsupported language or task kind")
        if not isinstance(task.get("prompt"), str) or not task["prompt"].strip():
            raise ValueError(f"{task_id}: prompt is required")
        if not isinstance(task.get("target_file"), str):
            raise ValueError(f"{task_id}: target_file is required")
        command = task.get("test_command")
        if not isinstance(command, list) or not command or not all(
            isinstance(part, str) and part for part in command
        ):
            raise ValueError(f"{task_id}: test_command must be a nonempty argument list")
        counts[(language, kind)] += 1
    short = [f"{language}/{kind}={counts[(language, kind)]}" for language in LANGUAGES for kind in TASK_KINDS if counts[(language, kind)] < 10]
    if short:
        raise ValueError("Suite needs at least 10 tasks for every language and kind: " + ", ".join(short))
    return tasks


def _safe_fixture(suite: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Fixture must be a suite-relative directory")
    root = suite.resolve(strict=True)
    fixture = (root / relative).resolve(strict=True)
    if not fixture.is_relative_to(root) or not fixture.is_dir():
        raise ValueError("Fixture must be a directory inside the evaluation suite")
    return fixture


def _target_path(fixture: Path, relative: str) -> Path:
    target = Path(relative)
    if target.is_absolute() or ".." in target.parts:
        raise ValueError("target_file must be relative to its fixture")
    root = fixture.resolve(strict=True)
    original = root / target
    if original.is_symlink():
        raise ValueError("Symlink targets are not supported")
    resolved = original.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("target_file must be an existing file inside its fixture")
    return resolved


def _clean_generated(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]) + ("\n" if text.endswith("\n") else "")
    return text


def _fixture_sha256(fixture: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(fixture.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        digest.update(path.relative_to(fixture).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


class _PeakMemory:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.peak = max(self.peak, self.process.memory_info().rss)
            except psutil.Error:
                return
            self.stop_event.wait(0.02)

    def __enter__(self) -> _PeakMemory:
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop_event.set()
        self.thread.join()


def evaluate_suite(
    model_path: Path,
    suite: Path,
    *,
    limits: SandboxLimits | None = None,
    threads: int = 2,
) -> dict[str, Any]:
    suite = suite.resolve(strict=True)
    tasks_path = suite / "tasks.jsonl"
    tasks = _read_tasks(tasks_path)
    runtime = ModelRuntime(model_path, threads=threads)
    by_language: dict[str, dict[str, Any]] = {
        language: {kind: {"total": 0, "passed": 0, "pass_rate": 0.0} for kind in TASK_KINDS}
        for language in LANGUAGES
    }
    losses: dict[str, dict[str, float]] = defaultdict(lambda: {"sum": 0.0, "tokens": 0})
    speeds: list[float] = []
    results: list[dict[str, Any]] = []
    selected_limits = limits or SandboxLimits()

    with _PeakMemory() as memory:
        for task in tasks:
            task_id = task["id"]
            language = task["language"]
            kind = task["kind"]
            fixture = _safe_fixture(suite, task["fixture"])
            target = _target_path(fixture, task["target_file"])
            source = target.read_text(encoding="utf-8")
            generation_prompt = task["prompt"]

            if kind == "completion":
                if source.count(COMPLETION_MARKER) != 1:
                    raise ValueError(f"{task_id}: completion fixture needs exactly one {COMPLETION_MARKER}")
                prefix, suffix = source.split(COMPLETION_MARKER)
                fim_prefix = task.get("fim_prefix", "<|fim_prefix|>")
                fim_suffix = task.get("fim_suffix", "<|fim_suffix|>")
                fim_middle = task.get("fim_middle", "<|fim_middle|>")
                generation_prompt += (
                    f"\n{fim_prefix}{prefix}{fim_suffix}{suffix}{fim_middle}"
                )
            else:
                generation_prompt += f"\n\nCurrent file ({task['target_file']}):\n{source}\n\nReturn the complete corrected file."

            generated = runtime.generate(generation_prompt, int(task.get("max_new_tokens", 256)))
            generated_text = _clean_generated(generated["text"])
            speeds.append(float(generated["tokens_per_second"]))
            if kind == "completion":
                replacement = f"{prefix}{generated_text}{suffix}"
            else:
                replacement = generated_text

            outcome = run_in_sandbox(
                fixture,
                language,
                task["test_command"],
                limits=selected_limits,
                image=task.get("image") or LANGUAGE_IMAGES[language],
                changes={task["target_file"]: replacement},
            )
            stats = by_language[language][kind]
            stats["total"] += 1
            stats["passed"] += int(bool(outcome["passed"]))
            stats["pass_rate"] = stats["passed"] / stats["total"]

            loss_info = None
            if isinstance(task.get("loss_prompt"), str) and isinstance(task.get("reference"), str):
                loss, token_count = runtime.score(task["loss_prompt"], task["reference"])
                losses[language]["sum"] += loss * token_count
                losses[language]["tokens"] += token_count
                loss_info = {"cross_entropy": loss, "tokens": token_count}
            results.append({
                "id": task_id,
                "language": language,
                "kind": kind,
                "passed": outcome["passed"],
                "status": outcome["status"],
                "generation_tokens_per_second": generated["tokens_per_second"],
                "generation_seconds": generated["seconds"],
                "sandbox": outcome,
                "loss": loss_info,
            })

    manifest = json.loads((model_path / "manifest.json").read_text(encoding="utf-8"))
    language_loss = {
        language: {
            "cross_entropy": values["sum"] / values["tokens"] if values["tokens"] else None,
            "tokens": values["tokens"],
        }
        for language, values in losses.items()
    }
    return {
        "format_version": 1,
        "model": {
            "path": str(model_path.resolve()),
            "track": runtime.track,
            "model_revision": runtime.model_revision,
            "tokenizer_revision": runtime.tokenizer_revision,
            "manifest_sha256": file_sha256(model_path / "manifest.json"),
        },
        "data": {
            "tasks_sha256": file_sha256(tasks_path),
            "task_count": len(tasks),
            "fixture_sha256": {task["id"]: _fixture_sha256(_safe_fixture(suite, task["fixture"])) for task in tasks},
        },
        "configuration": {"threads": threads, "sandbox": {
            "cpus": selected_limits.cpus, "memory": selected_limits.memory,
            "timeout_seconds": selected_limits.timeout_seconds, "network": "disabled",
        }},
        "task_results": results,
        "pass_rates": by_language,
        "language_loss": language_loss,
        "generation": {
            "mean_tokens_per_second": statistics.fmean(speeds) if speeds else None,
            "tasks_with_generation_speed": len(speeds),
        },
        "peak_cpu_memory_bytes": memory.peak,
        "quality_verified": all(
            by_language[language][kind]["total"] >= 10
            for language in LANGUAGES for kind in TASK_KINDS
        ) and all(
            result["status"] != "unavailable" for result in results
        ),
        "base_manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Native bundle or LoRA adapter output")
    parser.add_argument("--suite", type=Path, required=True, help="Directory containing tasks.jsonl and fixtures")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cpus", type=float, default=2.0)
    parser.add_argument("--memory", choices=["1g", "2g", "4g", "8g", "16g"], default="4g")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    report = evaluate_suite(
        args.model,
        args.suite,
        threads=args.threads,
        limits=SandboxLimits(args.cpus, args.memory, args.timeout_seconds),
    )
    if args.output.exists():
        raise FileExistsError("Choose a new evaluation output file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "quality_verified": report["quality_verified"], "pass_rates": report["pass_rates"]}, indent=2))


if __name__ == "__main__":
    main()
