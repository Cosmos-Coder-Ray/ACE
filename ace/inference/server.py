"""Loopback-only coding assistant service."""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ace.inference.runtime import ModelRuntime
from ace.inference.sandbox import LANGUAGE_IMAGES, SandboxLimits, run_in_sandbox
from ace.inference.workspace import retrieve_files, workspace_file

MAX_BODY_BYTES = 128_000


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]) + ("\n" if text.endswith("\n") else "")
    return text


def build_handler(runtime: ModelRuntime, workspace: Path, token: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ACE/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            logging.info("%s - %s", self.address_string(), format % args)

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if secrets.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}"):
                return True
            self._json(401, {"error": "unauthorized"})
            return False

        def do_GET(self) -> None:
            if urlparse(self.path).path != "/health":
                self._json(404, {"error": "not found"})
                return
            if self._authorized():
                self._json(200, {
                    "status": "ready", "track": runtime.track,
                    "model_revision": runtime.model_revision,
                    "tokenizer_revision": runtime.tokenizer_revision,
                })

        def do_POST(self) -> None:
            if not self._authorized():
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > MAX_BODY_BYTES:
                    raise ValueError("Request body must be between 1 byte and 128 KB")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Request body must be a JSON object")
                endpoint = urlparse(self.path).path
                if endpoint == "/completion":
                    result = runtime.generate(data.get("prompt"), int(data.get("max_new_tokens", 128)))
                    self._json(200, result)
                    return
                if endpoint == "/patch":
                    result = self._patch(data)
                    self._json(200, result)
                    return
                self._json(404, {"error": "not found"})
            except (ValueError, TypeError, KeyError, FileNotFoundError) as exc:
                self._json(400, {"error": str(exc)})
            except Exception:
                logging.exception("Assistant request failed")
                self._json(500, {"error": "generation failed; inspect the local service log"})

        def _patch(self, data: dict[str, Any]) -> dict[str, Any]:
            relative = data.get("file")
            instructions = data.get("instructions")
            if not isinstance(relative, str) or not isinstance(instructions, str) or not instructions.strip():
                raise ValueError("patch requires file and nonempty instructions")
            original = workspace_file(workspace, relative)
            context = retrieve_files(workspace, instructions, max_files=6)
            context_text = "\n\n".join(
                f"--- {name} ---\n{content}" for name, content in context if name != relative
            )
            prompt = (
                "You are editing one file in a software project. Return only the complete replacement "
                "contents for the target file. Do not use Markdown fences or explain your answer.\n"
                f"Task: {instructions}\nTarget file: {relative}\n\n"
                f"Relevant project files:\n{context_text or '(none)'}\n\n"
                f"Current target file:\n{original}\n\nReplacement file contents:"
            )
            generated = runtime.generate(prompt, int(data.get("max_new_tokens", 512)))
            replacement = _strip_fence(generated["text"])
            patch = "".join(difflib.unified_diff(
                original.splitlines(keepends=True), replacement.splitlines(keepends=True),
                fromfile=f"a/{relative}", tofile=f"b/{relative}",
            ))
            tests = self._run_tests(relative, replacement)
            return {**generated, "file": relative, "diff": patch, "replacement": replacement, "tests": tests}

        def _run_tests(self, relative: str, replacement: str) -> dict[str, Any]:
            test_config_path = workspace / "ace-tests.json"
            if test_config_path.is_symlink() or not test_config_path.is_file():
                return {
                    "status": "unavailable", "passed": False,
                    "reason": "Add ace-tests.json with a language and test_command to run project tests",
                }
            if test_config_path.stat().st_size > 32_000:
                raise ValueError("ace-tests.json exceeds the 32 KB limit")
            config = json.loads(test_config_path.read_text(encoding="utf-8"))
            language = config.get("language")
            command = config.get("test_command")
            if language not in LANGUAGE_IMAGES or not isinstance(command, list):
                raise ValueError("ace-tests.json requires a supported language and test_command array")
            return run_in_sandbox(
                workspace,
                language,
                command,
                limits=SandboxLimits(
                    cpus=float(config.get("cpus", 2)),
                    memory=str(config.get("memory", "4g")),
                    timeout_seconds=int(config.get("timeout_seconds", 300)),
                ),
                image=config.get("image"),
                changes={relative: replacement},
            )

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Native bundle or pretrained adapter directory")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The assistant service must bind to a loopback address")
    workspace = args.workspace.resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError("workspace must be a directory")
    runtime = ModelRuntime(args.model, threads=args.threads)
    token = secrets.token_urlsafe(32)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(runtime, workspace, token))
    server.daemon_threads = True
    print(json.dumps({"url": f"http://{args.host}:{server.server_port}", "token": token}))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
