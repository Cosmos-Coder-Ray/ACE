"""Create the versioned starter suite of executable coding tasks for seven languages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

OPERATIONS = [
    ("add", "Add the two integers", "a, b", "a + b", [(2, 3, 5), (-4, 9, 5), (0, 0, 0)]),
    ("subtract", "Subtract the second integer from the first", "a, b", "a - b", [(8, 3, 5), (2, 7, -5), (-3, -2, -1)]),
    ("multiply", "Multiply the two integers", "a, b", "a * b", [(3, 4, 12), (-2, 5, -10), (0, 7, 0)]),
    ("maximum", "Return the larger integer", "a, b", "a > b ? a : b", [(3, 8, 8), (9, -1, 9), (4, 4, 4)]),
    ("minimum", "Return the smaller integer", "a, b", "a < b ? a : b", [(3, 8, 3), (9, -1, -1), (4, 4, 4)]),
    ("absolute", "Return the nonnegative magnitude of the integer", "a", "a < 0 ? -a : a", [(-7, 7), (0, 0), (5, 5)]),
    ("is_even", "Return 1 for an even integer and 0 for an odd integer", "a", "a % 2 == 0 ? 1 : 0", [(4, 1), (7, 0), (0, 1)]),
    ("square", "Return the integer multiplied by itself", "a", "a * a", [(4, 16), (-3, 9), (0, 0)]),
    ("clamp", "Clamp value to the inclusive range from low to high", "value, low, high", "value < low ? low : (value > high ? high : value)", [(2, 4, 9, 4), (12, 4, 9, 9), (6, 4, 9, 6)]),
    ("sum_three", "Return the sum of all three integers", "a, b, c", "a + b + c", [(2, 3, 4, 9), (-2, 3, 5, 6), (0, 0, 0, 0)]),
]

LANGUAGES = ("python", "javascript", "typescript", "java", "rust", "c", "cpp")
MARKER = "{{ACE_COMPLETION}}"


def _body(operation: tuple[Any, ...], language: str, buggy: bool) -> str:
    name = operation[0]
    expected = {
        "python": {
            "add": "a + b", "subtract": "a - b", "multiply": "a * b",
            "maximum": "a if a > b else b", "minimum": "a if a < b else b",
            "absolute": "-a if a < 0 else a", "is_even": "1 if a % 2 == 0 else 0",
            "square": "a * a", "clamp": "low if value < low else (high if value > high else value)",
            "sum_three": "a + b + c",
        },
        "rust": {
            "add": "a + b", "subtract": "a - b", "multiply": "a * b",
            "maximum": "if a > b { a } else { b }", "minimum": "if a < b { a } else { b }",
            "absolute": "if a < 0 { -a } else { a }", "is_even": "if a % 2 == 0 { 1 } else { 0 }",
            "square": "a * a", "clamp": "if value < low { low } else if value > high { high } else { value }",
            "sum_three": "a + b + c",
        },
    }.get(language, {}).get(name, operation[3])
    if buggy:
        wrong_common = {
            "add": "a - b", "subtract": "a + b", "multiply": "a + b",
            "maximum": "a < b ? a : b", "minimum": "a > b ? a : b",
            "absolute": "a", "is_even": "a % 2 != 0 ? 1 : 0",
            "square": "a", "clamp": "value", "sum_three": "a + b",
        }[name]
        wrong = wrong_common
        if language == "python":
            wrong = {
                "maximum": "a if a < b else b", "minimum": "a if a > b else b",
                "is_even": "1 if a % 2 != 0 else 0",
            }.get(name, wrong_common)
        elif language == "rust":
            wrong = {
                "maximum": "if a < b { a } else { b }", "minimum": "if a > b { a } else { b }",
                "is_even": "if a % 2 != 0 { 1 } else { 0 }",
            }.get(name, wrong_common)
    else:
        wrong = expected
    if language == "python":
        return f"return {wrong}"
    if language in {"javascript", "typescript"}:
        return f"return {wrong};"
    if language == "rust":
        return f"{wrong}"
    return f"return {wrong};"


def _source(language: str, function: str, args: str, body: str) -> str:
    typed_args = args
    if language == "typescript":
        typed_args = ", ".join(f"{item.strip()}: number" for item in args.split(","))
    if language == "python":
        return f"def {function}({args}):\n    {body}\n"
    if language == "javascript":
        return f"function {function}({args}) {{\n    {body}\n}}\n\nmodule.exports = {{ {function} }};\n"
    if language == "typescript":
        return f"export function {function}({typed_args}): number {{\n    {body}\n}}\n"
    if language == "java":
        return f"public final class Solution {{\n    public static int {function}({', '.join('int ' + part.strip() for part in args.split(','))}) {{\n        {body}\n    }}\n}}\n"
    if language == "rust":
        return f"pub fn {function}({', '.join(part.strip() + ': i32' for part in args.split(','))}) -> i32 {{\n    {body}\n}}\n"
    if language == "c":
        return f"int {function}({', '.join('int ' + part.strip() for part in args.split(','))}) {{\n    {body}\n}}\n"
    return f"int {function}({', '.join('int ' + part.strip() for part in args.split(','))}) {{\n    {body}\n}}\n"


def _test_source(language: str, operation: tuple[Any, ...]) -> str:
    name, _, args, _, cases = operation
    calls = [f"{name}({', '.join(str(value) for value in case[:-1])})" for case in cases]
    expected = [case[-1] for case in cases]
    if language == "python":
        return "from solution import " + name + "\n\n" + "\n".join(
            f"assert {call} == {answer}, ({call}, {answer})" for call, answer in zip(calls, expected)
        ) + "\n"
    if language == "javascript":
        return "const { " + name + " } = require('./solution');\n" + "\n".join(
            f"if ({call} !== {answer}) throw new Error('{call} failed');" for call, answer in zip(calls, expected)
        ) + "\n"
    if language == "typescript":
        return f"import {{ {name} }} from './solution.ts';\n" + "\n".join(
            f"if ({call} !== {answer}) throw new Error('{call} failed');" for call, answer in zip(calls, expected)
        ) + "\n"
    if language == "java":
        return "public final class Test {\n    public static void main(String[] args) {\n" + "\n".join(
            f"        if (Solution.{call} != {answer}) throw new AssertionError(\"{call} failed\");" for call, answer in zip(calls, expected)
        ) + "\n    }\n}\n"
    if language == "rust":
        return "\n#[cfg(test)]\nmod tests {\n    use super::*;\n    #[test]\n    fn behavior() {\n" + "\n".join(
            f"        assert_eq!({call}, {answer});" for call, answer in zip(calls, expected)
        ) + "\n    }\n}\n"
    if language == "c":
        declaration = f"int {name}({', '.join('int ' + part.strip() for part in args.split(','))});"
        return "#include <assert.h>\n" + declaration + "\nint main(void) {\n" + "\n".join(
            f"    assert({call} == {answer});" for call, answer in zip(calls, expected)
        ) + "\n    return 0;\n}\n"
    declaration = f"int {name}({', '.join('int ' + part.strip() for part in args.split(','))});"
    return "#include <cassert>\n" + declaration + "\nint main() {\n" + "\n".join(
        f"    assert({call} == {answer});" for call, answer in zip(calls, expected)
    ) + "\n    return 0;\n}\n"


def _runner(language: str, source_path: str) -> list[str]:
    return {
        "python": ["python", "tests.py"],
        "javascript": ["node", "tests.js"],
        "typescript": ["deno", "run", "tests.ts"],
        "java": ["sh", "-c", "javac Solution.java Test.java && java Test"],
        "rust": ["sh", "-c", "rustc --edition=2021 --test solution.rs -o /tmp/ace-test && /tmp/ace-test"],
        "c": ["sh", "-c", "gcc -std=c11 -O2 solution.c tests.c -o /tmp/ace-test && /tmp/ace-test"],
        "cpp": ["sh", "-c", "g++ -std=c++17 -O2 solution.cpp tests.cpp -o /tmp/ace-test && /tmp/ace-test"],
    }[language]


def create_suite(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Choose a new evaluation suite directory: {output}")
    output.mkdir(parents=True)
    tasks: list[dict[str, Any]] = []
    extensions = {"python": "py", "javascript": "js", "typescript": "ts", "java": "java", "rust": "rs", "c": "c", "cpp": "cpp"}
    test_names = {"python": "tests.py", "javascript": "tests.js", "typescript": "tests.ts", "java": "Test.java", "rust": None, "c": "tests.c", "cpp": "tests.cpp"}

    for language in LANGUAGES:
        for kind in ("completion", "bugfix"):
            for operation in OPERATIONS:
                name, description, args, _, _ = operation
                task_id = f"{language}-{kind}-{name}"
                fixture_relative = Path("fixtures") / task_id
                fixture = output / fixture_relative
                fixture.mkdir(parents=True)
                expected_body = _body(operation, language, False)
                body = MARKER if kind == "completion" else _body(operation, language, True)
                source = _source(language, name, args, body)
                source_name = f"solution.{extensions[language]}"
                source_path = fixture / source_name
                source_path.write_text(source, encoding="utf-8")
                test_name = test_names[language]
                test_source = _test_source(language, operation)
                if language == "rust":
                    source_path.write_text(source + test_source, encoding="utf-8")
                    test_name = source_name
                else:
                    (fixture / str(test_name)).write_text(test_source, encoding="utf-8")

                prefix, suffix = source.split(MARKER) if kind == "completion" else _source(
                    language, name, args, MARKER
                ).split(MARKER)
                language_instruction = (
                    f"Complete {name}({args}) to {description.lower()}. Return only the missing function body."
                    if kind == "completion"
                    else f"Fix the bug in {name} so it will {description.lower()}. Return the complete corrected file."
                )
                task: dict[str, Any] = {
                    "id": task_id,
                    "language": language,
                    "kind": kind,
                    "fixture": fixture_relative.as_posix(),
                    "target_file": source_name,
                    "prompt": language_instruction,
                    "test_command": _runner(language, source_name),
                    "max_new_tokens": 256 if kind == "completion" else 512,
                    "loss_prompt": f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>",
                    "reference": expected_body,
                }
                tasks.append(task)

    with (output / "tasks.jsonl").open("w", encoding="utf-8") as stream:
        for task in tasks:
            stream.write(json.dumps(task, ensure_ascii=False) + "\n")
    manifest = {
        "format_version": 1,
        "suite": "ACE starter coding tasks",
        "description": "Compile/test-based unit tasks for code completion and bug fixing. Expand with project-level held-out tasks before using scores as a broad capability claim.",
        "languages": list(LANGUAGES),
        "task_kinds": ["completion", "bugfix"],
        "tasks_per_language_and_kind": 10,
        "task_count": len(tasks),
        "test_images": {
            "python": "python:3.12-slim", "javascript": "node:22-slim",
            "typescript": "denoland/deno:2.1.4", "java": "eclipse-temurin:21-jdk",
            "rust": "rust:1.85-slim", "c": "gcc:14.2.0", "cpp": "gcc:14.2.0",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evaluation/coding_tasks"))
    args = parser.parse_args()
    print(json.dumps(create_suite(args.output), indent=2))


if __name__ == "__main__":
    main()
