import json
import subprocess
from pathlib import Path

import pytest

from ace.data.repositories import _validate_license, prepare_repository_corpus
from ace.model.config import AceConfig


def _make_repo(path: Path, unique_text: str) -> None:
    (path / "src").mkdir(parents=True)
    (path / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    (path / "src" / "unique.py").write_text(unique_text, encoding="utf-8")
    (path / "src" / "shared.py").write_text("def shared():\n    return 42\n", encoding="utf-8")
    subprocess.run(["git", "init", "--quiet", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=ACE Test",
            "-c",
            "user.email=ace-test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "source",
        ],
        check=True,
    )


def test_license_allowlist_accepts_only_permissive_expressions():
    _validate_license("MIT OR Apache-2.0")
    with pytest.raises(ValueError, match="outside the permissive allowlist"):
        _validate_license("GPL-3.0-only")


def test_repository_split_isolation_and_license_provenance(tmp_path):
    training_repo = tmp_path / "training-repo"
    validation_repo = tmp_path / "validation-repo"
    _make_repo(training_repo, "def training_only():\n    return 'train'\n")
    _make_repo(validation_repo, "def validation_only():\n    return 'validation'\n")
    source_manifest = tmp_path / "sources.json"
    source_manifest.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "name": "training",
                        "url": str(training_repo),
                        "language": "python",
                        "license": "MIT",
                        "license_file": "LICENSE",
                        "split": "train",
                        "include": ["src/**/*.py"],
                    },
                    {
                        "name": "validation",
                        "url": str(validation_repo),
                        "language": "python",
                        "license": "MIT",
                        "license_file": "LICENSE",
                        "split": "validation",
                        "include": ["src/**/*.py"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    AceConfig(
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        n_layers=2,
        max_seq_len=64,
        vocab_size=300,
        moe_layers=[],
    ).to_yaml(config_path)

    output = tmp_path / "prepared"
    manifest = prepare_repository_corpus(source_manifest, output, config_path)

    train = [
        json.loads(line)
        for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    validation = [
        json.loads(line)
        for line in (output / "validation.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["repository"] for row in train} == {"training"}
    assert {row["repository"] for row in validation} == {"validation"}
    assert {row["sha256"] for row in train}.isdisjoint({row["sha256"] for row in validation})
    assert manifest["exact_duplicates_removed"] == 1
    assert all(len(source["resolved_commit"]) == 40 for source in manifest["sources"])
    assert all((output / source["license_snapshot"]).is_file() for source in manifest["sources"])
