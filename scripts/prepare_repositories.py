"""Prepare a repository-level split code corpus from a source manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ace.data.repositories import prepare_repository_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/ace_laptop.yaml"))
    parser.add_argument("--max-file-bytes", type=int, default=1_000_000)
    args = parser.parse_args()
    result = prepare_repository_corpus(
        args.sources, args.output, args.config, max_file_bytes=args.max_file_bytes
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
