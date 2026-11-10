"""Prepare an explicit local UTF-8 or JSONL corpus without downloading data."""

import argparse
import json
from pathlib import Path

from ace.data.preprocessing import prepare_corpus
from ace.model.config import AceConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="configs/ace_laptop.yaml")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-license", required=True)
    parser.add_argument("--tokenizer", type=Path, help="Reuse the tokenizer of a base model")
    args = parser.parse_args()
    manifest = prepare_corpus(
        args.input,
        args.output,
        AceConfig.from_yaml(args.config),
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        source_license=args.source_license,
        tokenizer_path=args.tokenizer,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
