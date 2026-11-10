"""Expand selected dense ACE FFNs into duplicated experts on CPU."""

import argparse
import json
from pathlib import Path

from ace.merging import upcycle_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, required=True, help="Zero-based indices")
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()
    metadata = upcycle_bundle(
        args.model, args.output, args.layers, num_experts=args.experts, top_k=args.top_k
    )
    print(json.dumps({"parameters": metadata["parameters"], "new_skills_trained": False}, indent=2))


if __name__ == "__main__":
    main()
