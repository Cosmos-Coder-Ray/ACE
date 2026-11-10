"""CLI client for the local ACE coding assistant service."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request


def _post(url: str, token: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Assistant service returned {exc.code}: {detail}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("Assistant service returned an invalid response")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    completion = subparsers.add_parser("completion")
    completion.add_argument("--prompt", required=True)
    completion.add_argument("--max-new-tokens", type=int, default=128)
    patch = subparsers.add_parser("patch")
    patch.add_argument("--file", required=True)
    patch.add_argument("--instructions", required=True)
    patch.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    token = os.environ.get("ACE_SERVICE_TOKEN")
    if not token:
        raise SystemExit("Set ACE_SERVICE_TOKEN to the token printed by ace-serve")
    payload = vars(args)
    payload.pop("url")
    payload.pop("operation")
    result = _post(f"{args.url.rstrip('/')}/{args.operation}", token, payload)
    if args.operation == "patch":
        print(result.get("diff", ""))
    else:
        print(result.get("text", ""))


if __name__ == "__main__":
    main()
