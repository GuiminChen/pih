#!/usr/bin/env python3
"""Write a new authenticated-input candidate for native V4.1 batching.

This tool edits deployment configuration only. It never loads or executes a model.
Review and authenticate the resulting file before sealing the service Lock.
"""
import argparse
import hashlib
import json
from pathlib import Path


def configure(document, sequences, tokens, pending):
    if document.get("schema") not in ("pih.deepseek-v41.supervisor.v1", "pih.deepseek-v41.supervisor.v2"):
        raise ValueError("expected V4.1 supervisor configuration v1 or v2")
    if any(type(value) is not int for value in (sequences, tokens, pending)):
        raise ValueError("batch limits must be integers")
    if not 1 <= sequences <= 32 or not sequences <= tokens <= 4096 or not 1 <= pending <= 4096:
        raise ValueError("batch limits exceed native capacities")
    result = dict(document)
    result["schema"] = "pih.deepseek-v41.supervisor.v2"
    result["batching"] = {"maximum_batch_sequences": sequences,
                          "maximum_batch_tokens": tokens,
                          "maximum_pending_requests": pending}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-batch-sequences", type=int, default=4)
    parser.add_argument("--maximum-batch-tokens", type=int, default=4096)
    parser.add_argument("--maximum-pending-requests", type=int, default=64)
    args = parser.parse_args()
    if args.input.stat().st_size > 2 << 20:
        parser.error("input exceeds the native 2 MiB limit")
    try:
        result = configure(json.loads(args.input.read_text(encoding="utf-8")),
                           args.maximum_batch_sequences, args.maximum_batch_tokens,
                           args.maximum_pending_requests)
        payload = (json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        with args.output.open("xb") as output:
            output.write(payload)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"configuration": str(args.output), "sha256": hashlib.sha256(payload).hexdigest(),
                      "review_required_before_sealing": True}))


if __name__ == "__main__":
    main()
