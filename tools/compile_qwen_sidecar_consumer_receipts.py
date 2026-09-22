#!/usr/bin/env python3
"""Project a verified Qwen runtime sidecar into closure-consumer receipts.

The resulting six receipts are inputs to, rather than a substitute for, the
complete source-closure receipt registry.  Legal, provenance, redistribution,
and independent source-verifier receipts remain caller-owned authorities.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping


_SCHEMA = "pih.qwen_sidecar_consumer_receipts.v1"
_ROWS = 6
_TOP_FIELDS = {"schema", "sidecar_root", "rows", "consumer_receipts_root"}
_ROW_FIELDS = {
    "revision", "role", "path", "source_sha256", "source_bytes", "state",
    "consumer_id",
}


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(item not in "0123456789abcdef" for item in value):
        raise ValueError(f"{name} is not canonical lowercase hexadecimal")
    return value


def _load_verifier() -> Any:
    path = Path(__file__).with_name("verify_qwen_runtime_sidecar.py")
    spec = importlib.util.spec_from_file_location("verify_qwen_runtime_sidecar", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Qwen runtime sidecar verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_consumer_receipts(
    sidecar_root: Path, authority: list[dict[str, str]],
) -> dict[str, object]:
    verifier = _load_verifier()
    verified = verifier.verify_sidecar(sidecar_root, authority)
    rows_value = verified["rows"]
    if not isinstance(rows_value, list) or len(rows_value) != _ROWS:
        raise ValueError("verified sidecar row count is invalid")
    receipts: list[dict[str, object]] = []
    for row in rows_value:
        if not isinstance(row, Mapping):
            raise ValueError("verified sidecar row is invalid")
        source_bytes = row.get("source_bytes")
        if type(source_bytes) is not int or source_bytes < 0:
            raise ValueError("verified sidecar source_bytes is invalid")
        receipt = {
            "revision": verified["revision"], "role": row.get("role"),
            "path": row.get("path"), "source_sha256": row.get("source_sha256"),
            "source_bytes": source_bytes, "state": "verified",
            "consumer_id": row.get("consumer"),
        }
        if set(receipt) != _ROW_FIELDS:
            raise ValueError("sidecar consumer receipt fields are invalid")
        receipts.append(receipt)
    if [item["role"] for item in receipts] != sorted(item["role"] for item in receipts):
        raise ValueError("verified sidecar receipt roles are not sorted")
    payload = {"schema": _SCHEMA, "sidecar_root": verified["sidecar_root"], "rows": receipts}
    return {**payload, "consumer_receipts_root": hashlib.sha256(_canonical(payload)).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-root", required=True, type=Path)
    parser.add_argument("--authority", type=Path, default=Path("specs/qwen-source-closure.csv"))
    args = parser.parse_args()
    verifier = _load_verifier()
    authority = verifier._load_authority(args.authority)
    print(json.dumps(compile_consumer_receipts(args.sidecar_root, authority), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
