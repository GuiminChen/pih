#!/usr/bin/env python3
"""Independently replay non-authorizing Qwen M2 4090 D core receipts."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


_MAX_RECEIPT_BYTES = 64 * 1024
_PAYLOAD_FIELDS = {
    "schema", "qualification_scope", "support_state", "profile_id",
    "device_ordinal", "run_ordinal", "required_repetitions", "context_tokens",
    "maximum_new_tokens", "artifact_sha256", "source_closure_root", "commit_sha",
    "started_ns", "finished_ns", "gpu_name", "gpu_uuid", "compute_capability",
    "total_memory_bytes", "driver_version", "cuda_version", "cubin_sha256",
    "generated_token_ids", "memory_before_bytes", "memory_observed_max_bytes",
    "memory_after_bytes", "teardown_tolerance_bytes", "clean_teardown",
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _digest(value: object, name: str, lengths: frozenset[int] = frozenset({64})) -> str:
    if (
        type(value) is not str or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} is not canonical lowercase hexadecimal")
    return value


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be nonempty text")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be ASCII") from error
    if len(encoded) > 256:
        raise ValueError(f"{name} exceeds its byte bound")
    return value


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"receipt is absent or a symlink: {path.name}")
    source = path.read_bytes()
    if not source or len(source) > _MAX_RECEIPT_BYTES:
        raise ValueError(f"receipt exceeds its byte bound: {path.name}")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate receipt field: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(source.decode("ascii"), object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"receipt is not canonical ASCII JSON: {path.name}") from error
    if type(value) is not dict or set(value) != _PAYLOAD_FIELDS | {"evidence_root"}:
        raise ValueError(f"receipt fields are invalid: {path.name}")
    if source != _canonical(value) + b"\n":
        raise ValueError(f"receipt is not canonical ASCII JSON: {path.name}")
    return value


def verify_receipts(
    evidence_root: Path,
    *,
    repetitions: int,
    artifact_sha256: str,
    source_closure_root: str,
    commit_sha: str,
    cubin_sha256: str,
    gpu_uuid: str,
) -> str:
    if evidence_root.is_symlink():
        raise ValueError("evidence_root cannot be a symlink")
    root = evidence_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("evidence_root must be a directory")
    if type(repetitions) is not int or not 5 <= repetitions <= 100:
        raise ValueError("repetitions must be in [5, 100]")
    expected = {
        "artifact_sha256": _digest(artifact_sha256, "artifact_sha256"),
        "source_closure_root": _digest(source_closure_root, "source_closure_root"),
        "commit_sha": _digest(commit_sha, "commit_sha", frozenset({40, 64})),
        "cubin_sha256": _digest(cubin_sha256, "cubin_sha256"),
        "gpu_uuid": gpu_uuid,
    }
    if not _text(gpu_uuid, "gpu_uuid").startswith("GPU-"):
        raise ValueError("gpu_uuid is invalid")
    names = tuple(f"rtx4090d-qwen-int4-32k-run{i:02d}.json" for i in range(1, repetitions + 1))
    observed_names = tuple(sorted(path.name for path in root.iterdir()))
    if observed_names != names:
        raise ValueError("evidence directory is not the exact receipt set")

    receipt_roots: list[str] = []
    stable: dict[str, object] | None = None
    previous_finished = 0
    for ordinal, name in enumerate(names, 1):
        document = _load(root / name)
        evidence_digest = _digest(document.pop("evidence_root"), "evidence_root")
        if sha256(_canonical(document)).hexdigest() != evidence_digest:
            raise ValueError(f"receipt evidence root mismatch: {name}")
        for field, value in expected.items():
            if document[field] != value:
                raise ValueError(f"receipt {field} differs from authorized identity")
        fixed = {
            field: document[field]
            for field in (
                "required_repetitions", "gpu_name", "gpu_uuid",
                "compute_capability", "total_memory_bytes", "driver_version",
                "cuda_version", "teardown_tolerance_bytes",
            )
        }
        if stable is None:
            stable = fixed
        elif stable != fixed:
            raise ValueError("cross-run hardware or environment identity drifted")
        canonical_integer_axes = (
            document["device_ordinal"], document["run_ordinal"],
            document["required_repetitions"], document["context_tokens"],
            document["maximum_new_tokens"],
        )
        if (
            document["schema"] != "pih.qwen_m2.hardware_observation.v1"
            or document["qualification_scope"] != "core_32k_smoke_non_authorizing"
            or document["support_state"] != "hardware_evidence_open"
            or document["profile_id"] != "QW-4090-INT4"
            or any(type(value) is not int for value in canonical_integer_axes)
            or document["device_ordinal"] != 0
            or document["run_ordinal"] != ordinal
            or document["required_repetitions"] != repetitions
            or document["context_tokens"] != 32_768
            or document["maximum_new_tokens"] != 1
            or document["gpu_name"] != "NVIDIA GeForce RTX 4090 D"
            or document["compute_capability"] != "8.9"
            or document["clean_teardown"] is not True
        ):
            raise ValueError(f"receipt canonical axes are invalid: {name}")
        _text(document["driver_version"], "driver_version")
        _text(document["cuda_version"], "cuda_version")
        tokens = document["generated_token_ids"]
        integers = (
            document["started_ns"], document["finished_ns"],
            document["total_memory_bytes"], document["memory_before_bytes"],
            document["memory_observed_max_bytes"], document["memory_after_bytes"],
            document["teardown_tolerance_bytes"],
        )
        if (
            type(tokens) is not list or len(tokens) != 1
            or type(tokens[0]) is not int or not 0 <= tokens[0] < 151_936
            or any(type(value) is not int or value < 0 for value in integers)
            or document["total_memory_bytes"] <= 0
            or document["started_ns"] <= previous_finished
            or document["finished_ns"] <= document["started_ns"]
            or document["memory_observed_max_bytes"] < document["memory_before_bytes"]
            or document["memory_after_bytes"] > document["memory_before_bytes"] + document["teardown_tolerance_bytes"]
            or not 0 <= document["teardown_tolerance_bytes"] <= 64 * 1024 * 1024
        ):
            raise ValueError(f"receipt observation values are invalid: {name}")
        previous_finished = document["finished_ns"]
        receipt_roots.append(evidence_digest)
    return sha256(_canonical({"receipt_roots": receipt_roots, **expected})).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_root", type=Path)
    parser.add_argument("--repetitions", required=True, type=int)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--source-closure-root", required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--cubin-sha256", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    arguments = parser.parse_args()
    root = verify_receipts(
        arguments.evidence_root, repetitions=arguments.repetitions,
        artifact_sha256=arguments.artifact_sha256,
        source_closure_root=arguments.source_closure_root,
        commit_sha=arguments.commit_sha, cubin_sha256=arguments.cubin_sha256,
        gpu_uuid=arguments.gpu_uuid,
    )
    print(json.dumps({"verification_root": root}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
