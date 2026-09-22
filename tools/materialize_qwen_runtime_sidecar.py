#!/usr/bin/env python3
"""Materialize the six byte-preserving Qwen runtime sidecar objects.

This local-only tool is deliberately an antecedent to artifact publication.  It
does not alter the source-closure authority status and it never consults a
Hub, cache, or tokenizer path at runtime.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import secrets
from typing import Any


_ABI = "byte_preserving_semantic_closure_v1"
_SCHEMA = "pih.qwen_runtime_sidecar_receipt.v1"
_RECEIPT_NAME = "runtime-sidecar-receipt.json"
_RUNTIME_ROLES = (
    "generation_config",
    "model_config",
    "tokenizer_config_chat_template",
    "tokenizer_json",
    "tokenizer_merges",
    "tokenizer_vocab",
)
_MEDIA_TYPES = {
    "generation_config": "application/json",
    "model_config": "application/json",
    "tokenizer_config_chat_template": "application/json",
    "tokenizer_json": "application/json",
    "tokenizer_merges": "text/plain; charset=utf-8",
    "tokenizer_vocab": "application/json",
}
_CONSUMERS = {
    "generation_config": "qwen3_generation_policy_v1",
    "model_config": "qwen3_compiled_model_schema_v1",
    "tokenizer_config_chat_template": "qwen3_python_tokenizer_v1",
    "tokenizer_json": "qwen3_python_tokenizer_v1",
    "tokenizer_merges": "qwen3_python_tokenizer_v1",
    "tokenizer_vocab": "qwen3_python_tokenizer_v1",
}
_AUTHORITY_COLUMNS = (
    "revision", "role", "path", "required", "length_bytes",
    "remote_identity_kind", "remote_identity", "local_materialized",
    "local_length_verified", "local_sha256", "local_sha256_verified",
    "license_applicability_state", "content_provenance_state",
    "redistribution_state", "runtime_sidecar_required",
    "runtime_sidecar_membership_verified", "source_mirror_required",
    "source_mirror_membership_verified", "independent_verifier_state",
    "closure_state",
)
_MAX_OBJECT_BYTES = 16 << 30
_CHUNK_BYTES = 1 << 20


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: object, name: str, size: int = 64) -> str:
    if (
        type(value) is not str or len(value) != size
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} is not canonical lowercase hexadecimal")
    return value


def _relative(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{name} is invalid")
    if Path(value).is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"{name} is invalid")
    if value == _RECEIPT_NAME:
        raise ValueError(f"{name} collides with the reserved sidecar receipt")
    return value


def _validate_authority(authority: object) -> list[dict[str, str]]:
    if (
        not isinstance(authority, list) or len(authority) != 10
        or any(
            not isinstance(row, dict) or set(row) != set(_AUTHORITY_COLUMNS)
            or any(not isinstance(value, str) for value in row.values())
            for row in authority
        )
        or len({row["role"] for row in authority}) != 10
    ):
        raise ValueError("source authority must contain exactly ten complete unique roles")
    revision = authority[0]["revision"]
    _digest(revision, "authority revision", 40)
    runtime_roles: set[str] = set()
    paths: set[str] = set()
    for row in authority:
        kind = row["remote_identity_kind"]
        if row["revision"] != revision or row["required"] != "true" or kind not in {"git_blob_sha1", "lfs_sha256"}:
            raise ValueError("source authority row is not sidecar-materialization eligible")
        path = _relative(row["path"], "authority path")
        if path in paths:
            raise ValueError("source authority paths must be unique")
        paths.add(path)
        _digest(row["remote_identity"], "authority remote identity", 40 if kind == "git_blob_sha1" else 64)
        length = row["length_bytes"]
        if (
            not length.isascii() or not length.isdecimal()
            or (len(length) > 1 and length.startswith("0"))
            or int(length) > _MAX_OBJECT_BYTES
        ):
            raise ValueError("authority length is invalid")
        if row["runtime_sidecar_required"] == "true":
            runtime_roles.add(row["role"])
            if row["runtime_sidecar_membership_verified"] not in {"false", "true"}:
                raise ValueError("runtime role sidecar membership state is invalid")
        elif row["runtime_sidecar_required"] == "false":
            if row["runtime_sidecar_membership_verified"] != "not_applicable":
                raise ValueError("non-runtime role sidecar membership state is invalid")
        else:
            raise ValueError("authority runtime sidecar requirement is invalid")
    if runtime_roles != set(_RUNTIME_ROLES):
        raise ValueError("source authority runtime role set is invalid")
    return authority


def _load_authority(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != _AUTHORITY_COLUMNS:
            raise ValueError("source authority columns are invalid")
        rows = list(reader)
    return _validate_authority(rows)


def _source_member(root: Path, relative: str) -> Path:
    candidate = root
    for component in relative.split("/"):
        candidate /= component
        if candidate.is_symlink():
            raise ValueError("source path cannot traverse a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file() or root not in resolved.parents:
        raise ValueError("source path escapes root or is not a regular file")
    return resolved


def _target_member(root: Path, relative: str) -> Path:
    candidate = root
    for component in relative.split("/")[:-1]:
        candidate /= component
        if candidate.exists() and (candidate.is_symlink() or not candidate.is_dir()):
            raise ValueError("sidecar parent is not a directory")
        candidate.mkdir(exist_ok=True)
    target = candidate / relative.split("/")[-1]
    if target.exists() or target.is_symlink():
        raise FileExistsError("sidecar target already exists")
    return target


def _copy_verified(source: Path, target: Path, expected_bytes: int) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(f"blob {expected_bytes}\0".encode("ascii"))
    observed = 0
    with source.open("rb") as input_file:
        before = os.fstat(input_file.fileno())
        if before.st_size != expected_bytes:
            raise ValueError("source descriptor length differs from authority")
        with target.open("xb") as output_file:
            while chunk := input_file.read(_CHUNK_BYTES):
                observed += len(chunk)
                if observed > expected_bytes:
                    raise ValueError("source object is longer than authority")
                sha256.update(chunk)
                git_sha1.update(chunk)
                output_file.write(chunk)
            output_file.flush()
            os.fsync(output_file.fileno())
        after = os.fstat(input_file.fileno())
    if (
        observed != expected_bytes
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ValueError("source descriptor changed while copying")
    return sha256.hexdigest(), git_sha1.hexdigest()


def _sync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_tree(root: Path) -> None:
    for directory, _, _ in os.walk(root, topdown=False):
        _sync_directory(Path(directory))


def _publish_directory(staging: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError("sidecar output already exists")
    if os.name == "nt":
        os.rename(staging, output)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) is required on POSIX")
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(staging), -100, os.fsencode(output), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), output)
    _sync_directory(output.parent)


def materialize_sidecar(
    source_root: Path, authority: list[dict[str, str]], output: Path,
) -> dict[str, object]:
    authority = _validate_authority(authority)
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("source root must be a non-symlink directory")
    source_root = source_root.resolve(strict=True)
    output = output.resolve(strict=False)
    if output.exists() or output.is_symlink() or output.parent.is_symlink():
        raise FileExistsError("sidecar output already exists or has unsafe parent")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.parent.is_dir():
        raise ValueError("sidecar output parent is invalid")
    by_role = {row["role"]: row for row in authority}
    staging = output.parent / f".{output.name}.staging-{secrets.token_hex(16)}"
    staging.mkdir()
    rows: list[dict[str, object]] = []
    try:
        for role in sorted(_RUNTIME_ROLES):
            source_row = by_role[role]
            relative = source_row["path"]
            expected_bytes = int(source_row["length_bytes"])
            source = _source_member(source_root, relative)
            target = _target_member(staging, relative)
            source_sha256, source_git_sha1 = _copy_verified(source, target, expected_bytes)
            if (
                source_row["remote_identity_kind"] == "lfs_sha256"
                and source_sha256 != source_row["remote_identity"]
            ) or (
                source_row["remote_identity_kind"] == "git_blob_sha1"
                and source_git_sha1 != source_row["remote_identity"]
            ):
                raise ValueError("copied source bytes differ from remote authority")
            rows.append({
                "role": role, "path": relative,
                "source_bytes": expected_bytes, "source_sha256": source_sha256,
                "copy_bytes": expected_bytes, "copy_sha256": source_sha256,
                "media_type": _MEDIA_TYPES[role], "consumer": _CONSUMERS[role],
            })
        role_set_root = hashlib.sha256(_canonical(list(sorted(_RUNTIME_ROLES)))).hexdigest()
        authority_root = hashlib.sha256(_canonical(authority)).hexdigest()
        payload = {
            "abi": _ABI, "revision": authority[0]["revision"],
            "authority_root": authority_root, "role_set_root": role_set_root,
            "rows": rows,
        }
        receipt = {
            "schema": _SCHEMA, **payload,
            "sidecar_root": hashlib.sha256(_canonical(payload)).hexdigest(),
        }
        manifest = staging / _RECEIPT_NAME
        with manifest.open("xb") as target:
            target.write(_canonical(receipt) + b"\n")
            target.flush()
            os.fsync(target.fileno())
        _sync_tree(staging)
        _publish_directory(staging, output)
        return receipt
    except BaseException:
        # An incomplete generation remains unpublishable for external quarantine.
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--authority", type=Path, default=Path("specs/qwen-source-closure.csv"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    materialize_sidecar(args.source_root, _load_authority(args.authority), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
