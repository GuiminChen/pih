#!/usr/bin/env python3
"""Independently replay a published Qwen byte-preserving runtime sidecar."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping


_ABI = "byte_preserving_semantic_closure_v1"
_SCHEMA = "pih.qwen_runtime_sidecar_receipt.v1"
_RECEIPT_NAME = "runtime-sidecar-receipt.json"
_RUNTIME_ROLES = (
    "generation_config", "model_config", "tokenizer_config_chat_template",
    "tokenizer_json", "tokenizer_merges", "tokenizer_vocab",
)
_MEDIA_TYPES = {
    "generation_config": "application/json", "model_config": "application/json",
    "tokenizer_config_chat_template": "application/json",
    "tokenizer_json": "application/json",
    "tokenizer_merges": "text/plain; charset=utf-8", "tokenizer_vocab": "application/json",
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
_TOP_FIELDS = {
    "schema", "abi", "revision", "authority_root", "role_set_root", "rows", "sidecar_root",
}
_ROW_FIELDS = {
    "role", "path", "source_bytes", "source_sha256", "copy_bytes", "copy_sha256",
    "media_type", "consumer",
}
_MAX_OBJECT_BYTES = 16 << 30
_MAX_RECEIPT_BYTES = 1 << 20
_CHUNK_BYTES = 1 << 20


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: object, name: str, size: int = 64) -> str:
    if type(value) is not str or len(value) != size or any(item not in "0123456789abcdef" for item in value):
        raise ValueError(f"{name} is not canonical lowercase hexadecimal")
    return value


def _relative(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{name} is invalid")
    if Path(value).is_absolute() or any(item in {"", ".", ".."} for item in value.split("/")):
        raise ValueError(f"{name} is invalid")
    if value == _RECEIPT_NAME:
        raise ValueError(f"{name} collides with the reserved sidecar receipt")
    return value


def _validate_authority(authority: object) -> list[dict[str, str]]:
    if (
        not isinstance(authority, list) or len(authority) != 10
        or any(not isinstance(row, dict) or set(row) != set(_AUTHORITY_COLUMNS) or any(not isinstance(item, str) for item in row.values()) for row in authority)
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
            raise ValueError("source authority row is not sidecar-verifier eligible")
        path = _relative(row["path"], "authority path")
        if path in paths:
            raise ValueError("source authority paths must be unique")
        paths.add(path)
        _digest(row["remote_identity"], "authority remote identity", 40 if kind == "git_blob_sha1" else 64)
        length = row["length_bytes"]
        if not length.isascii() or not length.isdecimal() or (len(length) > 1 and length.startswith("0")) or int(length) > _MAX_OBJECT_BYTES:
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


def _object(value: object, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields are invalid")
    return value


def _read_receipt(path: Path) -> bytes:
    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= _MAX_RECEIPT_BYTES:
            raise ValueError("sidecar receipt exceeds its control-byte bound")
        raw = source.read(_MAX_RECEIPT_BYTES + 1)
        after = os.fstat(source.fileno())
    if len(raw) != before.st_size or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("sidecar receipt changed while reading")
    return raw


def _member(root: Path, relative: object) -> Path:
    relative = _relative(relative, "sidecar path")
    candidate = root
    for component in relative.split("/"):
        candidate /= component
        if candidate.is_symlink():
            raise ValueError("sidecar path cannot traverse a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file() or root not in resolved.parents:
        raise ValueError("sidecar path escapes root or is not a regular file")
    return resolved


def _identity(path: Path, expected_bytes: int) -> tuple[str, str]:
    if type(expected_bytes) is not int or not 0 <= expected_bytes <= _MAX_OBJECT_BYTES:
        raise ValueError("sidecar object length is invalid")
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(f"blob {expected_bytes}\0".encode("ascii"))
    observed = 0
    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        if before.st_size != expected_bytes:
            raise ValueError("sidecar descriptor length differs")
        while chunk := source.read(_CHUNK_BYTES):
            observed += len(chunk)
            if observed > expected_bytes:
                raise ValueError("sidecar object is longer than receipt")
            sha256.update(chunk)
            git_sha1.update(chunk)
        after = os.fstat(source.fileno())
    if observed != expected_bytes or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("sidecar descriptor changed while hashing")
    return sha256.hexdigest(), git_sha1.hexdigest()


def _tree(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    for directory, child_directories, child_files in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in child_directories:
            child = current / name
            if child.is_symlink() or not child.is_dir():
                raise ValueError("sidecar tree contains an unsafe directory")
            directories.add(child.relative_to(root).as_posix())
        for name in child_files:
            child = current / name
            if child.is_symlink() or not child.is_file():
                raise ValueError("sidecar tree contains an unsafe file")
            files.add(child.relative_to(root).as_posix())
    return files, directories


def verify_sidecar(sidecar_root: Path, authority: list[dict[str, str]]) -> dict[str, object]:
    authority = _validate_authority(authority)
    if sidecar_root.is_symlink() or not sidecar_root.is_dir():
        raise ValueError("sidecar root must be a non-symlink directory")
    root = sidecar_root.resolve(strict=True)
    receipt_path = root / _RECEIPT_NAME
    if receipt_path.is_symlink():
        raise ValueError("sidecar receipt path is invalid")
    raw = _read_receipt(receipt_path)
    try:
        receipt = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("sidecar receipt is invalid JSON") from error
    value = _object(receipt, _TOP_FIELDS, "sidecar receipt")
    if raw != _canonical(value) + b"\n":
        raise ValueError("sidecar receipt is not canonical")
    authority_root = hashlib.sha256(_canonical(authority)).hexdigest()
    role_set_root = hashlib.sha256(_canonical(list(sorted(_RUNTIME_ROLES)))).hexdigest()
    if value["schema"] != _SCHEMA or value["abi"] != _ABI or value["revision"] != authority[0]["revision"] or value["authority_root"] != authority_root or value["role_set_root"] != role_set_root:
        raise ValueError("sidecar receipt identity differs")
    rows_value = value["rows"]
    if not isinstance(rows_value, list) or len(rows_value) != len(_RUNTIME_ROLES):
        raise ValueError("sidecar receipt row count is invalid")
    payload = {"abi": _ABI, "revision": authority[0]["revision"], "authority_root": authority_root, "role_set_root": role_set_root, "rows": rows_value}
    if _digest(value["sidecar_root"], "sidecar root") != hashlib.sha256(_canonical(payload)).hexdigest():
        raise ValueError("sidecar receipt root does not replay")
    by_role = {row["role"]: row for row in authority}
    roles: list[str] = []
    expected_files = {_RECEIPT_NAME}
    expected_dirs: set[str] = set()
    for index, item in enumerate(rows_value):
        row = _object(item, _ROW_FIELDS, f"sidecar row {index}")
        role = row["role"]
        if not isinstance(role, str) or role not in by_role or role not in _RUNTIME_ROLES:
            raise ValueError("sidecar role is unknown")
        roles.append(role)
        expected = by_role[role]
        expected_bytes = int(expected["length_bytes"])
        if row["path"] != expected["path"] or type(row["source_bytes"]) is not int or type(row["copy_bytes"]) is not int or row["source_bytes"] != expected_bytes or row["copy_bytes"] != expected_bytes or row["media_type"] != _MEDIA_TYPES[role] or row["consumer"] != _CONSUMERS[role]:
            raise ValueError("sidecar row differs from authority")
        observed_sha256, observed_git_sha1 = _identity(_member(root, row["path"]), expected_bytes)
        if _digest(row["source_sha256"], "sidecar source SHA-256") != observed_sha256 or _digest(row["copy_sha256"], "sidecar copy SHA-256") != observed_sha256:
            raise ValueError("sidecar SHA-256 differs")
        if (expected["remote_identity_kind"] == "lfs_sha256" and observed_sha256 != expected["remote_identity"]) or (expected["remote_identity_kind"] == "git_blob_sha1" and observed_git_sha1 != expected["remote_identity"]):
            raise ValueError("sidecar bytes differ from remote authority")
        relative = row["path"]
        expected_files.add(relative)
        parts = relative.split("/")[:-1]
        expected_dirs.update("/".join(parts[:item]) for item in range(1, len(parts) + 1))
    if roles != sorted(_RUNTIME_ROLES):
        raise ValueError("sidecar roles must be unique and sorted")
    if _tree(root) != (expected_files, expected_dirs):
        raise ValueError("sidecar tree membership differs from authority")
    return dict(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-root", required=True, type=Path)
    parser.add_argument("--authority", type=Path, default=Path("specs/qwen-source-closure.csv"))
    args = parser.parse_args()
    verify_sidecar(args.sidecar_root, _load_authority(args.authority))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
