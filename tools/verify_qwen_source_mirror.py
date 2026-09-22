#!/usr/bin/env python3
"""Independently verify a published preopened_verified_source_mirror_v1 tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping


_ABI = "preopened_verified_source_mirror_v1"
_SCHEMA = "pih.qwen_source_mirror_receipt.v1"
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
_TOP_FIELDS = {"schema", "abi", "revision", "authority_root", "rows", "mirror_root"}
_ROW_FIELDS = {"role", "path", "bytes", "sha256", "remote_identity_kind", "remote_identity"}
_MAX_OBJECT_BYTES = 16 << 30
_MAX_RECEIPT_BYTES = 1 << 20
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
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"{name} is invalid")
    if value == "mirror-receipt.json":
        raise ValueError(f"{name} collides with the reserved mirror receipt")
    return value


def _validate_authority(authority: object) -> list[dict[str, str]]:
    if (
        not isinstance(authority, list)
        or len(authority) != 10
        or any(
            not isinstance(row, dict)
            or set(row) != set(_AUTHORITY_COLUMNS)
            or any(not isinstance(value, str) for value in row.values())
            for row in authority
        )
        or len({row["role"] for row in authority}) != 10
    ):
        raise ValueError("source authority must contain exactly ten complete unique roles")
    revision = authority[0]["revision"]
    _digest(revision, "authority revision", 40)
    for row in authority:
        kind = row["remote_identity_kind"]
        if (
            row["revision"] != revision
            or row["required"] != "true"
            or row["source_mirror_required"] != "true"
            or kind not in {"git_blob_sha1", "lfs_sha256"}
        ):
            raise ValueError("source authority row is not mirror-verifier eligible")
        _relative(row["path"], "authority path")
        _digest(
            row["remote_identity"], "authority remote identity",
            40 if kind == "git_blob_sha1" else 64,
        )
        length = row["length_bytes"]
        if (
            not length.isascii() or not length.isdecimal()
            or (len(length) > 1 and length.startswith("0"))
            or int(length) > _MAX_OBJECT_BYTES
        ):
            raise ValueError("authority length is invalid")
    if len({row["path"] for row in authority}) != 10:
        raise ValueError("source authority paths must be unique")
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


def _member(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("mirror path is invalid")
    candidate = root
    for component in relative.split("/"):
        if component in {"", ".", ".."}:
            raise ValueError("mirror path is invalid")
        candidate /= component
        if candidate.is_symlink():
            raise ValueError("mirror path cannot traverse a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file() or root not in resolved.parents:
        raise ValueError("mirror path escapes root or is not a regular file")
    return resolved


def _tree_members(root: Path) -> tuple[set[str], set[str]]:
    members: set[str] = set()
    directories: set[str] = set()
    for directory, child_directories, child_files in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in child_directories:
            child = current / name
            if child.is_symlink() or not child.is_dir():
                raise ValueError("mirror tree contains an unsafe directory")
            directories.add(child.relative_to(root).as_posix())
        for name in child_files:
            child = current / name
            if child.is_symlink() or not child.is_file():
                raise ValueError("mirror tree contains an unsafe file")
            members.add(child.relative_to(root).as_posix())
    return members, directories


def _read_receipt(path: Path) -> bytes:
    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= _MAX_RECEIPT_BYTES:
            raise ValueError("mirror receipt exceeds its control-byte bound")
        raw = source.read(_MAX_RECEIPT_BYTES + 1)
        after = os.fstat(source.fileno())
    if (
        len(raw) != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ValueError("mirror receipt changed while reading")
    return raw


def _identity(path: Path, expected_bytes: int) -> tuple[str, str]:
    if type(expected_bytes) is not int or not 0 <= expected_bytes <= _MAX_OBJECT_BYTES:
        raise ValueError("mirror object length is invalid")
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(f"blob {expected_bytes}\0".encode("ascii"))
    observed = 0
    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        if before.st_size != expected_bytes:
            raise ValueError("mirror descriptor length differs")
        while chunk := source.read(_CHUNK_BYTES):
            observed += len(chunk)
            if observed > expected_bytes:
                raise ValueError("mirror object is longer than receipt")
            sha256.update(chunk)
            git_sha1.update(chunk)
        after = os.fstat(source.fileno())
    if (
        observed != expected_bytes or
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ValueError("mirror descriptor changed while hashing")
    return sha256.hexdigest(), git_sha1.hexdigest()


def verify_mirror(mirror_root: Path, authority: list[dict[str, str]]) -> dict[str, object]:
    authority = _validate_authority(authority)
    if mirror_root.is_symlink() or not mirror_root.is_dir():
        raise ValueError("mirror root must be a non-symlink directory")
    root = mirror_root.resolve(strict=True)
    manifest_path = root / "mirror-receipt.json"
    if manifest_path.is_symlink():
        raise ValueError("mirror receipt path is invalid")
    raw = _read_receipt(manifest_path)
    try:
        receipt = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("mirror receipt is invalid JSON") from error
    value = _object(receipt, _TOP_FIELDS, "mirror receipt")
    if raw != _canonical(value) + b"\n":
        raise ValueError("mirror receipt is not canonical")
    authority_root = hashlib.sha256(_canonical(authority)).hexdigest()
    revision = authority[0]["revision"]
    if (
        value["schema"] != _SCHEMA or value["abi"] != _ABI
        or value["revision"] != revision or value["authority_root"] != authority_root
    ):
        raise ValueError("mirror receipt identity differs")
    rows_value = value["rows"]
    if not isinstance(rows_value, list) or len(rows_value) != 10:
        raise ValueError("mirror receipt row count is invalid")
    payload = {
        "abi": _ABI, "revision": revision, "authority_root": authority_root,
        "rows": rows_value,
    }
    if _digest(value["mirror_root"], "mirror root") != hashlib.sha256(_canonical(payload)).hexdigest():
        raise ValueError("mirror receipt root does not replay")
    authority_by_role = {row["role"]: row for row in authority}
    roles: list[str] = []
    for index, item in enumerate(rows_value):
        row = _object(item, _ROW_FIELDS, f"mirror row {index}")
        role = row["role"]
        if not isinstance(role, str) or role not in authority_by_role:
            raise ValueError("mirror role is unknown")
        roles.append(role)
        expected = authority_by_role[role]
        length_text = expected["length_bytes"]
        if not length_text.isdecimal() or (len(length_text) > 1 and length_text.startswith("0")):
            raise ValueError("authority length is invalid")
        expected_bytes = int(length_text)
        if (
            row["path"] != expected["path"] or type(row["bytes"]) is not int
            or row["bytes"] != expected_bytes
            or row["remote_identity_kind"] != expected["remote_identity_kind"]
            or row["remote_identity"] != expected["remote_identity"]
        ):
            raise ValueError("mirror row differs from authority")
        observed_sha256, observed_git_sha1 = _identity(
            _member(root, row["path"]), expected_bytes
        )
        if _digest(row["sha256"], "mirror row sha256") != observed_sha256:
            raise ValueError("mirror SHA-256 differs")
        if (
            expected["remote_identity_kind"] == "lfs_sha256"
            and observed_sha256 != expected["remote_identity"]
        ) or (
            expected["remote_identity_kind"] == "git_blob_sha1"
            and observed_git_sha1 != expected["remote_identity"]
        ):
            raise ValueError("mirror bytes differ from remote authority")
    if roles != sorted(roles) or len(set(roles)) != 10:
        raise ValueError("mirror roles must be unique and sorted")
    expected_members = {row["path"] for row in authority}
    expected_members.add("mirror-receipt.json")
    expected_directories: set[str] = set()
    for row in authority:
        parts = row["path"].split("/")[:-1]
        expected_directories.update(
            "/".join(parts[:index]) for index in range(1, len(parts) + 1)
        )
    members, directories = _tree_members(root)
    if members != expected_members or directories != expected_directories:
        raise ValueError("mirror tree membership differs from authority")
    return dict(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror-root", required=True, type=Path)
    parser.add_argument("--authority", type=Path, default=Path("specs/qwen-source-closure.csv"))
    args = parser.parse_args()
    verify_mirror(args.mirror_root, _load_authority(args.authority))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
