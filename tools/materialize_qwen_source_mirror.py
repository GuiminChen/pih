#!/usr/bin/env python3
"""Materialize one verified Qwen source authority into a fresh mirror generation.

This tool intentionally has no network, Hub, cache, or overwrite mode.  It
copies from one caller-provided local source tree, revalidates each authority
object while streaming it, and atomically publishes a new directory only after
every object and the manifest have been fsynced.
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
_MAX_OBJECT_BYTES = 16 << 30
_CHUNK_BYTES = 1 << 20


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: object, name: str, size: int = 64) -> str:
    if (
        type(value) is not str
        or len(value) != size
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


def _load_authority(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != _AUTHORITY_COLUMNS:
            raise ValueError("source authority columns are invalid")
        rows = list(reader)
    if len(rows) != 10 or len({row["role"] for row in rows}) != 10:
        raise ValueError("source authority must contain exactly ten unique roles")
    revision = rows[0]["revision"]
    _digest(revision, "authority revision", 40)
    for row in rows:
        if (
            row["revision"] != revision
            or row["required"] != "true"
            or row["source_mirror_required"] != "true"
            or row["remote_identity_kind"] not in {"git_blob_sha1", "lfs_sha256"}
        ):
            raise ValueError("source authority row is not materialization eligible")
        _relative(row["path"], "authority path")
        _digest(
            row["remote_identity"], "authority remote identity",
            40 if row["remote_identity_kind"] == "git_blob_sha1" else 64,
        )
        length = row["length_bytes"]
        if (
            not length.isascii() or not length.isdecimal()
            or (len(length) > 1 and length.startswith("0"))
            or int(length) > _MAX_OBJECT_BYTES
        ):
            raise ValueError("authority length is invalid")
    if len({row["path"] for row in rows}) != 10:
        raise ValueError("source authority paths must be unique")
    return rows


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
            raise ValueError("mirror parent is not a directory")
        candidate.mkdir(exist_ok=True)
    target = candidate / relative.split("/")[-1]
    if target.exists() or target.is_symlink():
        raise FileExistsError("mirror target already exists")
    return target


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
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("source descriptor changed while copying")
    if observed != expected_bytes:
        raise ValueError("source object is shorter than authority")
    return sha256.hexdigest(), git_sha1.hexdigest()


def _publish_directory(staging: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError("mirror output already exists")
    if os.name == "nt":
        os.rename(staging, output)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) is required on POSIX")
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    if renameat2(
        at_fdcwd, os.fsencode(staging), at_fdcwd, os.fsencode(output), rename_noreplace
    ) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), output)
    _sync_directory(output.parent)


def materialize_mirror(
    source_root: Path, authority: list[dict[str, str]], output: Path
) -> dict[str, object]:
    if (
        not isinstance(authority, list)
        or len(authority) != 10
        or any(not isinstance(row, dict) for row in authority)
        or len({row.get("role") for row in authority}) != 10
    ):
        raise ValueError("source authority must contain exactly ten unique roles")
    revision = authority[0].get("revision")
    _digest(revision, "authority revision", 40)
    for row in authority:
        if (
            row.get("revision") != revision
            or row.get("required") != "true"
            or row.get("source_mirror_required") != "true"
            or row.get("remote_identity_kind") not in {"git_blob_sha1", "lfs_sha256"}
        ):
            raise ValueError("source authority row is not materialization eligible")
        _relative(row.get("path"), "authority path")
        length = row.get("length_bytes")
        if (
            not isinstance(length, str) or not length.isascii()
            or not length.isdecimal() or (len(length) > 1 and length.startswith("0"))
            or int(length) > _MAX_OBJECT_BYTES
        ):
            raise ValueError("authority length is invalid")
        _digest(
            row.get("remote_identity"), "authority remote identity",
            40 if row["remote_identity_kind"] == "git_blob_sha1" else 64,
        )
    if len({row["path"] for row in authority}) != 10:
        raise ValueError("source authority paths must be unique")
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("source root must be a non-symlink directory")
    source_root = source_root.resolve(strict=True)
    output = output.resolve(strict=False)
    if output.exists() or output.is_symlink() or output.parent.is_symlink():
        raise FileExistsError("mirror output already exists or has unsafe parent")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.parent.is_dir():
        raise ValueError("mirror output parent is invalid")
    staging = output.parent / f".{output.name}.staging-{secrets.token_hex(16)}"
    staging.mkdir()
    rows: list[dict[str, object]] = []
    try:
        for authority_row in authority:
            relative = authority_row["path"]
            expected_bytes = int(authority_row["length_bytes"])
            source = _source_member(source_root, relative)
            target = _target_member(staging, relative)
            sha256, git_sha1 = _copy_verified(source, target, expected_bytes)
            if (
                authority_row["remote_identity_kind"] == "lfs_sha256"
                and sha256 != authority_row["remote_identity"]
            ) or (
                authority_row["remote_identity_kind"] == "git_blob_sha1"
                and git_sha1 != authority_row["remote_identity"]
            ):
                raise ValueError("copied source bytes differ from remote authority")
            rows.append({
                "role": authority_row["role"], "path": relative,
                "bytes": expected_bytes, "sha256": sha256,
                "remote_identity_kind": authority_row["remote_identity_kind"],
                "remote_identity": authority_row["remote_identity"],
            })
        rows.sort(key=lambda row: str(row["role"]))
        authority_root = hashlib.sha256(_canonical(authority)).hexdigest()
        payload = {
            "abi": _ABI, "revision": authority[0]["revision"],
            "authority_root": authority_root, "rows": rows,
        }
        receipt = {
            "schema": _SCHEMA, **payload,
            "mirror_root": hashlib.sha256(_canonical(payload)).hexdigest(),
        }
        manifest = staging / "mirror-receipt.json"
        with manifest.open("xb") as target:
            target.write(_canonical(receipt) + b"\n")
            target.flush()
            os.fsync(target.fileno())
        _sync_tree(staging)
        _publish_directory(staging, output)
        return receipt
    except BaseException:
        # Staging is deliberately left for external quarantine/inspection. It
        # has no receipt-bearing published generation name.
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--authority", type=Path, default=Path("specs/qwen-source-closure.csv"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    materialize_mirror(args.source_root, _load_authority(args.authority), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
