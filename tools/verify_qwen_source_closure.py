#!/usr/bin/env python3
"""Independently replay a qwen_source_closure_v5 receipt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Mapping


_MAX_CONTROL_BYTES = 1 << 20
_MAX_OBJECT_BYTES = 16 << 30
_CHUNK_BYTES = 1 << 20
_TOP_FIELDS = {
    "schema", "abi", "revision", "authority_root", "receipt_registry_root", "rows",
    "mirror_generation_root", "runtime_sidecar_generation_root", "closure_state", "closure_root",
}
_ROW_FIELDS = {
    "role", "path", "bytes", "sha256", "remote_identity_kind",
    "remote_identity", "review_receipt_roots", "mirror_membership_verified",
    "runtime_sidecar_membership", "sidecar_consumer_receipt_root",
    "independent_receipt_root",
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


def _load_independent_verifier(filename: str, module_name: str) -> Any:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {module_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _object(value: object, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields are invalid")
    return value


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} is not a canonical SHA-256")
    return value


def _parse_json(source: bytes, name: str) -> object:
    if not source or len(source) > _MAX_CONTROL_BYTES:
        raise ValueError(f"{name} exceeds its control-byte bound")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(source.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not UTF-8 JSON") from error


def _load_authority(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != _AUTHORITY_COLUMNS:
            raise ValueError("authority columns are invalid")
        rows = list(reader)
    if len(rows) != 10 or len({row["role"] for row in rows}) != 10:
        raise ValueError("authority must contain exactly ten unique roles")
    return rows


def _root(path: Path, name: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} cannot be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{name} is not a directory")
    return resolved


def _member(root: Path, relative: str, name: str) -> Path:
    if not relative or "\\" in relative or Path(relative).is_absolute():
        raise ValueError(f"{name} path is invalid")
    candidate = root
    for component in relative.split("/"):
        if component in {"", ".", ".."}:
            raise ValueError(f"{name} path is invalid")
        candidate /= component
        if candidate.is_symlink():
            raise ValueError(f"{name} cannot traverse a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file() or root not in resolved.parents:
        raise ValueError(f"{name} escapes its root")
    return resolved


def _identity(path: Path, expected_bytes: int) -> tuple[str, str]:
    if type(expected_bytes) is not int or not 0 <= expected_bytes <= _MAX_OBJECT_BYTES:
        raise ValueError("receipt object length is invalid")
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(f"blob {expected_bytes}\0".encode("ascii"))
    observed = 0
    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        while chunk := source.read(_CHUNK_BYTES):
            observed += len(chunk)
            if observed > expected_bytes:
                raise ValueError("object is longer than its receipt")
            sha256.update(chunk)
            git_sha1.update(chunk)
        after = os.fstat(source.fileno())
    if (
        observed != expected_bytes
        or before.st_size != expected_bytes
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ValueError("object descriptor identity or length differs")
    return sha256.hexdigest(), git_sha1.hexdigest()


def _registry(path: Path, expected_root: str) -> dict[str, Mapping[str, Any]]:
    if path.is_symlink():
        raise ValueError("receipt registry cannot be a symlink")
    value = _parse_json(path.resolve(strict=True).read_bytes(), "receipt registry")
    row = _object(value, {"schema", "receipts"}, "receipt registry")
    receipts = row["receipts"]
    if row["schema"] != "pih.qwen_source_receipt_registry.v1" or not isinstance(receipts, list) or len(receipts) != 46 or any(not isinstance(item, dict) for item in receipts):
        raise ValueError("receipt registry schema or cardinality is invalid")
    roots = [hashlib.sha256(_canonical(item)).hexdigest() for item in receipts]
    if len(set(roots)) != 46 or roots != sorted(roots):
        raise ValueError("receipt registry objects are not unique and hash-sorted")
    if hashlib.sha256(_canonical(row)).hexdigest() != expected_root:
        raise ValueError("receipt registry root differs")
    return dict(zip(roots, receipts, strict=True))


def _registry_receipt(
    objects: Mapping[str, Mapping[str, Any]], root: object,
    fields: set[str], expected: Mapping[str, object], name: str,
) -> Mapping[str, Any]:
    digest = _digest(root, f"{name} root")
    if digest not in objects:
        raise ValueError(f"{name} is absent from the receipt registry")
    receipt = _object(objects[digest], fields, name)
    if (
        isinstance(receipt["source_bytes"], bool)
        or not isinstance(receipt["source_bytes"], int)
        or receipt["source_bytes"] < 0
    ):
        raise ValueError(f"{name}.source_bytes is not a canonical u64")
    for field in ("revision", "role", "path", "source_sha256", "source_bytes"):
        if receipt[field] != expected[field]:
            raise ValueError(f"{name} does not bind the source row")
    if receipt["state"] != "verified":
        raise ValueError(f"{name} is not verified")
    return receipt


def _require_authority_closure_ready(
    authority_row: Mapping[str, str], source_sha256: str,
) -> None:
    """Replay the producer's authority-state gate without importing it."""
    required = {
        "local_materialized": "true",
        "local_length_verified": "true",
        "local_sha256": source_sha256,
        "local_sha256_verified": "true",
        "license_applicability_state": "verified",
        "content_provenance_state": "verified",
        "redistribution_state": "verified",
        "source_mirror_required": "true",
        "source_mirror_membership_verified": "true",
        "independent_verifier_state": "verified",
        "closure_state": "source_closure_open",
    }
    if any(authority_row[field] != expected for field, expected in required.items()):
        raise ValueError("authority has an unclosed closure prerequisite")
    sidecar_required = authority_row["runtime_sidecar_required"]
    membership = authority_row["runtime_sidecar_membership_verified"]
    if (
        (sidecar_required == "true" and membership != "true")
        or (sidecar_required == "false" and membership != "not_applicable")
    ):
        raise ValueError("authority sidecar prerequisite is unclosed")


def verify_closure(
    receipt_path: Path,
    authority_path: Path,
    registry_path: Path,
    source_root: Path,
    mirror_root: Path,
    sidecar_root: Path,
    expected_authority_root: str,
    expected_registry_root: str,
) -> dict[str, object]:
    if receipt_path.is_symlink():
        raise ValueError("closure receipt cannot be a symlink")
    receipt_bytes = receipt_path.resolve(strict=True).read_bytes()
    receipt = _object(
        _parse_json(receipt_bytes, "closure receipt"),
        _TOP_FIELDS,
        "closure receipt",
    )
    if receipt_bytes != _canonical(receipt) + b"\n":
        raise ValueError("closure receipt must use canonical encoding")
    registry_root = _digest(expected_registry_root, "expected registry root")
    authority_root = _digest(expected_authority_root, "expected authority root")
    if (
        receipt["schema"] != "pih.qwen_source_closure_receipt.v2"
        or receipt["abi"] != "qwen_source_closure_v5"
        or receipt["closure_state"] != "sealed"
        or receipt["authority_root"] != authority_root
        or receipt["receipt_registry_root"] != registry_root
    ):
        raise ValueError("closure receipt identity is invalid")
    revision = receipt["revision"]
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("closure revision is invalid")
    rows_value = receipt["rows"]
    if not isinstance(rows_value, list) or len(rows_value) != 10:
        raise ValueError("closure receipt must contain exactly ten rows")
    payload = {
        "abi": receipt["abi"], "revision": revision,
        "authority_root": authority_root,
        "receipt_registry_root": registry_root,
        "mirror_generation_root": receipt["mirror_generation_root"],
        "runtime_sidecar_generation_root": receipt["runtime_sidecar_generation_root"],
        "rows": rows_value,
        "closure_state": "sealed",
    }
    if _digest(receipt["closure_root"], "closure root") != hashlib.sha256(_canonical(payload)).hexdigest():
        raise ValueError("closure root does not replay")
    registry_objects = _registry(registry_path, registry_root)
    roots = {
        "source": _root(source_root, "source root"),
        "mirror": _root(mirror_root, "mirror root"),
        "sidecar": _root(sidecar_root, "sidecar root"),
    }
    if len(set(roots.values())) != 3:
        raise ValueError("source, mirror, and sidecar roots must be distinct")
    authority = _load_authority(authority_path)
    if hashlib.sha256(_canonical(authority)).hexdigest() != authority_root:
        raise ValueError("source authority root differs")
    mirror_verifier = _load_independent_verifier(
        "verify_qwen_source_mirror.py", "verify_qwen_source_mirror",
    )
    sidecar_verifier = _load_independent_verifier(
        "verify_qwen_runtime_sidecar.py", "verify_qwen_runtime_sidecar",
    )
    mirror_receipt = mirror_verifier.verify_mirror(roots["mirror"], authority)
    sidecar_receipt = sidecar_verifier.verify_sidecar(roots["sidecar"], authority)
    if (
        _digest(receipt["mirror_generation_root"], "mirror generation root")
        != _digest(mirror_receipt.get("mirror_root"), "verified mirror generation root")
        or _digest(receipt["runtime_sidecar_generation_root"], "runtime sidecar generation root")
        != _digest(sidecar_receipt.get("sidecar_root"), "verified runtime sidecar generation root")
    ):
        raise ValueError("published generation root differs")
    by_role = {row["role"]: row for row in authority}
    observed_receipts: set[str] = set()
    observed_roles: set[str] = set()
    receipt_roles = [value.get("role") if isinstance(value, dict) else None for value in rows_value]
    if receipt_roles != sorted(receipt_roles) or len(set(receipt_roles)) != 10:
        raise ValueError("closure rows must be unique and role-sorted")
    for index, value in enumerate(rows_value):
        row = _object(value, _ROW_FIELDS, f"rows[{index}]")
        role = row["role"]
        if not isinstance(role, str) or role in observed_roles or role not in by_role:
            raise ValueError("closure row role is duplicate or unknown")
        observed_roles.add(role)
        expected = by_role[role]
        length = row["bytes"]
        length_text = expected["length_bytes"]
        if (
            expected["revision"] != revision
            or expected["required"] != "true"
            or row["path"] != expected["path"]
            or not length_text.isascii()
            or not length_text.isdecimal()
            or (len(length_text) > 1 and length_text.startswith("0"))
            or length != int(length_text)
            or row["remote_identity_kind"] != expected["remote_identity_kind"]
            or row["remote_identity"] != expected["remote_identity"]
            or row["mirror_membership_verified"] is not True
            or expected["source_mirror_required"] != "true"
        ):
            raise ValueError("closure row differs from authority")
        source_sha, git_sha = _identity(_member(roots["source"], row["path"], "source"), length)
        if _digest(row["sha256"], "row sha256") != source_sha:
            raise ValueError("source SHA-256 differs from closure row")
        if row["remote_identity_kind"] not in {"lfs_sha256", "git_blob_sha1"} or (row["remote_identity_kind"] == "lfs_sha256" and row["remote_identity"] != source_sha) or (row["remote_identity_kind"] == "git_blob_sha1" and row["remote_identity"] != git_sha):
            raise ValueError("source bytes differ from remote identity")
        _require_authority_closure_ready(expected, source_sha)
        mirror_sha, _ = _identity(_member(roots["mirror"], row["path"], "mirror"), length)
        if mirror_sha != source_sha:
            raise ValueError("mirror bytes differ from source")
        review = row["review_receipt_roots"]
        if not isinstance(review, list) or len(review) != 3:
            raise ValueError("review receipt roots are invalid")
        receipt_expected = {
            "revision": revision, "role": role, "path": row["path"],
            "source_sha256": source_sha, "source_bytes": length,
        }
        row_receipts: list[str] = []
        for receipt_root, axis in zip(
            review,
            ("license_applicability", "content_provenance", "redistribution"),
            strict=True,
        ):
            review_receipt = _registry_receipt(
                registry_objects, receipt_root,
                {"axis", "revision", "role", "path", "source_sha256", "source_bytes", "state", "receipt_id"},
                receipt_expected, f"{axis} receipt",
            )
            if review_receipt["axis"] != axis or not isinstance(review_receipt["receipt_id"], str) or not review_receipt["receipt_id"]:
                raise ValueError("review receipt identity or axis differs")
            row_receipts.append(_digest(receipt_root, "review receipt root"))
        independent = _registry_receipt(
            registry_objects, row["independent_receipt_root"],
            {"revision", "role", "path", "source_sha256", "source_bytes", "state", "verifier_id"},
            receipt_expected, "independent receipt",
        )
        if not isinstance(independent["verifier_id"], str) or not independent["verifier_id"]:
            raise ValueError("independent verifier identity is invalid")
        row_receipts.append(_digest(row["independent_receipt_root"], "independent receipt root"))
        sidecar_required = expected["runtime_sidecar_required"] == "true"
        if expected["runtime_sidecar_required"] not in {"true", "false"} or (sidecar_required and expected["runtime_sidecar_membership_verified"] not in {"true", "false"}) or (not sidecar_required and expected["runtime_sidecar_membership_verified"] != "not_applicable"):
            raise ValueError("authority sidecar applicability is invalid")
        if sidecar_required:
            sidecar_sha, _ = _identity(_member(roots["sidecar"], row["path"], "sidecar"), length)
            if sidecar_sha != source_sha or row["runtime_sidecar_membership"] != "verified":
                raise ValueError("sidecar membership differs")
            consumer = _registry_receipt(
                registry_objects, row["sidecar_consumer_receipt_root"],
                {"revision", "role", "path", "source_sha256", "source_bytes", "state", "consumer_id"},
                receipt_expected, "sidecar consumer receipt",
            )
            if not isinstance(consumer["consumer_id"], str) or not consumer["consumer_id"]:
                raise ValueError("sidecar consumer identity is invalid")
            row_receipts.append(_digest(row["sidecar_consumer_receipt_root"], "sidecar receipt root"))
        elif row["runtime_sidecar_membership"] != "not_applicable" or row["sidecar_consumer_receipt_root"] is not None:
            raise ValueError("non-runtime sidecar applicability differs")
        if any(item in observed_receipts for item in row_receipts):
            raise ValueError("receipt root was reused across closure roles")
        observed_receipts.update(row_receipts)
    if observed_roles != set(by_role) or observed_receipts != set(registry_objects):
        raise ValueError("closure does not consume the complete authority and receipt registry")
    return {
        "schema": "pih.qwen_source_closure_independent_verification.v1",
        "closure_root": receipt["closure_root"],
        "authority_root": authority_root,
        "receipt_registry_root": registry_root,
        "revision": revision,
        "verified_role_count": 10,
        "verified_receipt_count": 46,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--authority", type=Path, default=Path("specs/qwen-source-closure.csv"))
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--mirror-root", required=True, type=Path)
    parser.add_argument("--sidecar-root", required=True, type=Path)
    parser.add_argument("--expected-authority-root", required=True)
    parser.add_argument("--expected-registry-root", required=True)
    args = parser.parse_args()
    result = verify_closure(
        args.receipt, args.authority, args.registry, args.source_root,
        args.mirror_root, args.sidecar_root, args.expected_authority_root,
        args.expected_registry_root,
    )
    print(_canonical(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
