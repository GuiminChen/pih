#!/usr/bin/env python3
"""Compile a fail-closed qwen_source_closure_v5 receipt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


_ABI = "qwen_source_closure_v5"
_INPUT_SCHEMA = "pih.qwen_source_closure_input.v1"
_OUTPUT_SCHEMA = "pih.qwen_source_closure_receipt.v2"
_MAX_CONTROL_BYTES = 1 << 20
_MAX_OBJECT_BYTES = 16 << 30
_CHUNK_BYTES = 1 << 20
_TOP_FIELDS = {"schema", "revision", "authority_root", "receipt_registry_path", "receipt_registry_root", "source_root", "mirror_root", "sidecar_root", "rows"}
_ROW_FIELDS = {
    "role",
    "path",
    "license_receipt",
    "provenance_receipt",
    "redistribution_receipt",
    "independent_receipt",
    "sidecar_consumer_receipt",
}
_REVIEW_FIELDS = {"axis", "revision", "role", "path", "source_sha256", "source_bytes", "state", "receipt_id"}
_VERIFY_FIELDS = {"revision", "role", "path", "source_sha256", "source_bytes", "state", "verifier_id"}
_CONSUMER_FIELDS = {"revision", "role", "path", "source_sha256", "source_bytes", "state", "consumer_id"}
_REGISTRY_FIELDS = {"schema", "receipts"}
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
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _object(value: object, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields are invalid")
    return value


def _text(value: object, name: str, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not 1 <= len(value.encode("utf-8")) <= maximum:
        raise ValueError(f"{name} is invalid")
    return value


def _digest(value: object, name: str) -> str:
    result = _text(value, name, 64)
    if len(result) != 64 or any(c not in "0123456789abcdef" for c in result):
        raise ValueError(f"{name} is not a canonical SHA-256")
    return result


def parse_input(source: bytes) -> Mapping[str, Any]:
    if not isinstance(source, bytes) or not source or len(source) > _MAX_CONTROL_BYTES:
        raise ValueError("source-closure input exceeds its control-byte bound")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(source.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("source-closure input is not canonical UTF-8 JSON") from error
    return _object(value, _TOP_FIELDS, "input")


def load_authority(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != _AUTHORITY_COLUMNS:
            raise ValueError("source authority columns are invalid")
        rows = list(reader)
    if len(rows) != 10 or len({row["role"] for row in rows}) != 10:
        raise ValueError("source authority must contain exactly ten unique roles")
    return rows


def _beneath(root: Path, relative: str, name: str) -> Path:
    if not relative or "\\" in relative or Path(relative).is_absolute():
        raise ValueError(f"{name} path is invalid")
    root = root.resolve(strict=True)
    candidate = root
    for component in relative.split("/"):
        if component in {"", ".", ".."}:
            raise ValueError(f"{name} path is invalid")
        candidate = candidate / component
        if candidate.is_symlink():
            raise ValueError(f"{name} cannot traverse a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file() or resolved == root or root not in resolved.parents:
        raise ValueError(f"{name} escapes its root or is not a regular file")
    return resolved


def _stream_identity(path: Path, expected_bytes: int) -> tuple[str, str]:
    if expected_bytes < 0 or expected_bytes > _MAX_OBJECT_BYTES:
        raise ValueError("source object length is outside its bound")
    sha256 = hashlib.sha256()
    sha1 = hashlib.sha1()
    sha1.update(f"blob {expected_bytes}\0".encode("ascii"))
    observed = 0
    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        if before.st_size != expected_bytes:
            raise ValueError("source object descriptor length differs")
        while chunk := source.read(_CHUNK_BYTES):
            observed += len(chunk)
            if observed > expected_bytes:
                raise ValueError("source object is longer than authority")
            sha256.update(chunk)
            sha1.update(chunk)
        after = os.fstat(source.fileno())
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise ValueError("source object descriptor changed while hashing")
    if observed != expected_bytes:
        raise ValueError("source object is shorter than authority")
    return sha256.hexdigest(), sha1.hexdigest()


def _receipt(value: object, *, fields: set[str], name: str, expected: dict[str, object], registry_members: set[str], identity_field: str) -> str:
    row = _object(value, fields, name)
    if (
        isinstance(row["source_bytes"], bool)
        or not isinstance(row["source_bytes"], int)
        or row["source_bytes"] < 0
    ):
        raise ValueError(f"{name}.source_bytes is not a canonical u64")
    for field in ("revision", "role", "path", "source_sha256", "source_bytes"):
        if row[field] != expected[field]:
            raise ValueError(f"{name} does not bind the source row")
    if row["state"] != "verified":
        raise ValueError(f"{name} is not verified")
    identity = _text(row[identity_field], f"{name}.{identity_field}", 256)
    root = hashlib.sha256(_canonical(row)).hexdigest()
    if root not in registry_members:
        raise ValueError(f"{name} is not a member of the receipt registry")
    return root


def _load_receipt_registry(path: Path, expected_root: str) -> dict[str, Mapping[str, Any]]:
    source = path.read_bytes()
    if not source or len(source) > _MAX_CONTROL_BYTES:
        raise ValueError("receipt registry exceeds its byte bound")
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate receipt registry key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(source.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("receipt registry is invalid") from error
    row = _object(value, _REGISTRY_FIELDS, "receipt_registry")
    if row["schema"] != "pih.qwen_source_receipt_registry.v1":
        raise ValueError("receipt registry schema is invalid")
    receipts = row["receipts"]
    if not isinstance(receipts, list) or len(receipts) != 46 or any(not isinstance(item, dict) for item in receipts):
        raise ValueError("receipt registry must contain exactly 46 receipts")
    roots = [hashlib.sha256(_canonical(item)).hexdigest() for item in receipts]
    if len(set(roots)) != 46 or roots != sorted(roots):
        raise ValueError("receipt registry objects must be unique and hash-sorted")
    observed_root = hashlib.sha256(_canonical(row)).hexdigest()
    if observed_root != expected_root:
        raise ValueError("receipt registry root differs")
    return dict(zip(roots, receipts, strict=True))


def _require_authority_closure_ready(
    authority_row: Mapping[str, str], source_sha256: str,
) -> None:
    """Reject a sealed receipt unless its authority records every prior gate.

    Byte checks below establish what was read during this invocation.  They do
    not authorize promotion of a discovery-state authority table: that table
    must already record the independently reviewed, per-object closure state.
    """
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
        # This is the pre-publication authority state.  The published receipt,
        # not a hand-edited authority row, carries the sealed closure claim.
        "closure_state": "source_closure_open",
    }
    if any(authority_row[field] != expected for field, expected in required.items()):
        raise ValueError("source authority has an unclosed prerequisite")
    sidecar_required = authority_row["runtime_sidecar_required"]
    membership = authority_row["runtime_sidecar_membership_verified"]
    if (
        (sidecar_required == "true" and membership != "true")
        or (sidecar_required == "false" and membership != "not_applicable")
    ):
        raise ValueError("source authority sidecar prerequisite is unclosed")


def compile_closure(control: Mapping[str, Any], authority: list[dict[str, str]]) -> dict[str, object]:
    _object(control, _TOP_FIELDS, "input")
    if control["schema"] != _INPUT_SCHEMA:
        raise ValueError("source-closure input schema is invalid")
    revision = _text(control["revision"], "revision", 64)
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("revision is not a full lowercase commit identity")
    authority_root = _digest(control["authority_root"], "authority_root")
    if hashlib.sha256(_canonical(authority)).hexdigest() != authority_root:
        raise ValueError("source authority root differs")
    registry_root = _digest(control["receipt_registry_root"], "receipt_registry_root")
    raw_registry_path = Path(_text(control["receipt_registry_path"], "receipt_registry_path"))
    if raw_registry_path.is_symlink():
        raise ValueError("receipt registry path is invalid")
    registry_path = raw_registry_path.resolve(strict=True)
    if not registry_path.is_file():
        raise ValueError("receipt registry path is invalid")
    registry_objects = _load_receipt_registry(registry_path, registry_root)
    registry_members = set(registry_objects)
    raw_roots = {name: Path(_text(control[name], name)) for name in ("source_root", "mirror_root", "sidecar_root")}
    if any(path.is_symlink() for path in raw_roots.values()):
        raise ValueError("source-closure roots cannot be symlinks")
    roots = {name: path.resolve(strict=True) for name, path in raw_roots.items()}
    if not all(path.is_dir() for path in roots.values()):
        raise ValueError("source-closure roots must be directories")
    if len(set(roots.values())) != 3:
        raise ValueError("source, mirror, and sidecar roots must be distinct")
    values = control["rows"]
    if not isinstance(values, list) or len(values) != 10:
        raise ValueError("source-closure input must contain exactly ten rows")
    by_role: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(values):
        row = _object(value, _ROW_FIELDS, f"rows[{index}]")
        role = _text(row["role"], f"rows[{index}].role", 128)
        if role in by_role:
            raise ValueError("source-closure input contains a duplicate role")
        by_role[role] = row

    compiled: list[dict[str, object]] = []
    for authority_row in authority:
        role = authority_row["role"]
        if authority_row["revision"] != revision or authority_row["required"] != "true" or role not in by_role:
            raise ValueError("source authority revision, requirement, or role differs")
        value = by_role[role]
        relative = authority_row["path"]
        if value["path"] != relative:
            raise ValueError("source row path differs from authority")
        length_text = authority_row["length_bytes"]
        if not length_text.isascii() or not length_text.isdecimal() or (len(length_text) > 1 and length_text.startswith("0")):
            raise ValueError("source authority length is not canonical u64 decimal")
        expected_bytes = int(length_text)
        sidecar_required_text = authority_row["runtime_sidecar_required"]
        sidecar_membership = authority_row["runtime_sidecar_membership_verified"]
        if authority_row["source_mirror_required"] != "true" or authority_row["source_mirror_membership_verified"] not in {"false", "true"}:
            raise ValueError("source authority mirror applicability is invalid")
        if (sidecar_required_text == "true" and sidecar_membership not in {"false", "true"}) or (sidecar_required_text == "false" and sidecar_membership != "not_applicable"):
            raise ValueError("source authority sidecar applicability is invalid")
        if sidecar_required_text not in {"true", "false"}:
            raise ValueError("source authority sidecar requirement is invalid")
        source_path = _beneath(roots["source_root"], relative, "source")
        source_sha256, git_blob_sha1 = _stream_identity(source_path, expected_bytes)
        kind = authority_row["remote_identity_kind"]
        remote = authority_row["remote_identity"]
        if (kind == "lfs_sha256" and source_sha256 != remote) or (kind == "git_blob_sha1" and git_blob_sha1 != remote) or kind not in {"lfs_sha256", "git_blob_sha1"}:
            raise ValueError("local source bytes differ from remote authority")
        _require_authority_closure_ready(authority_row, source_sha256)
        mirror_path = _beneath(roots["mirror_root"], relative, "mirror")
        mirror_sha256, _ = _stream_identity(mirror_path, expected_bytes)
        if mirror_sha256 != source_sha256:
            raise ValueError("source mirror bytes differ")
        expected = {"revision": revision, "role": role, "path": relative, "source_sha256": source_sha256, "source_bytes": expected_bytes}
        review_roots = []
        for field, axis in (("license_receipt", "license_applicability"), ("provenance_receipt", "content_provenance"), ("redistribution_receipt", "redistribution")):
            receipt = _object(value[field], _REVIEW_FIELDS, field)
            if receipt["axis"] != axis:
                raise ValueError(f"{field} axis is invalid")
            review_roots.append(_receipt(receipt, fields=_REVIEW_FIELDS, name=field, expected=expected, registry_members=registry_members, identity_field="receipt_id"))
        independent_root = _receipt(value["independent_receipt"], fields=_VERIFY_FIELDS, name="independent_receipt", expected=expected, registry_members=registry_members, identity_field="verifier_id")
        sidecar_required = sidecar_required_text == "true"
        consumer_value = value["sidecar_consumer_receipt"]
        consumer_root: str | None = None
        if sidecar_required:
            sidecar_path = _beneath(roots["sidecar_root"], relative, "sidecar")
            sidecar_sha256, _ = _stream_identity(sidecar_path, expected_bytes)
            if sidecar_sha256 != source_sha256:
                raise ValueError("runtime sidecar bytes differ")
            consumer_root = _receipt(consumer_value, fields=_CONSUMER_FIELDS, name="sidecar_consumer_receipt", expected=expected, registry_members=registry_members, identity_field="consumer_id")
        elif consumer_value is not None:
            raise ValueError("non-runtime role cannot carry a sidecar receipt")
        compiled.append({
            "role": role, "path": relative, "bytes": expected_bytes,
            "sha256": source_sha256, "remote_identity_kind": kind,
            "remote_identity": remote, "review_receipt_roots": review_roots,
            "mirror_membership_verified": True,
            "runtime_sidecar_membership": "verified" if sidecar_required else "not_applicable",
            "sidecar_consumer_receipt_root": consumer_root,
            "independent_receipt_root": independent_root,
        })
    compiled.sort(key=lambda row: str(row["role"]))
    mirror_verifier = _load_independent_verifier(
        "verify_qwen_source_mirror.py", "verify_qwen_source_mirror",
    )
    sidecar_verifier = _load_independent_verifier(
        "verify_qwen_runtime_sidecar.py", "verify_qwen_runtime_sidecar",
    )
    mirror_receipt = mirror_verifier.verify_mirror(roots["mirror_root"], authority)
    sidecar_receipt = sidecar_verifier.verify_sidecar(roots["sidecar_root"], authority)
    mirror_generation_root = _digest(
        mirror_receipt.get("mirror_root"), "mirror generation root",
    )
    sidecar_generation_root = _digest(
        sidecar_receipt.get("sidecar_root"), "sidecar generation root",
    )
    payload = {
        "abi": _ABI, "revision": revision, "authority_root": authority_root,
        "receipt_registry_root": registry_root,
        "mirror_generation_root": mirror_generation_root,
        "runtime_sidecar_generation_root": sidecar_generation_root,
        "rows": compiled, "closure_state": "sealed",
    }
    return {"schema": _OUTPUT_SCHEMA, **payload, "closure_root": hashlib.sha256(_canonical(payload)).hexdigest()}


def _sync_directory(directory: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_receipt(receipt: dict[str, object], output: Path) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("source-closure output already exists")
    data = _canonical(receipt) + b"\n"
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=False) as target:
        temporary = Path(target.name)
        target.write(data)
        target.flush()
        os.fsync(target.fileno())
    published = False
    try:
        os.link(temporary, output)
        published = True
        _sync_directory(output.parent)
    except BaseException:
        if published:
            output.unlink(missing_ok=True)
            try:
                _sync_directory(output.parent)
            except OSError:
                pass
        raise
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--authority", type=Path, default=Path("specs/qwen-source-closure.csv"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    control = parse_input(args.input.read_bytes())
    publish_receipt(compile_closure(control, load_authority(args.authority)), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
