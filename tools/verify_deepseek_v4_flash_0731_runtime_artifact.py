#!/usr/bin/env python3
"""Independently verify a converted DeepSeek V4 Flash 0731 artifact."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import NoReturn, Sequence


def _load_layout_verifier() -> ModuleType:
    name = "_pih_deepseek_layout_verifier_for_runtime_artifact"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = Path(__file__).with_name(
        "verify_deepseek_v4_flash_0731_target_layout.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load independent target layout verifier: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


_layout = _load_layout_verifier()
_source_artifact = _layout._disposition._payload._source._semantic._artifact

_SCHEMA = "pih.deepseek_v4_flash_0731_runtime_artifact_verification.v1"
_MANIFEST_SCHEMA = "pih.deepseek_v4_flash_0731_runtime_artifact.v1"
_ARTIFACT_ABI = "deepseek_v4_runtime_artifact_v1"
_CONVERTER_ABI = "deepseek_v4_streaming_identity_converter_v1"
_MODEL_FAMILY = "deepseek_v4_flash_0731"
_SUPPORT_STATE = "hardware_evidence_open"
_VERIFICATION_SCOPE = "converted_bytes_and_source_payload_non_authorizing"
_MANIFEST_NAME = "pih.manifest.json"
_INDEX_NAME = "model.safetensors.index.json"
_RUNTIME_RECORDS_SCHEMA = "pih.deepseek_v4_flash_0731_runtime_records.v1"
_RUNTIME_RECORDS_ABI = "deepseek_v4_runtime_records_v1"
_RUNTIME_RECORDS_NAME = "pih.runtime-records.json"
_COPY_CHUNK_BYTES = 1 << 20
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_RUNTIME_RECORDS_BYTES = 128 * 1024 * 1024
_MAX_U32 = (1 << 32) - 1
_TYPE_U32 = 1
_TYPE_U64 = 2
_TYPE_BYTES = 3
_TYPE_HASH256 = 4


def _fail(message: str) -> NoReturn:
    raise ValueError(
        "invalid DeepSeek V4 Flash 0731 runtime artifact verification: "
        f"{message}"
    )


def _digest(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value == "0" * 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{name} must be a nonzero canonical lowercase SHA-256")
    return value


def _u32(value: int, name: str) -> bytes:
    if type(value) is not int or not 0 <= value <= (1 << 32) - 1:
        _fail(f"{name} is outside uint32")
    return value.to_bytes(4, "little")


def _u64(value: int, name: str) -> bytes:
    if type(value) is not int or not 0 <= value <= (1 << 64) - 1:
        _fail(f"{name} is outside uint64")
    return value.to_bytes(8, "little")


def _typed_sha256(domain: str, fields: Sequence[tuple[int, int, bytes]]) -> str:
    return _layout._typed_sha256(domain, fields)


def _canonical_json(value: object, name: str, maximum_bytes: int) -> bytes:
    try:
        source = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii", "strict")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as error:
        _fail(f"{name} cannot be encoded as canonical JSON: {error}")
    if not source or len(source) > maximum_bytes:
        _fail(f"{name} exceeds its canonical byte bound")
    return source


def _converted_shard_root(
    *, shard_layout_root: str, object_sha256: str, file_bytes: int,
    tensor_count: int, payload_bytes: int,
) -> str:
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-converted-shard:v1",
        (
            (1, _TYPE_HASH256, bytes.fromhex(_digest(shard_layout_root, "shard layout root"))),
            (2, _TYPE_HASH256, bytes.fromhex(_digest(object_sha256, "shard object SHA-256"))),
            (3, _TYPE_U64, _u64(file_bytes, "converted shard file bytes")),
            (4, _TYPE_U32, _u32(tensor_count, "converted shard tensor count")),
            (5, _TYPE_U64, _u64(payload_bytes, "converted shard payload bytes")),
        ),
    )


def _index_object_root(*, index_root: str, object_sha256: str, file_bytes: int) -> str:
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-converted-index:v1",
        (
            (1, _TYPE_HASH256, bytes.fromhex(_digest(index_root, "index root"))),
            (2, _TYPE_HASH256, bytes.fromhex(_digest(object_sha256, "index object SHA-256"))),
            (3, _TYPE_U64, _u64(file_bytes, "index object bytes")),
        ),
    )


def _runtime_record_identity(record: tuple[object, ...]) -> tuple[str, int]:
    name = record[0]
    namespace = record[3]
    if namespace == "endpoint":
        return ("embedding", _MAX_U32) if name == "embed.weight" else ("final_head", _MAX_U32)
    if type(namespace) is str and namespace.startswith("layers."):
        suffix = namespace.removeprefix("layers.")
        if not suffix.isascii() or not suffix.isdecimal():
            _fail("runtime record main-layer namespace is invalid")
        return "main_layer", int(suffix)
    if type(namespace) is str and namespace.startswith("mtp."):
        suffix = namespace.removeprefix("mtp.")
        if not suffix.isascii() or not suffix.isdecimal():
            _fail("runtime record DSpark namespace is invalid")
        return "dspark_stage", int(suffix)
    _fail("runtime record namespace is invalid")


def _storage_semantics(dtype: object) -> str:
    semantics = {
        "F32": "direct_f32_le_bits", "F16": "direct_f16_le_bits",
        "BF16": "direct_bf16_le_bits", "I8": "direct_mxfp4_e2m1_packed_bits",
        "U8": "direct_u8_bits", "F8_E4M3": "direct_fp8_e4m3_bits",
        "F8_E8M0": "direct_ue8m0_scale_bits", "I32": "direct_i32_le_bits",
        "I64": "direct_i64_le_bits", "BOOL": "direct_bool_bits",
    }
    value = semantics.get(dtype)
    if value is None:
        _fail("runtime record dtype has no bit-preserving semantics")
    return value


def _runtime_record_root(
    *, layout_record_root: str, target_logical_root: str,
    disposition_record_root: str, name: str, shard_name: str,
    namespace: str, role: str, logical_layer: int, owner_rank: int,
    dtype: str, shape: tuple[int, ...], file_begin: int, file_end: int,
    tensor_bytes: int, storage_semantics: str,
) -> str:
    packed_shape = b"".join(_u64(value, "runtime record shape") for value in shape)
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-runtime-record:v1",
        (
            (1, _TYPE_HASH256, bytes.fromhex(_digest(layout_record_root, "layout record root"))),
            (2, _TYPE_HASH256, bytes.fromhex(_digest(target_logical_root, "target logical root"))),
            (3, _TYPE_HASH256, bytes.fromhex(_digest(disposition_record_root, "disposition record root"))),
            (4, _TYPE_BYTES, name.encode("ascii", "strict")),
            (5, _TYPE_BYTES, shard_name.encode("ascii", "strict")),
            (6, _TYPE_BYTES, namespace.encode("ascii", "strict")),
            (7, _TYPE_BYTES, role.encode("ascii", "strict")),
            (8, _TYPE_U32, _u32(logical_layer, "runtime logical layer")),
            (9, _TYPE_U32, _u32(owner_rank, "runtime owner rank")),
            (10, _TYPE_BYTES, dtype.encode("ascii", "strict")),
            (11, _TYPE_U32, _u32(len(shape), "runtime record rank")),
            (12, _TYPE_BYTES, packed_shape),
            (13, _TYPE_U64, _u64(file_begin, "runtime file begin")),
            (14, _TYPE_U64, _u64(file_end, "runtime file end")),
            (15, _TYPE_U64, _u64(tensor_bytes, "runtime record bytes")),
            (16, _TYPE_BYTES, storage_semantics.encode("ascii", "strict")),
            (17, _TYPE_BYTES, b"identity_bytes"),
        ),
    )


def _runtime_records_root(
    *, layout_root: str, disposition_root: str, record_set_root: str,
    record_count: int, record_bytes: int, dspark_enabled: bool,
    world_size: int, body_sha256: str,
) -> str:
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-runtime-records:v1",
        (
            (1, _TYPE_BYTES, _RUNTIME_RECORDS_ABI.encode("ascii")),
            (2, _TYPE_HASH256, bytes.fromhex(_digest(layout_root, "layout root"))),
            (3, _TYPE_HASH256, bytes.fromhex(_digest(disposition_root, "disposition root"))),
            (4, _TYPE_HASH256, bytes.fromhex(_digest(record_set_root, "runtime record set root"))),
            (5, _TYPE_U32, _u32(record_count, "runtime record count")),
            (6, _TYPE_U64, _u64(record_bytes, "runtime record bytes")),
            (7, _TYPE_U32, _u32(1 if dspark_enabled else 0, "runtime DSpark flag")),
            (8, _TYPE_U32, _u32(world_size, "runtime world size")),
            (9, _TYPE_HASH256, bytes.fromhex(_digest(body_sha256, "runtime records body SHA-256"))),
        ),
    )


def _runtime_records_object_root(
    *, runtime_records_root: str, object_sha256: str, file_bytes: int,
) -> str:
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-runtime-records-object:v1",
        (
            (1, _TYPE_HASH256, bytes.fromhex(_digest(runtime_records_root, "runtime records root"))),
            (2, _TYPE_HASH256, bytes.fromhex(_digest(object_sha256, "runtime records SHA-256"))),
            (3, _TYPE_U64, _u64(file_bytes, "runtime records bytes")),
        ),
    )


def _runtime_record_set_root(roots: Sequence[str]) -> str:
    if not roots:
        _fail("runtime record root set is empty")
    chunks = tuple(
        _layout._root_set(
            "pih:deepseek-v4-flash-0731-runtime-record-chunk:v1",
            tuple(roots[begin : begin + 4096]),
            "runtime record root",
        )
        for begin in range(0, len(roots), 4096)
    )
    return _layout._root_set(
        "pih:deepseek-v4-flash-0731-runtime-record-set:v1",
        chunks,
        "runtime record chunk root",
    )


def _conversion_root(
    *, layout_root: str, converter_identity_root: str,
    shard_roots: Sequence[str], index_object_root: str,
    runtime_records_object_root: str,
    copied_tensor_count: int, copied_tensor_bytes: int,
) -> str:
    fields: list[tuple[int, int, bytes]] = [
        (1, _TYPE_BYTES, _CONVERTER_ABI.encode("ascii")),
        (2, _TYPE_HASH256, bytes.fromhex(_digest(layout_root, "target layout root"))),
        (3, _TYPE_HASH256, bytes.fromhex(_digest(converter_identity_root, "converter identity root"))),
        (4, _TYPE_U32, _u32(len(shard_roots), "converted shard count")),
        (5, _TYPE_HASH256, bytes.fromhex(_digest(index_object_root, "index object root"))),
        (6, _TYPE_U32, _u32(copied_tensor_count, "copied tensor count")),
        (7, _TYPE_U64, _u64(copied_tensor_bytes, "copied tensor bytes")),
        (8, _TYPE_U32, _u32(_COPY_CHUNK_BYTES, "maximum copy chunk bytes")),
        (9, _TYPE_U32, _u32(50, "source descriptor count")),
        (10, _TYPE_U32, _u32(1, "maximum output descriptors")),
        (11, _TYPE_HASH256, bytes.fromhex(_digest(runtime_records_object_root, "runtime records object root"))),
    ]
    fields.extend(
        (100 + ordinal, _TYPE_HASH256, bytes.fromhex(_digest(root, "converted shard root")))
        for ordinal, root in enumerate(shard_roots)
    )
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-identity-conversion:v1", fields
    )


def _artifact_root(*, conversion_root: str, layout_root: str, body_sha256: str) -> str:
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-runtime-artifact:v1",
        (
            (1, _TYPE_BYTES, _ARTIFACT_ABI.encode("ascii")),
            (2, _TYPE_HASH256, bytes.fromhex(_digest(conversion_root, "conversion root"))),
            (3, _TYPE_HASH256, bytes.fromhex(_digest(layout_root, "target layout root"))),
            (4, _TYPE_HASH256, bytes.fromhex(_digest(body_sha256, "manifest body SHA-256"))),
            (5, _TYPE_BYTES, _SUPPORT_STATE.encode("ascii")),
        ),
    )


def _expected_runtime_records(
    layout: dict[str, object],
    details: dict[str, object],
    dspark_enabled: bool,
    world_size: int,
) -> tuple[bytes, str, str, str]:
    rows: list[dict[str, object]] = []
    roots: list[str] = []
    for record in details["records"]:
        role, logical_layer = _runtime_record_identity(record)
        semantics = _storage_semantics(record[1])
        root = _runtime_record_root(
            layout_record_root=record[20], target_logical_root=record[10],
            disposition_record_root=record[11], name=record[0],
            shard_name=record[13], namespace=record[3], role=role,
            logical_layer=logical_layer, owner_rank=record[12],
            dtype=record[1], shape=record[2], file_begin=record[16],
            file_end=record[17], tensor_bytes=record[18],
            storage_semantics=semantics,
        )
        roots.append(root)
        rows.append({
            "name": record[0], "shard": record[13], "namespace": record[3],
            "role": role, "logical_layer": logical_layer,
            "owner_rank": record[12], "dtype": record[1],
            "shape": list(record[2]), "file_begin": record[16],
            "file_end": record[17], "tensor_bytes": record[18],
            "storage_semantics": semantics, "transform": "identity_bytes",
            "target_logical_root": record[10],
            "disposition_record_root": record[11],
            "layout_record_root": record[20], "runtime_record_root": root,
        })
    if (
        tuple(row["name"] for row in rows)
        != tuple(sorted(row["name"] for row in rows))
        or len({row["name"] for row in rows}) != len(rows)
        or len(rows) != layout["target_tensor_count"]
        or sum(row["tensor_bytes"] for row in rows)
        != layout["target_tensor_bytes"]
    ):
        _fail("independent runtime records differ from target layout")
    record_set_root = _runtime_record_set_root(tuple(roots))
    body = {
        "schema": _RUNTIME_RECORDS_SCHEMA, "abi": _RUNTIME_RECORDS_ABI,
        "model_family": _MODEL_FAMILY, "support_state": _SUPPORT_STATE,
        "layout_root": layout["layout_root"],
        "disposition_root": layout["disposition_root"],
        "dspark_enabled": dspark_enabled, "world_size": world_size,
        "record_count": layout["target_tensor_count"],
        "record_bytes": layout["target_tensor_bytes"],
        "record_set_root": record_set_root, "records": rows,
    }
    body_source = _canonical_json(body, "runtime records body", _MAX_RUNTIME_RECORDS_BYTES)
    body_sha256 = hashlib.sha256(body_source).hexdigest()
    runtime_root = _runtime_records_root(
        layout_root=layout["layout_root"],
        disposition_root=layout["disposition_root"],
        record_set_root=record_set_root,
        record_count=layout["target_tensor_count"],
        record_bytes=layout["target_tensor_bytes"],
        dspark_enabled=dspark_enabled, world_size=world_size,
        body_sha256=body_sha256,
    )
    value = dict(body)
    value["body_sha256"] = body_sha256
    value["runtime_records_root"] = runtime_root
    source = _canonical_json(value, "runtime records", _MAX_RUNTIME_RECORDS_BYTES)
    return source, runtime_root, record_set_root, body_sha256


def _entry_identity(path: Path) -> tuple[int, int, int, int, int]:
    try:
        info = os.lstat(path)
    except OSError as error:
        _fail(f"cannot inspect target member {path.name}: {error}")
    _source_artifact._reject_link(path, info, path.name)
    if not stat.S_ISREG(info.st_mode) or int(info.st_nlink) != 1:
        _fail(f"target member {path.name} is not a private regular file")
    _source_artifact._reject_sparse(info, path.name)
    return (
        int(info.st_dev), int(info.st_ino), int(info.st_mode), int(info.st_size),
        int(info.st_mtime_ns),
    )


def _root_identity(path: Path) -> tuple[object, ...]:
    if not isinstance(path, Path) or not path.is_absolute():
        _fail("target root must be an absolute Path")
    return _source_artifact._root_identity(path)


def _exact_member_names(root: Path) -> set[str]:
    try:
        return {entry.name for entry in root.iterdir()}
    except OSError as error:
        _fail(f"cannot enumerate target root: {error}")


def _open_member(
    root: Path, name: str, expected_bytes: int | None,
) -> tuple[int, tuple[int, int, int, int, int], int]:
    path = root / name
    before = _entry_identity(path)
    if not stat.S_ISREG(before[2]):
        _fail(f"target member {name} is not a concrete regular file")
    file_bytes = before[3]
    if expected_bytes is not None and file_bytes != expected_bytes:
        _fail(f"target member {name} has an unexpected length")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOINHERIT", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = _source_artifact._open_stable_member(path, flags, name)
    try:
        opened = os.fstat(descriptor)
        _source_artifact._reject_sparse(opened, name)
    except BaseException:
        os.close(descriptor)
        raise
    opened_identity = (
        int(opened.st_dev), int(opened.st_ino), int(opened.st_mode), int(opened.st_size),
        int(opened.st_mtime_ns),
    )
    if opened_identity != before or not stat.S_ISREG(opened.st_mode):
        os.close(descriptor)
        _fail(f"target member {name} changed while opening")
    return descriptor, before, file_bytes


def _read_exact(descriptor: int, length: int, name: str) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        try:
            chunk = os.read(descriptor, min(remaining, _COPY_CHUNK_BYTES))
        except InterruptedError:
            continue
        except OSError as error:
            _fail(f"cannot read target member {name}: {error}")
        if not chunk or len(chunk) > remaining:
            _fail(f"target member {name} is truncated or overlong")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_payload(
    descriptor: int, length: int, name: str, object_digest: object,
) -> str:
    digest = hashlib.sha256()
    remaining = length
    while remaining:
        try:
            chunk = os.read(descriptor, min(remaining, _COPY_CHUNK_BYTES))
        except InterruptedError:
            continue
        except OSError as error:
            _fail(f"cannot read target payload {name}: {error}")
        if not chunk or len(chunk) > remaining:
            _fail(f"target payload {name} is truncated or overlong")
        digest.update(chunk)
        object_digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def _finish_member(
    descriptor: int, root: Path, name: str,
    identity: tuple[int, int, int, int, int],
) -> None:
    try:
        trailing = os.read(descriptor, 1)
    except OSError as error:
        _fail(f"cannot finish target member {name}: {error}")
    if trailing:
        _fail(f"target member {name} has trailing bytes")
    if _entry_identity(root / name) != identity:
        _fail(f"target member {name} changed during verification")


def _verify_runtime_artifact(
    *, model_root: Path, target_root: Path, expected_model_digest: object,
    expected_semantic_root: object, expected_source_inventory_root: object,
    expected_payload_closure_root: object, expected_disposition_root: object,
    expected_layout_root: object, converter_identity_root: object,
    expected_artifact_root: object, dspark_enabled: object, world_size: object,
    leases: list[tuple[Path, int, tuple[object, ...]]],
) -> dict[str, object]:
    model_digest = _digest(expected_model_digest, "expected model digest")
    semantic_root = _digest(expected_semantic_root, "expected semantic root")
    source_root = _digest(expected_source_inventory_root, "expected source inventory root")
    payload_root = _digest(expected_payload_closure_root, "expected payload closure root")
    disposition_root = _digest(expected_disposition_root, "expected disposition root")
    layout_root = _digest(expected_layout_root, "expected target layout root")
    converter_root = _digest(converter_identity_root, "converter identity root")
    expected_root = _digest(expected_artifact_root, "expected runtime artifact root")
    if type(leases) is not list or leases:
        _fail("private retained source lease sink must be an empty concrete list")
    target_identity = _root_identity(target_root)
    details: dict[str, object] = {}
    layout = _layout._verify_deepseek_v4_flash_0731_target_layout(
        model_root=model_root,
        expected_model_digest=model_digest,
        expected_semantic_root=semantic_root,
        expected_source_inventory_root=source_root,
        expected_payload_closure_root=payload_root,
        expected_disposition_root=disposition_root,
        expected_layout_root=layout_root,
        dspark_enabled=dspark_enabled,
        world_size=world_size,
        leases=leases,
        details=details,
    )
    shard_names = tuple(shard["shard_name"] for shard in layout["shards"])
    expected_names = {
        *shard_names,
        _INDEX_NAME,
        _RUNTIME_RECORDS_NAME,
        _MANIFEST_NAME,
    }
    if _exact_member_names(target_root) != expected_names:
        _fail("target root file set differs from the canonical artifact set")
    records_by_shard: dict[str, list[tuple[object, ...]]] = {
        name: [] for name in shard_names
    }
    for record in details["records"]:
        records_by_shard[record[13]].append(record)
    headers = {header[1]: header for header in details["headers"]}
    identities: dict[str, tuple[int, int, int, int, int]] = {}
    converted_shards: list[dict[str, object]] = []
    converted_roots: list[str] = []
    for shard in layout["shards"]:
        name = shard["shard_name"]
        descriptor, identity, _ = _open_member(target_root, name, shard["file_bytes"])
        identities[name] = identity
        object_digest = hashlib.sha256()
        try:
            prefix = _read_exact(descriptor, shard["payload_file_begin"], name)
            if prefix != headers[name][4]:
                _fail(f"target shard {name} header differs from independent layout")
            object_digest.update(prefix)
            cursor = len(prefix)
            for record in sorted(records_by_shard[name], key=lambda value: value[16]):
                if record[16] != cursor or record[17] - record[16] != record[18]:
                    _fail(f"target shard {name} record coverage is noncanonical")
                observed_sha256 = _read_payload(
                    descriptor, record[18], str(record[0]), object_digest
                )
                if observed_sha256 != record[9]:
                    _fail(f"target payload differs from source closure for {record[0]}")
                cursor = record[17]
            if cursor != shard["file_bytes"]:
                _fail(f"target shard {name} records do not exactly cover the object")
            _finish_member(descriptor, target_root, name, identity)
        finally:
            os.close(descriptor)
        object_sha256 = object_digest.hexdigest()
        converted_root = _converted_shard_root(
            shard_layout_root=shard["shard_layout_root"],
            object_sha256=object_sha256,
            file_bytes=shard["file_bytes"],
            tensor_count=shard["tensor_count"],
            payload_bytes=shard["payload_bytes"],
        )
        converted_roots.append(converted_root)
        converted_shards.append({
            "namespace": shard["namespace"], "name": name,
            "file_bytes": shard["file_bytes"], "tensor_count": shard["tensor_count"],
            "payload_bytes": shard["payload_bytes"], "object_sha256": object_sha256,
            "shard_layout_root": shard["shard_layout_root"],
            "converted_shard_root": converted_root,
        })

    index_descriptor, index_identity, _ = _open_member(
        target_root, _INDEX_NAME, len(details["index_source"])
    )
    identities[_INDEX_NAME] = index_identity
    try:
        index_source = _read_exact(index_descriptor, len(details["index_source"]), _INDEX_NAME)
        if index_source != details["index_source"]:
            _fail("target index differs from independently compiled canonical index")
        _finish_member(index_descriptor, target_root, _INDEX_NAME, index_identity)
    finally:
        os.close(index_descriptor)
    index_sha256 = hashlib.sha256(index_source).hexdigest()
    index_object_root = _index_object_root(
        index_root=layout["index_root"], object_sha256=index_sha256,
        file_bytes=len(index_source),
    )
    (
        expected_runtime_records_source,
        runtime_records_root,
        runtime_record_set_root,
        runtime_records_body_sha256,
    ) = _expected_runtime_records(layout, details, dspark_enabled, world_size)
    runtime_descriptor, runtime_identity, runtime_records_bytes = _open_member(
        target_root,
        _RUNTIME_RECORDS_NAME,
        len(expected_runtime_records_source),
    )
    identities[_RUNTIME_RECORDS_NAME] = runtime_identity
    try:
        runtime_records_source = _read_exact(
            runtime_descriptor,
            runtime_records_bytes,
            _RUNTIME_RECORDS_NAME,
        )
        _finish_member(
            runtime_descriptor,
            target_root,
            _RUNTIME_RECORDS_NAME,
            runtime_identity,
        )
    finally:
        os.close(runtime_descriptor)
    if runtime_records_source != expected_runtime_records_source:
        _fail("runtime records differ from independent target layout")
    runtime_records_sha256 = hashlib.sha256(runtime_records_source).hexdigest()
    runtime_records_object_root = _runtime_records_object_root(
        runtime_records_root=runtime_records_root,
        object_sha256=runtime_records_sha256,
        file_bytes=runtime_records_bytes,
    )
    conversion_root = _conversion_root(
        layout_root=layout_root, converter_identity_root=converter_root,
        shard_roots=tuple(converted_roots), index_object_root=index_object_root,
        runtime_records_object_root=runtime_records_object_root,
        copied_tensor_count=layout["target_tensor_count"],
        copied_tensor_bytes=layout["target_tensor_bytes"],
    )
    body = {
        "schema": _MANIFEST_SCHEMA, "artifact_abi": _ARTIFACT_ABI,
        "converter_abi": _CONVERTER_ABI, "model_family": _MODEL_FAMILY,
        "support_state": _SUPPORT_STATE, "layout_root": layout_root,
        "logical_layout_root": layout["logical_layout_root"],
        "disposition_root": layout["disposition_root"],
        "source_inventory_root": layout["source_inventory_root"],
        "source_payload_closure_root": layout["source_payload_closure_root"],
        "converter_identity_root": converter_root, "conversion_root": conversion_root,
        "dspark_enabled": dspark_enabled, "world_size": world_size,
        "tensor_count": layout["target_tensor_count"],
        "tensor_bytes": layout["target_tensor_bytes"],
        "shard_count": layout["shard_count"], "shards": converted_shards,
        "index": {
            "name": _INDEX_NAME, "file_bytes": len(index_source),
            "object_sha256": index_sha256, "index_root": layout["index_root"],
            "index_object_root": index_object_root,
        },
        "runtime_records": {
            "name": _RUNTIME_RECORDS_NAME,
            "abi": _RUNTIME_RECORDS_ABI,
            "file_bytes": runtime_records_bytes,
            "object_sha256": runtime_records_sha256,
            "body_sha256": runtime_records_body_sha256,
            "record_count": layout["target_tensor_count"],
            "record_bytes": layout["target_tensor_bytes"],
            "record_set_root": runtime_record_set_root,
            "runtime_records_root": runtime_records_root,
            "runtime_records_object_root": runtime_records_object_root,
        },
        "resources": {
            "maximum_copy_chunk_bytes": _COPY_CHUNK_BYTES,
            "source_descriptor_count": 50, "maximum_output_descriptors": 1,
        },
    }
    body_source = _canonical_json(body, "runtime artifact manifest body", _MAX_MANIFEST_BYTES)
    body_sha256 = hashlib.sha256(body_source).hexdigest()
    artifact_root = _artifact_root(
        conversion_root=conversion_root, layout_root=layout_root,
        body_sha256=body_sha256,
    )
    expected_manifest = dict(body)
    expected_manifest["artifact_root"] = artifact_root
    expected_manifest["manifest_body_sha256"] = body_sha256
    expected_manifest_source = _canonical_json(
        expected_manifest, "runtime artifact manifest", _MAX_MANIFEST_BYTES
    )
    manifest_descriptor, manifest_identity, manifest_bytes = _open_member(
        target_root, _MANIFEST_NAME, None
    )
    identities[_MANIFEST_NAME] = manifest_identity
    if not 0 < manifest_bytes <= _MAX_MANIFEST_BYTES:
        os.close(manifest_descriptor)
        _fail("runtime artifact manifest exceeds its byte bound")
    try:
        manifest_source = _read_exact(manifest_descriptor, manifest_bytes, _MANIFEST_NAME)
        _finish_member(manifest_descriptor, target_root, _MANIFEST_NAME, manifest_identity)
    finally:
        os.close(manifest_descriptor)
    try:
        manifest = json.loads(manifest_source)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as error:
        _fail(f"runtime artifact manifest is malformed: {error}")
    if manifest_source != _canonical_json(manifest, "runtime artifact manifest", _MAX_MANIFEST_BYTES):
        _fail("runtime artifact manifest is not canonical JSON")
    if manifest != expected_manifest or manifest_source != expected_manifest_source:
        _fail("runtime artifact manifest differs from independently verified bytes")

    _source_artifact._revalidate_leases(leases)
    if _root_identity(target_root) != target_identity:
        _fail("target root or parent identity changed during verification")
    if _exact_member_names(target_root) != expected_names:
        _fail("target root file set changed during verification")
    for name, identity in identities.items():
        if _entry_identity(target_root / name) != identity:
            _fail(f"target member {name} changed after verification")
    if artifact_root != expected_root:
        _fail("expected runtime artifact root differs from verified artifact")
    return {
        "schema": _SCHEMA, "artifact_abi": _ARTIFACT_ABI,
        "converter_abi": _CONVERTER_ABI, "model_family": _MODEL_FAMILY,
        "artifact_root": artifact_root, "conversion_root": conversion_root,
        "layout_root": layout_root, "disposition_root": layout["disposition_root"],
        "converter_identity_root": converter_root,
        "manifest_body_sha256": body_sha256,
        "manifest_sha256": hashlib.sha256(manifest_source).hexdigest(),
        "manifest_bytes": len(manifest_source), "index_root": layout["index_root"],
        "index_object_root": index_object_root,
        "index_sha256": index_sha256, "index_bytes": len(index_source),
        "runtime_records_root": runtime_records_root,
        "runtime_record_set_root": runtime_record_set_root,
        "runtime_records_object_root": runtime_records_object_root,
        "runtime_records_body_sha256": runtime_records_body_sha256,
        "runtime_records_sha256": runtime_records_sha256,
        "runtime_records_bytes": runtime_records_bytes,
        "copied_tensor_count": layout["target_tensor_count"],
        "copied_tensor_bytes": layout["target_tensor_bytes"],
        "maximum_copy_chunk_bytes": _COPY_CHUNK_BYTES,
        "source_descriptor_count": 50, "maximum_output_descriptors": 1,
        "dspark_enabled": dspark_enabled, "world_size": world_size,
        "shards": converted_shards, "verification_scope": _VERIFICATION_SCOPE,
        "support_state": _SUPPORT_STATE,
    }


def verify_deepseek_v4_flash_0731_runtime_artifact(
    *, model_root: Path, target_root: Path, expected_model_digest: object,
    expected_semantic_root: object, expected_source_inventory_root: object,
    expected_payload_closure_root: object, expected_disposition_root: object,
    expected_layout_root: object, converter_identity_root: object,
    expected_artifact_root: object, dspark_enabled: object, world_size: object,
) -> dict[str, object]:
    leases: list[tuple[Path, int, tuple[object, ...]]] = []
    try:
        return _verify_runtime_artifact(
            model_root=model_root, target_root=target_root,
            expected_model_digest=expected_model_digest,
            expected_semantic_root=expected_semantic_root,
            expected_source_inventory_root=expected_source_inventory_root,
            expected_payload_closure_root=expected_payload_closure_root,
            expected_disposition_root=expected_disposition_root,
            expected_layout_root=expected_layout_root,
            converter_identity_root=converter_identity_root,
            expected_artifact_root=expected_artifact_root,
            dspark_enabled=dspark_enabled, world_size=world_size, leases=leases,
        )
    finally:
        _source_artifact._close_leases(leases)


def _canonical(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Independently verify a converted DeepSeek runtime artifact."
    )
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--expected-model-digest", required=True)
    parser.add_argument("--expected-semantic-root", required=True)
    parser.add_argument("--expected-source-inventory-root", required=True)
    parser.add_argument("--expected-payload-closure-root", required=True)
    parser.add_argument("--expected-disposition-root", required=True)
    parser.add_argument("--expected-layout-root", required=True)
    parser.add_argument("--converter-identity-root", required=True)
    parser.add_argument("--expected-artifact-root", required=True)
    parser.add_argument("--dspark-enabled", required=True, choices=("false", "true"))
    parser.add_argument("--world-size", required=True, type=int)
    arguments = parser.parse_args(argv)
    result = verify_deepseek_v4_flash_0731_runtime_artifact(
        model_root=Path(arguments.model_root), target_root=Path(arguments.target_root),
        expected_model_digest=arguments.expected_model_digest,
        expected_semantic_root=arguments.expected_semantic_root,
        expected_source_inventory_root=arguments.expected_source_inventory_root,
        expected_payload_closure_root=arguments.expected_payload_closure_root,
        expected_disposition_root=arguments.expected_disposition_root,
        expected_layout_root=arguments.expected_layout_root,
        converter_identity_root=arguments.converter_identity_root,
        expected_artifact_root=arguments.expected_artifact_root,
        dspark_enabled=arguments.dspark_enabled == "true", world_size=arguments.world_size,
    )
    sys.stdout.buffer.write(_canonical(result).encode("ascii") + b"\n")
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
