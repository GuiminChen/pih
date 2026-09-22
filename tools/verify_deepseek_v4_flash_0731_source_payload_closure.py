#!/usr/bin/env python3
"""Independently verify DeepSeek V4 Flash 0731 source payload bytes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import NoReturn, Sequence


def _load_source_verifier() -> ModuleType:
    name = "_pih_deepseek_source_inventory_verifier_for_payload"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = Path(__file__).with_name(
        "verify_deepseek_v4_flash_0731_source_inventory.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load independent source verifier: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


_source = _load_source_verifier()

_SCHEMA = (
    "pih.deepseek_v4_flash_0731_"
    "source_payload_closure_verification.v1"
)
_ABI = "deepseek_v4_flash_0731_source_payload_closure_v1"
_MODEL_FAMILY = "deepseek_v4_flash_0731"
_VERIFICATION_SCOPE = "exact_source_tensor_payload_bytes_non_authorizing"
_SUPPORT_STATE = "hardware_evidence_open"
_SOURCE_ABI = "deepseek_v4_flash_0731_source_inventory_v1"
_SOURCE_SCOPE = (
    "exact_pinned_source_name_set_and_header_semantics_non_authorizing"
)

_TENSOR_COUNT = 72_317
_TENSOR_BYTES = 166_878_536_440
_SHARD_COUNT = 48
_EXPECTED_SHARDS = tuple(
    f"model-{ordinal:05d}-of-00048.safetensors"
    for ordinal in range(1, _SHARD_COUNT + 1)
)
_EXPECTED_SHARD_SET = frozenset(_EXPECTED_SHARDS)
_READ_CHUNK_BYTES = 1 << 20
_MAX_U32 = (1 << 32) - 1
_MAX_U64 = (1 << 64) - 1
_TYPE_U32 = 1
_TYPE_U64 = 2
_TYPE_BYTES = 3
_TYPE_HASH256 = 4


def _fail(message: str) -> NoReturn:
    raise ValueError(
        "invalid DeepSeek V4 Flash 0731 source payload verification: "
        f"{message}"
    )


def _u32(value: int, name: str) -> bytes:
    if type(value) is not int or not 0 <= value <= _MAX_U32:
        _fail(f"{name} is outside uint32")
    return value.to_bytes(4, "little")


def _u64(value: int, name: str) -> bytes:
    if type(value) is not int or not 0 <= value <= _MAX_U64:
        _fail(f"{name} is outside uint64")
    return value.to_bytes(8, "little")


def _digest(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or value == "0" * 64
    ):
        _fail(f"{name} must be a nonzero canonical lowercase SHA-256")
    return value


def _typed_sha256(
    domain: str,
    fields: Sequence[tuple[int, int, bytes]],
) -> str:
    try:
        domain_bytes = domain.encode("ascii", "strict")
    except UnicodeEncodeError as error:
        _fail(f"typed hash domain is not ASCII: {error}")
    if (
        not domain_bytes
        or len(domain_bytes) > 127
        or not domain.startswith("pih:")
        or any(byte < 0x20 or byte > 0x7E for byte in domain_bytes)
        or len(fields) > _MAX_U32
    ):
        _fail("typed hash domain or field count is invalid")
    digest = hashlib.sha256()
    digest.update(domain_bytes)
    digest.update(b"\0")
    digest.update(_u32(len(fields), "typed field count"))
    previous = 0
    for field_id, field_type, value in fields:
        if (
            type(field_id) is not int
            or not previous < field_id <= 0xFFFF
            or field_type
            not in {_TYPE_U32, _TYPE_U64, _TYPE_BYTES, _TYPE_HASH256}
            or type(value) is not bytes
        ):
            _fail("typed hash field is invalid or unordered")
        expected = {
            _TYPE_U32: 4,
            _TYPE_U64: 8,
            _TYPE_HASH256: 32,
        }.get(field_type)
        if expected is not None and len(value) != expected:
            _fail("typed hash fixed-width field has the wrong size")
        digest.update(field_id.to_bytes(2, "little"))
        digest.update(bytes((field_type,)))
        digest.update(_u64(len(value), "typed field length"))
        digest.update(value)
        previous = field_id
    return digest.hexdigest()


def _sha256_descriptor_range(
    descriptor: int,
    file_begin: int,
    file_end: int,
    name: str,
) -> str:
    if (
        type(descriptor) is not int
        or descriptor < 0
        or type(file_begin) is not int
        or type(file_end) is not int
        or not 0 <= file_begin < file_end <= _MAX_U64
        or type(name) is not str
        or not name
    ):
        _fail("payload descriptor range is invalid")
    try:
        info = os.fstat(descriptor)
    except OSError as error:
        _fail(f"payload descriptor is unavailable for {name}: {error}")
    if file_end > int(info.st_size):
        _fail(f"payload range exceeds retained descriptor for {name}")
    try:
        os.lseek(descriptor, file_begin, os.SEEK_SET)
    except OSError as error:
        _fail(f"cannot seek retained payload descriptor for {name}: {error}")
    digest = hashlib.sha256()
    remaining = file_end - file_begin
    while remaining:
        requested = min(_READ_CHUNK_BYTES, remaining)
        try:
            chunk = os.read(descriptor, requested)
        except InterruptedError:
            continue
        except OSError as error:
            _fail(f"cannot read retained payload descriptor for {name}: {error}")
        if not chunk or len(chunk) > requested:
            _fail(f"retained descriptor ended during payload read: {name}")
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def _payload_record_root(
    *,
    source_record_root: str,
    artifact_object_root: str,
    name: str,
    shard_name: str,
    data_begin: int,
    data_end: int,
    file_begin: int,
    file_end: int,
    payload_sha256: str,
) -> str:
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-source-payload-record:v1",
        (
            (
                1,
                _TYPE_HASH256,
                bytes.fromhex(_digest(source_record_root, "source record root")),
            ),
            (
                2,
                _TYPE_HASH256,
                bytes.fromhex(
                    _digest(artifact_object_root, "artifact shard object root")
                ),
            ),
            (3, _TYPE_BYTES, name.encode("ascii", "strict")),
            (4, _TYPE_BYTES, shard_name.encode("ascii", "strict")),
            (5, _TYPE_U64, _u64(data_begin, "payload relative begin")),
            (6, _TYPE_U64, _u64(data_end, "payload relative end")),
            (7, _TYPE_U64, _u64(file_begin, "payload file begin")),
            (8, _TYPE_U64, _u64(file_end, "payload file end")),
            (
                9,
                _TYPE_HASH256,
                bytes.fromhex(_digest(payload_sha256, "payload SHA-256")),
            ),
        ),
    )


def _payload_record_set_root(record_roots: Sequence[str]) -> str:
    fields: list[tuple[int, int, bytes]] = [
        (1, _TYPE_U32, _u32(len(record_roots), "payload record count"))
    ]
    fields.extend(
        (
            100 + ordinal,
            _TYPE_HASH256,
            bytes.fromhex(_digest(root, "payload record root")),
        )
        for ordinal, root in enumerate(record_roots)
    )
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-source-payload-record-set:v1",
        fields,
    )


def _payload_shard_root(
    *,
    shard_name: str,
    artifact_object_root: str,
    source_shard_inventory_root: str,
    tensor_count: int,
    tensor_bytes: int,
    payload_record_set_root: str,
) -> str:
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-source-payload-shard:v1",
        (
            (1, _TYPE_BYTES, shard_name.encode("ascii", "strict")),
            (
                2,
                _TYPE_HASH256,
                bytes.fromhex(
                    _digest(artifact_object_root, "artifact shard object root")
                ),
            ),
            (
                3,
                _TYPE_HASH256,
                bytes.fromhex(
                    _digest(
                        source_shard_inventory_root,
                        "source shard inventory root",
                    )
                ),
            ),
            (4, _TYPE_U32, _u32(tensor_count, "payload shard tensor count")),
            (5, _TYPE_U64, _u64(tensor_bytes, "payload shard tensor bytes")),
            (
                6,
                _TYPE_HASH256,
                bytes.fromhex(
                    _digest(payload_record_set_root, "payload record-set root")
                ),
            ),
        ),
    )


def _payload_closure_root(
    *,
    artifact_model_digest: str,
    semantic_root: str,
    source_inventory_root: str,
    shard_roots: Sequence[str],
    tensor_count: int,
    tensor_bytes: int,
) -> str:
    fields: list[tuple[int, int, bytes]] = [
        (1, _TYPE_BYTES, _ABI.encode("ascii")),
        (2, _TYPE_BYTES, _MODEL_FAMILY.encode("ascii")),
        (
            3,
            _TYPE_HASH256,
            bytes.fromhex(_digest(artifact_model_digest, "artifact model digest")),
        ),
        (
            4,
            _TYPE_HASH256,
            bytes.fromhex(_digest(semantic_root, "semantic root")),
        ),
        (
            5,
            _TYPE_HASH256,
            bytes.fromhex(_digest(source_inventory_root, "source inventory root")),
        ),
        (6, _TYPE_U32, _u32(len(shard_roots), "payload shard count")),
        (7, _TYPE_U64, _u64(tensor_count, "payload tensor count")),
        (8, _TYPE_U64, _u64(tensor_bytes, "payload tensor bytes")),
        (9, _TYPE_BYTES, _VERIFICATION_SCOPE.encode("ascii")),
        (10, _TYPE_BYTES, _SUPPORT_STATE.encode("ascii")),
    ]
    fields.extend(
        (
            100 + ordinal,
            _TYPE_HASH256,
            bytes.fromhex(_digest(root, "payload shard root")),
        )
        for ordinal, root in enumerate(shard_roots)
    )
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-source-payload-closure:v1",
        fields,
    )


def _validate_source_projection(value: object) -> dict[str, object]:
    if (
        type(value) is not dict
        or value.get("abi") != _SOURCE_ABI
        or value.get("model_family") != _MODEL_FAMILY
        or value.get("verification_scope") != _SOURCE_SCOPE
        or value.get("support_state") != _SUPPORT_STATE
        or value.get("shard_count") != _SHARD_COUNT
        or value.get("tensor_count") != _TENSOR_COUNT
        or value.get("tensor_bytes") != _TENSOR_BYTES
    ):
        _fail("source inventory projection has invalid ABI or geometry")
    for key in (
        "artifact_model_digest",
        "semantic_root",
        "source_inventory_root",
    ):
        _digest(value.get(key), f"source projection {key}")
    shards = value.get("shards")
    if (
        type(shards) is not list
        or len(shards) != _SHARD_COUNT
        or tuple(
            shard.get("shard_name")
            for shard in shards
            if type(shard) is dict
        )
        != _EXPECTED_SHARDS
    ):
        _fail("source projection shard set is invalid or noncanonical")
    return value


def _compile_payload_projection(
    source_projection: object,
    source_details: object,
    descriptors: object,
    expected_payload_closure_root: object,
    _details_sink: dict[str, object] | None = None,
) -> dict[str, object]:
    source_value = _validate_source_projection(source_projection)
    expected_root = _digest(
        expected_payload_closure_root, "expected payload closure root"
    )
    if _details_sink is not None and (
        type(_details_sink) is not dict or _details_sink
    ):
        _fail("private payload detail sink must be an empty concrete dict")
    if (
        type(descriptors) is not dict
        or set(descriptors)
        != {"config.json", "model.safetensors.index.json", *_EXPECTED_SHARDS}
        or any(type(value) is not int or value < 0 for value in descriptors.values())
    ):
        _fail("retained payload descriptor set differs from the exact artifact")
    if type(source_details) is not dict or set(source_details) != {
        "artifact_model_digest",
        "records",
        "semantic_root",
        "shards",
        "source_inventory_root",
    }:
        _fail("source verifier returned an invalid private detail set")
    for key in (
        "artifact_model_digest",
        "semantic_root",
        "source_inventory_root",
    ):
        if source_details[key] != source_value[key]:
            _fail("source private details differ from the public antecedent")

    public_source_shards = source_value["shards"]
    private_shards = source_details["shards"]
    if (
        type(private_shards) is not tuple
        or len(private_shards) != _SHARD_COUNT
        or tuple(
            shard[0]
            for shard in private_shards
            if type(shard) is tuple and len(shard) == 5
        )
        != _EXPECTED_SHARDS
    ):
        _fail("source private shard set is invalid or noncanonical")
    shard_inputs: dict[str, tuple[str, str, int, int]] = {}
    for private, public in zip(
        private_shards, public_source_shards, strict=True
    ):
        if type(private) is not tuple or len(private) != 5:
            _fail("source private shard record is malformed")
        shard_name, artifact_root, source_shard_root, count, byte_count = private
        if (
            type(public) is not dict
            or public.get("shard_name") != shard_name
            or public.get("shard_inventory_root") != source_shard_root
            or public.get("tensor_count") != count
            or public.get("tensor_bytes") != byte_count
            or type(count) is not int
            or count <= 0
            or type(byte_count) is not int
            or byte_count <= 0
        ):
            _fail("source private shard differs from its public receipt")
        shard_inputs[shard_name] = (
            _digest(artifact_root, "artifact shard object root"),
            _digest(source_shard_root, "source shard inventory root"),
            count,
            byte_count,
        )

    raw_records = source_details["records"]
    if (
        type(raw_records) is not tuple
        or len(raw_records) != _TENSOR_COUNT
        or tuple(
            record[0]
            for record in raw_records
            if type(record) is tuple and len(record) == 12
        )
        != tuple(
            sorted(
                record[0]
                for record in raw_records
                if type(record) is tuple and len(record) == 12
            )
        )
    ):
        _fail("source private record set is invalid or noncanonical")
    roots_by_shard: dict[str, list[str]] = {
        name: [] for name in _EXPECTED_SHARDS
    }
    bytes_by_shard = {name: 0 for name in _EXPECTED_SHARDS}
    names_seen: set[str] = set()
    private_records: list[tuple[object, ...]] = []
    for raw_record in raw_records:
        if type(raw_record) is not tuple or len(raw_record) != 12:
            _fail("source private record is malformed")
        (
            name,
            shard_name,
            dtype,
            shape,
            data_begin,
            data_end,
            semantic_tensor_root,
            namespace,
            role_suffix,
            file_begin,
            file_end,
            source_record_root,
        ) = raw_record
        if (
            type(name) is not str
            or not name
            or name in names_seen
            or type(shard_name) is not str
            or shard_name not in _EXPECTED_SHARD_SET
            or type(dtype) is not str
            or dtype not in _source._DTYPE_ELEMENT_BYTES
            or type(shape) is not tuple
            or not 1 <= len(shape) <= 8
            or any(
                type(dimension) is not int
                or not 0 < dimension <= _MAX_U64
                for dimension in shape
            )
            or type(semantic_tensor_root) is not str
            or type(namespace) is not str
            or not namespace
            or type(role_suffix) is not str
            or not role_suffix
            or type(data_begin) is not int
            or type(data_end) is not int
            or type(file_begin) is not int
            or type(file_end) is not int
            or not 0 <= data_begin < data_end <= _MAX_U64
            or not 0 <= file_begin < file_end <= _MAX_U64
            or data_end - data_begin != file_end - file_begin
        ):
            _fail("source private payload record geometry is invalid")
        try:
            name.encode("ascii", "strict")
            dtype.encode("ascii", "strict")
            namespace.encode("ascii", "strict")
            role_suffix.encode("ascii", "strict")
        except UnicodeEncodeError as error:
            _fail(f"source private payload identity is not ASCII: {error}")
        source_draft = _source._SourceRecord(
            name=name,
            shard_name=shard_name,
            dtype=dtype,
            shape=shape,
            data_begin=data_begin,
            data_end=data_end,
            semantic_tensor_root=_digest(
                semantic_tensor_root, "semantic tensor root"
            ),
            namespace=namespace,
            role_suffix=role_suffix,
            file_begin=file_begin,
            file_end=file_end,
            record_root="0" * 64,
        )
        if source_record_root != _source._source_record_root(source_draft):
            _fail("source private record root differs from independent replay")
        names_seen.add(name)
        artifact_root = shard_inputs[shard_name][0]
        payload_sha256 = _sha256_descriptor_range(
            descriptors[shard_name], file_begin, file_end, name
        )
        record_root = _payload_record_root(
            source_record_root=_digest(
                source_record_root, "source record root"
            ),
            artifact_object_root=artifact_root,
            name=name,
            shard_name=shard_name,
            data_begin=data_begin,
            data_end=data_end,
            file_begin=file_begin,
            file_end=file_end,
            payload_sha256=payload_sha256,
        )
        roots_by_shard[shard_name].append(record_root)
        bytes_by_shard[shard_name] += file_end - file_begin
        private_records.append(
            (
                name,
                shard_name,
                dtype,
                shape,
                namespace,
                role_suffix,
                semantic_tensor_root,
                data_begin,
                data_end,
                file_begin,
                file_end,
                source_record_root,
                artifact_root,
                payload_sha256,
                record_root,
            )
        )

    shard_receipts: list[dict[str, object]] = []
    shard_roots: list[str] = []
    for shard_name in _EXPECTED_SHARDS:
        artifact_root, source_shard_root, count, byte_count = shard_inputs[
            shard_name
        ]
        roots = roots_by_shard[shard_name]
        if len(roots) != count or bytes_by_shard[shard_name] != byte_count:
            _fail("payload shard records differ from source geometry")
        record_set_root = _payload_record_set_root(roots)
        shard_root = _payload_shard_root(
            shard_name=shard_name,
            artifact_object_root=artifact_root,
            source_shard_inventory_root=source_shard_root,
            tensor_count=count,
            tensor_bytes=byte_count,
            payload_record_set_root=record_set_root,
        )
        shard_roots.append(shard_root)
        shard_receipts.append(
            {
                "shard_name": shard_name,
                "artifact_object_root": artifact_root,
                "source_shard_inventory_root": source_shard_root,
                "tensor_count": count,
                "tensor_bytes": byte_count,
                "payload_record_set_root": record_set_root,
                "payload_shard_root": shard_root,
            }
        )
    if (
        len(names_seen) != _TENSOR_COUNT
        or sum(bytes_by_shard.values()) != _TENSOR_BYTES
    ):
        _fail("payload record set differs from exact global geometry")
    payload_root = _payload_closure_root(
        artifact_model_digest=source_value["artifact_model_digest"],
        semantic_root=source_value["semantic_root"],
        source_inventory_root=source_value["source_inventory_root"],
        shard_roots=shard_roots,
        tensor_count=_TENSOR_COUNT,
        tensor_bytes=_TENSOR_BYTES,
    )
    if payload_root != expected_root:
        _fail(
            "expected payload closure root differs from independently replayed root"
        )
    result = {
        "schema": _SCHEMA,
        "abi": _ABI,
        "model_family": _MODEL_FAMILY,
        "payload_closure_root": payload_root,
        "artifact_model_digest": source_value["artifact_model_digest"],
        "semantic_root": source_value["semantic_root"],
        "source_inventory_root": source_value["source_inventory_root"],
        "shard_count": len(shard_receipts),
        "shards": shard_receipts,
        "tensor_count": len(names_seen),
        "tensor_bytes": sum(bytes_by_shard.values()),
        "verification_scope": _VERIFICATION_SCOPE,
        "support_state": _SUPPORT_STATE,
    }
    if _details_sink is not None:
        _details_sink.update(
            {
                "artifact_model_digest": source_value[
                    "artifact_model_digest"
                ],
                "semantic_root": source_value["semantic_root"],
                "source_inventory_root": source_value[
                    "source_inventory_root"
                ],
                "payload_closure_root": payload_root,
                "records": tuple(private_records),
            }
        )
    return result


def _verify_deepseek_v4_flash_0731_source_payload_closure(
    *,
    model_root: Path,
    expected_model_digest: object,
    expected_semantic_root: object,
    expected_source_inventory_root: object,
    expected_payload_closure_root: object,
    leases: list[tuple[Path, int, tuple[object, ...]]],
    details: dict[str, object],
) -> dict[str, object]:
    model_digest = _digest(expected_model_digest, "expected model digest")
    semantic_root = _digest(expected_semantic_root, "expected semantic root")
    source_inventory_root = _digest(
        expected_source_inventory_root, "expected source inventory root"
    )
    payload_closure_root = _digest(
        expected_payload_closure_root, "expected payload closure root"
    )
    if type(leases) is not list or leases:
        _fail("private retained lease sink must be an empty concrete list")
    if type(details) is not dict or details:
        _fail("private payload detail sink must be an empty concrete dict")
    source_details: dict[str, object] = {}
    source_projection = (
        _source._verify_deepseek_v4_flash_0731_source_inventory(
            model_root=model_root,
            expected_model_digest=model_digest,
            expected_semantic_root=semantic_root,
            expected_source_inventory_root=source_inventory_root,
            leases=leases,
            details=source_details,
        )
    )
    _source._semantic._require_exact_safetensors_set(model_root)
    payload_root_before = _source._semantic._artifact._root_identity(
        model_root
    )
    _source._semantic._artifact._revalidate_leases(leases)
    descriptors: dict[str, int] = {}
    for path, descriptor, _identity in leases:
        if path.name in descriptors:
            _fail("retained payload descriptor set has a duplicate basename")
        descriptors[path.name] = descriptor
    candidate_details: dict[str, object] = {}
    result = _compile_payload_projection(
        source_projection,
        source_details,
        descriptors,
        payload_closure_root,
        candidate_details,
    )
    _source._semantic._require_exact_safetensors_set(model_root)
    _source._semantic._artifact._revalidate_leases(leases)
    _source._semantic._require_exact_safetensors_set(model_root)
    if (
        _source._semantic._artifact._root_identity(model_root)
        != payload_root_before
    ):
        _fail("artifact root or parent identity changed during payload replay")
    details.update(candidate_details)
    return result


def verify_deepseek_v4_flash_0731_source_payload_closure(
    *,
    model_root: Path,
    expected_model_digest: object,
    expected_semantic_root: object,
    expected_source_inventory_root: object,
    expected_payload_closure_root: object,
) -> dict[str, object]:
    leases: list[tuple[Path, int, tuple[object, ...]]] = []
    details: dict[str, object] = {}
    try:
        return _verify_deepseek_v4_flash_0731_source_payload_closure(
            model_root=model_root,
            expected_model_digest=expected_model_digest,
            expected_semantic_root=expected_semantic_root,
            expected_source_inventory_root=expected_source_inventory_root,
            expected_payload_closure_root=expected_payload_closure_root,
            leases=leases,
            details=details,
        )
    finally:
        _source._semantic._artifact._close_leases(leases)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Independently verify DeepSeek V4 Flash 0731 source payload bytes."
        )
    )
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--expected-model-digest", required=True)
    parser.add_argument("--expected-semantic-root", required=True)
    parser.add_argument("--expected-source-inventory-root", required=True)
    parser.add_argument("--expected-payload-closure-root", required=True)
    arguments = parser.parse_args(argv)
    result = verify_deepseek_v4_flash_0731_source_payload_closure(
        model_root=Path(arguments.model_root),
        expected_model_digest=arguments.expected_model_digest,
        expected_semantic_root=arguments.expected_semantic_root,
        expected_source_inventory_root=arguments.expected_source_inventory_root,
        expected_payload_closure_root=arguments.expected_payload_closure_root,
    )
    sys.stdout.buffer.write(_canonical(result).encode("ascii") + b"\n")
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
