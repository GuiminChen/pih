#!/usr/bin/env python3
"""Independently verify DeepSeek V4 Flash 0731 canonical target layout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import NoReturn, Sequence


def _load_disposition_verifier() -> ModuleType:
    name = "_pih_deepseek_disposition_verifier_for_target_layout"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = Path(__file__).with_name(
        "verify_deepseek_v4_flash_0731_target_disposition.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot load independent target disposition verifier: {path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


_disposition = _load_disposition_verifier()

_SCHEMA = (
    "pih.deepseek_v4_flash_0731_target_layout_verification.v1"
)
_ABI = "deepseek_v4_flash_0731_target_layout_v1"
_MODEL_FAMILY = "deepseek_v4_flash_0731"
_VERIFICATION_SCOPE = "canonical_target_safetensors_layout_non_authorizing"
_SUPPORT_STATE = "hardware_evidence_open"
_DISPOSITION_ABI = "deepseek_v4_flash_0731_target_disposition_v1"
_DISPOSITION_SCOPE = "total_source_to_target_logical_disposition_non_authorizing"

_SOURCE_TENSOR_COUNT = 72_317
_SOURCE_TENSOR_BYTES = 166_878_536_440
_ENABLED_TARGET_COUNT = 72_317
_ENABLED_TARGET_BYTES = 166_878_536_440
_DISABLED_TARGET_COUNT = 67_612
_DISABLED_TARGET_BYTES = 156_015_698_140
_ENABLED_SHARD_COUNT = 47
_DISABLED_SHARD_COUNT = 44
_ENABLED_TOTAL_HEADER_BYTES = 7_412_504
_ENABLED_TOTAL_SHARD_FILE_BYTES = 166_885_949_320
_ENABLED_INDEX_BYTES = 4_772_827
_ENABLED_TOTAL_ARTIFACT_BYTES = 166_890_722_147
_DISABLED_TOTAL_HEADER_BYTES = 6_946_944
_DISABLED_TOTAL_SHARD_FILE_BYTES = 156_022_645_436
_DISABLED_INDEX_BYTES = 4_495_575
_DISABLED_TOTAL_ARTIFACT_BYTES = 156_027_141_011
_MAX_SHARD_HEADER_BYTES = 16 * 1024 * 1024
_MAX_INDEX_BYTES = 64 * 1024 * 1024
_MAX_TENSORS_PER_SHARD = 4_096
_MAX_U64 = (1 << 64) - 1
_TYPE_U32 = 1
_TYPE_U64 = 2
_TYPE_BYTES = 3
_TYPE_HASH256 = 4


def _fail(message: str) -> NoReturn:
    raise ValueError(
        "invalid DeepSeek V4 Flash 0731 target layout verification: "
        f"{message}"
    )


def _u32(value: int, name: str) -> bytes:
    if type(value) is not int or not 0 <= value <= (1 << 32) - 1:
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
    domain: str, fields: Sequence[tuple[int, int, bytes]]
) -> str:
    return _disposition._payload._typed_sha256(domain, fields)


def _root_set(domain: str, roots: Sequence[str], label: str) -> str:
    fields: list[tuple[int, int, bytes]] = [
        (1, _TYPE_U32, _u32(len(roots), f"{label} count"))
    ]
    fields.extend(
        (
            100 + ordinal,
            _TYPE_HASH256,
            bytes.fromhex(_digest(root, label)),
        )
        for ordinal, root in enumerate(roots)
    )
    return _typed_sha256(domain, fields)


def _checked_add(left: int, right: int, name: str) -> int:
    if (
        type(left) is not int
        or type(right) is not int
        or left < 0
        or right < 0
        or left > _MAX_U64 - right
    ):
        _fail(f"{name} overflows uint64")
    return left + right


def _checked_mul(left: int, right: int, name: str) -> int:
    if (
        type(left) is not int
        or type(right) is not int
        or left < 0
        or right < 0
        or (left != 0 and right > _MAX_U64 // left)
    ):
        _fail(f"{name} overflows uint64")
    return left * right


def _canonical_json(value: object, name: str, maximum_bytes: int) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii", "strict")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as error:
        _fail(f"{name} cannot be encoded as canonical JSON: {error}")
    if not encoded or len(encoded) > maximum_bytes:
        _fail(f"{name} exceeds its canonical byte bound")
    return encoded


def _target_shard_name(namespace: str) -> str:
    expected = tuple(
        name
        for name, _members in _disposition._payload._source._expected_name_groups()
    )
    if type(namespace) is not str or namespace not in expected:
        _fail("target namespace is outside the frozen source family")
    return f"model-{namespace.replace('.', '-')}.safetensors"


def _tensor_payload_bytes(dtype: str, shape: tuple[int, ...]) -> int:
    element_bytes = _disposition._payload._source._DTYPE_ELEMENT_BYTES
    if type(dtype) is not str or dtype not in element_bytes:
        _fail("target tensor dtype is invalid")
    if type(shape) is not tuple or not 1 <= len(shape) <= 8:
        _fail("target tensor shape is invalid")
    elements = 1
    for dimension in shape:
        if type(dimension) is not int or not 0 < dimension <= _MAX_U64:
            _fail("target tensor dimension is invalid")
        elements = _checked_mul(elements, dimension, "target tensor elements")
    return _checked_mul(elements, element_bytes[dtype], "target tensor bytes")


def _weight_map_entry_root(
    name: str, shard_name: str, target_logical_root: str
) -> str:
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-target-weight-map-entry:v1",
        (
            (1, _TYPE_BYTES, name.encode("ascii", "strict")),
            (2, _TYPE_BYTES, shard_name.encode("ascii", "strict")),
            (
                3,
                _TYPE_HASH256,
                bytes.fromhex(_digest(target_logical_root, "target logical root")),
            ),
        ),
    )


def _layout_record_root(
    *,
    source_payload_record_root: str,
    target_logical_root: str,
    source_payload_sha256: str,
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    namespace: str,
    shard_name: str,
    data_begin: int,
    data_end: int,
    file_begin: int,
    file_end: int,
    tensor_bytes: int,
    header_prefix_sha256: str,
) -> str:
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-target-layout-record:v1",
        (
            (
                1,
                _TYPE_HASH256,
                bytes.fromhex(
                    _digest(source_payload_record_root, "source payload record root")
                ),
            ),
            (
                2,
                _TYPE_HASH256,
                bytes.fromhex(_digest(target_logical_root, "target logical root")),
            ),
            (
                3,
                _TYPE_HASH256,
                bytes.fromhex(_digest(source_payload_sha256, "payload SHA-256")),
            ),
            (4, _TYPE_BYTES, name.encode("ascii", "strict")),
            (5, _TYPE_BYTES, dtype.encode("ascii", "strict")),
            (6, _TYPE_U32, _u32(len(shape), "target layout rank")),
            (
                7,
                _TYPE_BYTES,
                b"".join(_u64(value, "shape dimension") for value in shape),
            ),
            (8, _TYPE_BYTES, namespace.encode("ascii", "strict")),
            (9, _TYPE_BYTES, shard_name.encode("ascii", "strict")),
            (10, _TYPE_U64, _u64(data_begin, "target data begin")),
            (11, _TYPE_U64, _u64(data_end, "target data end")),
            (12, _TYPE_U64, _u64(file_begin, "target file begin")),
            (13, _TYPE_U64, _u64(file_end, "target file end")),
            (14, _TYPE_U64, _u64(tensor_bytes, "target tensor bytes")),
            (
                15,
                _TYPE_HASH256,
                bytes.fromhex(
                    _digest(header_prefix_sha256, "header prefix SHA-256")
                ),
            ),
        ),
    )


def _shard_layout_root(value: dict[str, object]) -> str:
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-target-shard-layout:v1",
        (
            (1, _TYPE_BYTES, value["namespace"].encode("ascii", "strict")),
            (2, _TYPE_BYTES, value["shard_name"].encode("ascii", "strict")),
            (3, _TYPE_U32, _u32(value["tensor_count"], "shard tensor count")),
            (4, _TYPE_U64, _u64(value["payload_bytes"], "shard payload bytes")),
            (5, _TYPE_U64, _u64(value["header_json_bytes"], "header JSON bytes")),
            (6, _TYPE_U32, _u32(value["header_padding_bytes"], "header padding")),
            (7, _TYPE_U64, _u64(value["header_bytes"], "header bytes")),
            (8, _TYPE_U64, _u64(value["payload_file_begin"], "payload file begin")),
            (9, _TYPE_U64, _u64(value["file_bytes"], "shard file bytes")),
            (
                10,
                _TYPE_HASH256,
                bytes.fromhex(_digest(value["header_prefix_sha256"], "header root")),
            ),
            (
                11,
                _TYPE_HASH256,
                bytes.fromhex(_digest(value["record_layout_set_root"], "record set root")),
            ),
            (
                12,
                _TYPE_HASH256,
                bytes.fromhex(_digest(value["weight_map_set_root"], "weight map set root")),
            ),
        ),
    )


def _index_root(
    index_bytes: int,
    index_sha256: str,
    weight_map_root: str,
    tensor_count: int,
    tensor_bytes: int,
    shard_count: int,
) -> str:
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-target-index:v1",
        (
            (1, _TYPE_U64, _u64(index_bytes, "target index bytes")),
            (2, _TYPE_HASH256, bytes.fromhex(_digest(index_sha256, "index SHA-256"))),
            (3, _TYPE_HASH256, bytes.fromhex(_digest(weight_map_root, "weight map root"))),
            (4, _TYPE_U32, _u32(tensor_count, "index tensor count")),
            (5, _TYPE_U64, _u64(tensor_bytes, "index tensor bytes")),
            (6, _TYPE_U32, _u32(shard_count, "index shard count")),
        ),
    )


def _logical_layout_root(
    *,
    source_inventory_root: str,
    source_payload_closure_root: str,
    dspark_enabled: bool,
    target_tensor_count: int,
    target_tensor_bytes: int,
    total_header_bytes: int,
    total_shard_file_bytes: int,
    index_bytes: int,
    total_artifact_bytes: int,
    shard_roots: Sequence[str],
    index_root: str,
) -> str:
    fields: list[tuple[int, int, bytes]] = [
        (1, _TYPE_BYTES, _ABI.encode("ascii")),
        (2, _TYPE_BYTES, _MODEL_FAMILY.encode("ascii")),
        (3, _TYPE_HASH256, bytes.fromhex(_digest(source_inventory_root, "source root"))),
        (4, _TYPE_HASH256, bytes.fromhex(_digest(source_payload_closure_root, "payload root"))),
        (5, _TYPE_U32, _u32(int(dspark_enabled), "DSpark enabled")),
        (6, _TYPE_U32, _u32(target_tensor_count, "target tensor count")),
        (7, _TYPE_U64, _u64(target_tensor_bytes, "target tensor bytes")),
        (8, _TYPE_U32, _u32(len(shard_roots), "target shard count")),
        (9, _TYPE_U64, _u64(total_header_bytes, "total header bytes")),
        (10, _TYPE_U64, _u64(total_shard_file_bytes, "total shard file bytes")),
        (11, _TYPE_U64, _u64(index_bytes, "target index bytes")),
        (12, _TYPE_U64, _u64(total_artifact_bytes, "target artifact bytes")),
        (13, _TYPE_HASH256, bytes.fromhex(_digest(index_root, "target index root"))),
    ]
    fields.extend(
        (
            100 + ordinal,
            _TYPE_HASH256,
            bytes.fromhex(_digest(root, "target shard layout root")),
        )
        for ordinal, root in enumerate(shard_roots)
    )
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-target-logical-layout:v1", fields
    )


def _owner_projection_root(owner_roots: Sequence[str]) -> str:
    return _root_set(
        "pih:deepseek-v4-flash-0731-target-layout-owner-projection:v1",
        owner_roots,
        "target disposition owner root",
    )


def _layout_root(
    *,
    disposition_root: str,
    logical_layout_root: str,
    dspark_enabled: bool,
    world_size: int,
    owner_projection_root: str,
    target_tensor_count: int,
    target_tensor_bytes: int,
    shard_count: int,
) -> str:
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-target-layout:v1",
        (
            (1, _TYPE_BYTES, _ABI.encode("ascii")),
            (2, _TYPE_BYTES, _MODEL_FAMILY.encode("ascii")),
            (3, _TYPE_HASH256, bytes.fromhex(_digest(disposition_root, "disposition root"))),
            (4, _TYPE_HASH256, bytes.fromhex(_digest(logical_layout_root, "logical layout root"))),
            (5, _TYPE_U32, _u32(int(dspark_enabled), "DSpark enabled")),
            (6, _TYPE_U32, _u32(world_size, "target world size")),
            (
                7,
                _TYPE_HASH256,
                bytes.fromhex(
                    _digest(owner_projection_root, "owner projection root")
                ),
            ),
            (8, _TYPE_U32, _u32(target_tensor_count, "target tensor count")),
            (9, _TYPE_U64, _u64(target_tensor_bytes, "target tensor bytes")),
            (10, _TYPE_U32, _u32(shard_count, "target shard count")),
            (11, _TYPE_BYTES, _VERIFICATION_SCOPE.encode("ascii")),
            (12, _TYPE_BYTES, _SUPPORT_STATE.encode("ascii")),
        ),
    )


def _validate_family(dspark_enabled: object, world_size: object) -> None:
    if type(dspark_enabled) is not bool:
        _fail("target family DSpark flag must be a concrete bool")
    if type(world_size) is not int or not 1 <= world_size <= 4:
        _fail("target family world size must be an integer in 1..4")


def _validate_disposition_projection(
    value: object, dspark_enabled: bool, world_size: int
) -> dict[str, object]:
    expected = (
        (_ENABLED_TARGET_COUNT, _ENABLED_TARGET_BYTES)
        if dspark_enabled
        else (_DISABLED_TARGET_COUNT, _DISABLED_TARGET_BYTES)
    )
    if (
        type(value) is not dict
        or value.get("abi") != _DISPOSITION_ABI
        or value.get("model_family") != _MODEL_FAMILY
        or value.get("verification_scope") != _DISPOSITION_SCOPE
        or value.get("support_state") != _SUPPORT_STATE
        or value.get("dspark_enabled") is not dspark_enabled
        or value.get("world_size") != world_size
        or value.get("source_tensor_count") != _SOURCE_TENSOR_COUNT
        or value.get("source_tensor_bytes") != _SOURCE_TENSOR_BYTES
        or (value.get("target_tensor_count"), value.get("target_tensor_bytes"))
        != expected
        or value.get("namespace_count") != 47
        or value.get("owner_count") != world_size
    ):
        _fail("target disposition projection has invalid ABI or geometry")
    for key in (
        "disposition_root",
        "source_inventory_root",
        "source_payload_closure_root",
    ):
        _digest(value.get(key), f"target disposition {key}")
    return value


def _compile_target_layout_projection(
    disposition_projection: object,
    disposition_details: object,
    expected_layout_root: object,
    dspark_enabled: object,
    world_size: object,
    _details_sink: dict[str, object] | None = None,
) -> dict[str, object]:
    _validate_family(dspark_enabled, world_size)
    disposition = _validate_disposition_projection(
        disposition_projection, dspark_enabled, world_size
    )
    expected_root = _digest(expected_layout_root, "expected target layout root")
    if _details_sink is not None and (
        type(_details_sink) is not dict or _details_sink
    ):
        _fail("private target layout detail sink must be an empty concrete dict")
    expected_detail_keys = {
        "artifact_model_digest",
        "semantic_root",
        "source_inventory_root",
        "source_payload_closure_root",
        "disposition_root",
        "dspark_enabled",
        "world_size",
        "records",
    }
    if type(disposition_details) is not dict or set(disposition_details) != expected_detail_keys:
        _fail("target disposition private details are malformed")
    if (
        disposition_details["source_inventory_root"]
        != disposition["source_inventory_root"]
        or disposition_details["source_payload_closure_root"]
        != disposition["source_payload_closure_root"]
        or disposition_details["disposition_root"] != disposition["disposition_root"]
        or disposition_details["dspark_enabled"] is not dspark_enabled
        or disposition_details["world_size"] != world_size
    ):
        _fail("target disposition private details differ from public closure")
    raw_records = disposition_details["records"]
    if (
        type(raw_records) is not tuple
        or len(raw_records) != _SOURCE_TENSOR_COUNT
        or tuple(record[0] for record in raw_records if type(record) is tuple and len(record) == 24)
        != tuple(
            sorted(
                record[0]
                for record in raw_records
                if type(record) is tuple and len(record) == 24
            )
        )
    ):
        _fail("target disposition private record set is noncanonical")
    namespace_names = tuple(
        namespace
        for namespace, _names in _disposition._payload._source._expected_name_groups()
    )
    selected_namespaces = tuple(
        namespace
        for namespace in namespace_names
        if dspark_enabled or not namespace.startswith("mtp.")
    )
    by_namespace: dict[str, list[dict[str, object]]] = {
        namespace: [] for namespace in selected_namespaces
    }
    selected_count = 0
    selected_bytes = 0
    names_seen: set[str] = set()
    for raw in raw_records:
        if type(raw) is not tuple or len(raw) != 24:
            _fail("target disposition private record is malformed")
        (
            name, dtype, shape, namespace, role_suffix, source_shard_name,
            source_data_begin, source_data_end, source_file_begin, source_file_end,
            semantic_tensor_root, source_record_root, artifact_object_root,
            source_payload_record_root, source_payload_sha256, tensor_bytes,
            action, transform, exclusion_reason, target_name, target_shard_key,
            owner_rank, target_logical_root, disposition_record_root,
        ) = raw
        if (
            type(name) is not str or not name or name in names_seen
            or type(namespace) is not str or namespace not in namespace_names
            or type(dtype) is not str or type(shape) is not tuple
            or type(tensor_bytes) is not int or tensor_bytes <= 0
            or tensor_bytes != source_file_end - source_file_begin
            or tensor_bytes != _tensor_payload_bytes(dtype, shape)
        ):
            _fail("target disposition logical record geometry is invalid")
        names_seen.add(name)
        for value, label in (
            (source_payload_record_root, "source payload record root"),
            (source_payload_sha256, "source payload SHA-256"),
            (disposition_record_root, "disposition record root"),
        ):
            _digest(value, label)
        if owner_rank is None:
            if (
                dspark_enabled
                or not namespace.startswith("mtp.")
                or action != "explicit_excluded"
                or transform != ""
                or exclusion_reason != "dspark_disabled"
                or target_name != ""
                or target_shard_key != ""
                or target_logical_root is not None
            ):
                _fail("target excluded record has invalid feature fields")
            continue
        if (
            type(owner_rank) is not int
            or not 0 <= owner_rank < world_size
            or action != "identity_copy"
            or transform != "identity_copy_v1"
            or exclusion_reason != ""
            or target_name != name
            or target_shard_key != namespace
            or namespace not in by_namespace
            or type(target_logical_root) is not str
        ):
            _fail("target selected record has invalid logical fields")
        replayed_logical = _disposition._target_logical_root(
            source_record_root=source_record_root,
            source_payload_record_root=source_payload_record_root,
            payload_sha256=source_payload_sha256,
            name=name,
            dtype=dtype,
            shape=shape,
            target_shard_key=namespace,
            tensor_bytes=tensor_bytes,
        )
        replayed_disposition = _disposition._selected_disposition_root(
            source_record_root=source_record_root,
            source_payload_record_root=source_payload_record_root,
            target_logical_root=replayed_logical,
            name=name,
            source_namespace=namespace,
            tensor_bytes=tensor_bytes,
            owner_rank=owner_rank,
        )
        if (
            target_logical_root != replayed_logical
            or disposition_record_root != replayed_disposition
        ):
            _fail("target selected record roots differ from independent replay")
        record = {
            "name": name, "dtype": dtype, "shape": shape, "namespace": namespace,
            "source_shard_name": source_shard_name,
            "source_file_begin": source_file_begin, "source_file_end": source_file_end,
            "artifact_object_root": artifact_object_root,
            "source_payload_record_root": source_payload_record_root,
            "source_payload_sha256": source_payload_sha256,
            "target_logical_root": target_logical_root,
            "disposition_record_root": disposition_record_root,
            "owner_rank": owner_rank, "tensor_bytes": tensor_bytes,
        }
        by_namespace[namespace].append(record)
        selected_count += 1
        selected_bytes += tensor_bytes
    if (
        len(names_seen) != _SOURCE_TENSOR_COUNT
        or selected_count != disposition["target_tensor_count"]
        or selected_bytes != disposition["target_tensor_bytes"]
    ):
        _fail("target layout selection differs from disposition totals")

    public_shards: list[dict[str, object]] = []
    private_records: list[tuple[object, ...]] = []
    private_headers: list[tuple[object, ...]] = []
    weight_map: dict[str, str] = {}
    shard_weight_roots: list[str] = []
    for namespace in selected_namespaces:
        records = sorted(by_namespace[namespace], key=lambda record: record["name"])
        if not records or len(records) > _MAX_TENSORS_PER_SHARD:
            _fail("target shard is empty or exceeds its tensor bound")
        header: dict[str, object] = {"__metadata__": {"format": "pt"}}
        cursor = 0
        for record in records:
            end = _checked_add(cursor, record["tensor_bytes"], "target data offset")
            header[record["name"]] = {
                "dtype": record["dtype"],
                "shape": list(record["shape"]),
                "data_offsets": [cursor, end],
            }
            record["target_data_begin"] = cursor
            record["target_data_end"] = end
            cursor = end
        header_json = _canonical_json(header, "target Safetensors header", _MAX_SHARD_HEADER_BYTES)
        padding = (-len(header_json)) % 8
        padded = header_json + b" " * padding
        if len(padded) > _MAX_SHARD_HEADER_BYTES:
            _fail("padded target header exceeds its byte bound")
        header_prefix = len(padded).to_bytes(8, "little") + padded
        header_root = hashlib.sha256(header_prefix).hexdigest()
        shard_name = _target_shard_name(namespace)
        layout_roots: list[str] = []
        weight_roots: list[str] = []
        for record in records:
            file_begin = _checked_add(
                len(header_prefix),
                record["target_data_begin"],
                "target file begin",
            )
            file_end = _checked_add(
                len(header_prefix),
                record["target_data_end"],
                "target file end",
            )
            root = _layout_record_root(
                source_payload_record_root=record["source_payload_record_root"],
                target_logical_root=record["target_logical_root"],
                source_payload_sha256=record["source_payload_sha256"],
                name=record["name"], dtype=record["dtype"], shape=record["shape"],
                namespace=namespace, shard_name=shard_name,
                data_begin=record["target_data_begin"], data_end=record["target_data_end"],
                file_begin=file_begin, file_end=file_end,
                tensor_bytes=record["tensor_bytes"], header_prefix_sha256=header_root,
            )
            layout_roots.append(root)
            weight_roots.append(
                _weight_map_entry_root(
                    record["name"],
                    shard_name,
                    record["target_logical_root"],
                )
            )
            if record["name"] in weight_map:
                _fail("target weight map contains a duplicate name")
            weight_map[record["name"]] = shard_name
            private_records.append((
                record["name"], record["dtype"], record["shape"], namespace,
                record["source_shard_name"], record["source_file_begin"], record["source_file_end"],
                record["artifact_object_root"], record["source_payload_record_root"],
                record["source_payload_sha256"], record["target_logical_root"],
                record["disposition_record_root"], record["owner_rank"], shard_name,
                record["target_data_begin"], record["target_data_end"], file_begin, file_end,
                record["tensor_bytes"], header_root, root,
            ))
        record_set_root = _root_set(
            "pih:deepseek-v4-flash-0731-target-layout-record-set:v1",
            tuple(layout_roots), "target layout record root",
        )
        weight_set_root = _root_set(
            "pih:deepseek-v4-flash-0731-target-weight-map-shard:v1",
            tuple(weight_roots), "target weight map entry root",
        )
        shard_weight_roots.append(weight_set_root)
        value: dict[str, object] = {
            "namespace": namespace, "shard_name": shard_name,
            "tensor_count": len(records), "payload_bytes": cursor,
            "header_json_bytes": len(header_json), "header_padding_bytes": padding,
            "header_bytes": len(padded), "payload_file_begin": len(header_prefix),
            "file_bytes": _checked_add(len(header_prefix), cursor, "target shard file bytes"),
            "header_prefix_sha256": header_root,
            "record_layout_set_root": record_set_root,
            "weight_map_set_root": weight_set_root,
        }
        value["shard_layout_root"] = _shard_layout_root(value)
        public_shards.append(value)
        private_headers.append(
            (
                namespace,
                shard_name,
                header_json,
                padding,
                header_prefix,
                header_root,
            )
        )

    canonical_records = tuple(sorted(private_records, key=lambda record: record[0]))
    weight_map_root = _root_set(
        "pih:deepseek-v4-flash-0731-target-weight-map:v1",
        tuple(shard_weight_roots), "target shard weight map root",
    )
    index_source = _canonical_json(
        {"metadata": {"total_size": selected_bytes}, "weight_map": weight_map},
        "target weight index", _MAX_INDEX_BYTES,
    )
    index_sha256 = hashlib.sha256(index_source).hexdigest()
    index_root = _index_root(
        len(index_source), index_sha256, weight_map_root,
        selected_count, selected_bytes, len(public_shards),
    )
    total_header_bytes = sum(shard["header_bytes"] for shard in public_shards)
    total_shard_file_bytes = sum(shard["file_bytes"] for shard in public_shards)
    total_artifact_bytes = _checked_add(total_shard_file_bytes, len(index_source), "artifact bytes")
    expected_files = (
        (
            _ENABLED_TOTAL_HEADER_BYTES,
            _ENABLED_TOTAL_SHARD_FILE_BYTES,
            _ENABLED_INDEX_BYTES,
            _ENABLED_TOTAL_ARTIFACT_BYTES,
        )
        if dspark_enabled
        else (
            _DISABLED_TOTAL_HEADER_BYTES,
            _DISABLED_TOTAL_SHARD_FILE_BYTES,
            _DISABLED_INDEX_BYTES,
            _DISABLED_TOTAL_ARTIFACT_BYTES,
        )
    )
    if (
        total_header_bytes,
        total_shard_file_bytes,
        len(index_source),
        total_artifact_bytes,
    ) != expected_files:
        _fail("target layout aggregate file geometry differs from frozen family")
    logical_root = _logical_layout_root(
        source_inventory_root=disposition["source_inventory_root"],
        source_payload_closure_root=disposition["source_payload_closure_root"],
        dspark_enabled=dspark_enabled, target_tensor_count=selected_count,
        target_tensor_bytes=selected_bytes, total_header_bytes=total_header_bytes,
        total_shard_file_bytes=total_shard_file_bytes, index_bytes=len(index_source),
        total_artifact_bytes=total_artifact_bytes,
        shard_roots=tuple(shard["shard_layout_root"] for shard in public_shards),
        index_root=index_root,
    )
    owner_roots = tuple(owner["owner_root"] for owner in disposition["owners"])
    owner_projection = _owner_projection_root(owner_roots)
    layout_root = _layout_root(
        disposition_root=disposition["disposition_root"], logical_layout_root=logical_root,
        dspark_enabled=dspark_enabled, world_size=world_size,
        owner_projection_root=owner_projection, target_tensor_count=selected_count,
        target_tensor_bytes=selected_bytes, shard_count=len(public_shards),
    )
    if layout_root != expected_root:
        _fail("expected target layout root differs from independent replay")
    result = {
        "schema": _SCHEMA, "abi": _ABI, "model_family": _MODEL_FAMILY,
        "layout_root": layout_root, "logical_layout_root": logical_root,
        "disposition_root": disposition["disposition_root"],
        "source_inventory_root": disposition["source_inventory_root"],
        "source_payload_closure_root": disposition["source_payload_closure_root"],
        "dspark_enabled": dspark_enabled, "world_size": world_size,
        "target_tensor_count": selected_count, "target_tensor_bytes": selected_bytes,
        "shard_count": len(public_shards), "total_header_bytes": total_header_bytes,
        "total_shard_file_bytes": total_shard_file_bytes, "index_bytes": len(index_source),
        "total_artifact_bytes": total_artifact_bytes, "index_sha256": index_sha256,
        "weight_map_root": weight_map_root, "index_root": index_root,
        "shards": public_shards, "owner_roots": list(owner_roots),
        "owner_projection_root": owner_projection,
        "verification_scope": _VERIFICATION_SCOPE, "support_state": _SUPPORT_STATE,
    }
    if _details_sink is not None:
        _details_sink.update({
            "artifact_model_digest": disposition_details["artifact_model_digest"],
            "semantic_root": disposition_details["semantic_root"],
            "source_inventory_root": disposition["source_inventory_root"],
            "source_payload_closure_root": disposition["source_payload_closure_root"],
            "disposition_root": disposition["disposition_root"],
            "layout_root": layout_root, "logical_layout_root": logical_root,
            "dspark_enabled": dspark_enabled, "world_size": world_size,
            "records": canonical_records, "headers": tuple(private_headers),
            "index_source": index_source,
        })
    return result


def _verify_deepseek_v4_flash_0731_target_layout(
    *, model_root: Path, expected_model_digest: object,
    expected_semantic_root: object, expected_source_inventory_root: object,
    expected_payload_closure_root: object, expected_disposition_root: object,
    expected_layout_root: object, dspark_enabled: object, world_size: object,
    leases: list[tuple[Path, int, tuple[object, ...]]], details: dict[str, object],
) -> dict[str, object]:
    _validate_family(dspark_enabled, world_size)
    model_digest = _digest(expected_model_digest, "expected model digest")
    semantic_root = _digest(expected_semantic_root, "expected semantic root")
    source_root = _digest(expected_source_inventory_root, "expected source inventory root")
    payload_root = _digest(expected_payload_closure_root, "expected payload closure root")
    disposition_root = _digest(expected_disposition_root, "expected disposition root")
    layout_root = _digest(expected_layout_root, "expected target layout root")
    if type(leases) is not list or leases:
        _fail("private retained lease sink must be an empty concrete list")
    if type(details) is not dict or details:
        _fail("private target layout detail sink must be an empty concrete dict")
    disposition_details: dict[str, object] = {}
    disposition_projection = _disposition._verify_deepseek_v4_flash_0731_target_disposition(
        model_root=model_root, expected_model_digest=model_digest,
        expected_semantic_root=semantic_root, expected_source_inventory_root=source_root,
        expected_payload_closure_root=payload_root,
        expected_disposition_root=disposition_root, dspark_enabled=dspark_enabled,
        world_size=world_size, leases=leases, details=disposition_details,
    )
    artifact = _disposition._payload._source._semantic._artifact
    _disposition._payload._source._semantic._require_exact_safetensors_set(model_root)
    root_before = artifact._root_identity(model_root)
    artifact._revalidate_leases(leases)
    candidate: dict[str, object] = {}
    result = _compile_target_layout_projection(
        disposition_projection, disposition_details, layout_root,
        dspark_enabled, world_size, candidate,
    )
    _disposition._payload._source._semantic._require_exact_safetensors_set(model_root)
    artifact._revalidate_leases(leases)
    if artifact._root_identity(model_root) != root_before:
        _fail("artifact root or parent identity changed during target layout replay")
    details.update(candidate)
    return result


def verify_deepseek_v4_flash_0731_target_layout(
    *, model_root: Path, expected_model_digest: object,
    expected_semantic_root: object, expected_source_inventory_root: object,
    expected_payload_closure_root: object, expected_disposition_root: object,
    expected_layout_root: object, dspark_enabled: object, world_size: object,
) -> dict[str, object]:
    leases: list[tuple[Path, int, tuple[object, ...]]] = []
    details: dict[str, object] = {}
    try:
        return _verify_deepseek_v4_flash_0731_target_layout(
            model_root=model_root, expected_model_digest=expected_model_digest,
            expected_semantic_root=expected_semantic_root,
            expected_source_inventory_root=expected_source_inventory_root,
            expected_payload_closure_root=expected_payload_closure_root,
            expected_disposition_root=expected_disposition_root,
            expected_layout_root=expected_layout_root,
            dspark_enabled=dspark_enabled, world_size=world_size,
            leases=leases, details=details,
        )
    finally:
        _disposition._payload._source._semantic._artifact._close_leases(leases)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independently verify a DeepSeek target layout.")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--expected-model-digest", required=True)
    parser.add_argument("--expected-semantic-root", required=True)
    parser.add_argument("--expected-source-inventory-root", required=True)
    parser.add_argument("--expected-payload-closure-root", required=True)
    parser.add_argument("--expected-disposition-root", required=True)
    parser.add_argument("--expected-layout-root", required=True)
    parser.add_argument("--dspark-enabled", required=True, choices=("false", "true"))
    parser.add_argument("--world-size", required=True, type=int)
    arguments = parser.parse_args(argv)
    result = verify_deepseek_v4_flash_0731_target_layout(
        model_root=Path(arguments.model_root),
        expected_model_digest=arguments.expected_model_digest,
        expected_semantic_root=arguments.expected_semantic_root,
        expected_source_inventory_root=arguments.expected_source_inventory_root,
        expected_payload_closure_root=arguments.expected_payload_closure_root,
        expected_disposition_root=arguments.expected_disposition_root,
        expected_layout_root=arguments.expected_layout_root,
        dspark_enabled=arguments.dspark_enabled == "true",
        world_size=arguments.world_size,
    )
    sys.stdout.buffer.write(_canonical(result).encode("ascii") + b"\n")
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
