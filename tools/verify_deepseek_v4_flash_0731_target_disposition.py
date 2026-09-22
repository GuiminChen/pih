#!/usr/bin/env python3
"""Independently verify DeepSeek V4 Flash 0731 target disposition."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import NoReturn, Sequence


def _load_payload_verifier() -> ModuleType:
    name = "_pih_deepseek_payload_verifier_for_target_disposition"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = Path(__file__).with_name(
        "verify_deepseek_v4_flash_0731_source_payload_closure.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load independent payload verifier: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


_payload = _load_payload_verifier()

_SCHEMA = (
    "pih.deepseek_v4_flash_0731_"
    "target_disposition_verification.v1"
)
_ABI = "deepseek_v4_flash_0731_target_disposition_v1"
_MODEL_FAMILY = "deepseek_v4_flash_0731"
_VERIFICATION_SCOPE = (
    "total_source_to_target_logical_disposition_non_authorizing"
)
_SUPPORT_STATE = "hardware_evidence_open"
_PAYLOAD_ABI = "deepseek_v4_flash_0731_source_payload_closure_v1"
_PAYLOAD_SCOPE = "exact_source_tensor_payload_bytes_non_authorizing"

_SOURCE_TENSOR_COUNT = 72_317
_SOURCE_TENSOR_BYTES = 166_878_536_440
_ENABLED_TARGET_COUNT = 72_317
_ENABLED_TARGET_BYTES = 166_878_536_440
_DISABLED_TARGET_COUNT = 67_612
_DISABLED_TARGET_BYTES = 156_015_698_140
_DISABLED_EXCLUDED_COUNT = 4_705
_DISABLED_EXCLUDED_BYTES = 10_862_838_300
_IDENTITY_ACTION = "identity_copy"
_IDENTITY_TRANSFORM = "identity_copy_v1"
_EXCLUDED_ACTION = "explicit_excluded"
_DISABLED_EXCLUSION_REASON = "dspark_disabled"
_TYPE_U32 = 1
_TYPE_U64 = 2
_TYPE_BYTES = 3
_TYPE_HASH256 = 4

_RANGES = {
    (True, 1): ((0, 42),),
    (True, 2): ((0, 22), (23, 42)),
    (True, 3): ((0, 14), (15, 30), (31, 42)),
    (True, 4): ((0, 10), (11, 22), (23, 34), (35, 42)),
    (False, 1): ((0, 42),),
    (False, 2): ((0, 21), (22, 42)),
    (False, 3): ((0, 13), (14, 28), (29, 42)),
    (False, 4): ((0, 10), (11, 21), (22, 32), (33, 42)),
}

_OWNER_GEOMETRY = {
    (True, 1): ((72_317, 166_878_536_440),),
    (True, 2): (
        (36_157, 83_371_890_664),
        (36_160, 83_506_645_776),
    ),
    (True, 3): (
        (23_577, 54_738_100_520),
        (25_160, 57_267_580_288),
        (23_580, 54_872_855_632),
    ),
    (True, 4): (
        (17_287, 40_421_205_448),
        (18_870, 42_950_685_216),
        (18_870, 42_950_685_216),
        (17_290, 40_555_960_560),
    ),
    (False, 1): ((67_612, 156_015_698_140),),
    (False, 2): (
        (34_581, 79_782_039_440),
        (33_031, 76_233_658_700),
    ),
    (False, 3): (
        (22_001, 51_148_249_296),
        (23_591, 53_698_983_976),
        (22_020, 51_168_464_868),
    ),
    (False, 4): (
        (17_287, 40_421_205_448),
        (17_294, 39_360_833_992),
        (17_301, 39_382_088_904),
        (15_730, 36_851_569_796),
    ),
}


def _fail(message: str) -> NoReturn:
    raise ValueError(
        "invalid DeepSeek V4 Flash 0731 target disposition verification: "
        f"{message}"
    )


def _u32(value: int, name: str) -> bytes:
    if type(value) is not int or not 0 <= value <= (1 << 32) - 1:
        _fail(f"{name} is outside uint32")
    return value.to_bytes(4, "little")


def _u64(value: int, name: str) -> bytes:
    if type(value) is not int or not 0 <= value <= (1 << 64) - 1:
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
    return _payload._typed_sha256(domain, fields)


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


def _target_logical_root(
    *,
    source_record_root: str,
    source_payload_record_root: str,
    payload_sha256: str,
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    target_shard_key: str,
    tensor_bytes: int,
) -> str:
    packed_shape = b"".join(
        _u64(dimension, "target shape dimension") for dimension in shape
    )
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-target-logical-record:v1",
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
                    _digest(
                        source_payload_record_root,
                        "source payload record root",
                    )
                ),
            ),
            (
                3,
                _TYPE_HASH256,
                bytes.fromhex(_digest(payload_sha256, "source payload SHA-256")),
            ),
            (4, _TYPE_BYTES, name.encode("ascii", "strict")),
            (5, _TYPE_BYTES, dtype.encode("ascii", "strict")),
            (6, _TYPE_U32, _u32(len(shape), "target tensor rank")),
            (7, _TYPE_BYTES, packed_shape),
            (8, _TYPE_BYTES, target_shard_key.encode("ascii", "strict")),
            (9, _TYPE_U64, _u64(tensor_bytes, "target tensor bytes")),
            (10, _TYPE_BYTES, _IDENTITY_TRANSFORM.encode("ascii")),
        ),
    )


def _selected_disposition_root(
    *,
    source_record_root: str,
    source_payload_record_root: str,
    target_logical_root: str,
    name: str,
    source_namespace: str,
    tensor_bytes: int,
    owner_rank: int,
) -> str:
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-selected-disposition:v1",
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
                    _digest(
                        source_payload_record_root,
                        "source payload record root",
                    )
                ),
            ),
            (
                3,
                _TYPE_HASH256,
                bytes.fromhex(_digest(target_logical_root, "target logical root")),
            ),
            (4, _TYPE_BYTES, name.encode("ascii", "strict")),
            (5, _TYPE_BYTES, source_namespace.encode("ascii", "strict")),
            (6, _TYPE_U64, _u64(tensor_bytes, "disposition tensor bytes")),
            (7, _TYPE_BYTES, _IDENTITY_ACTION.encode("ascii")),
            (8, _TYPE_BYTES, _IDENTITY_TRANSFORM.encode("ascii")),
            (9, _TYPE_U32, _u32(owner_rank, "target owner rank")),
        ),
    )


def _excluded_disposition_root(
    *,
    source_record_root: str,
    source_payload_record_root: str,
    payload_sha256: str,
    name: str,
    source_namespace: str,
    tensor_bytes: int,
) -> str:
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-excluded-disposition:v1",
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
                    _digest(
                        source_payload_record_root,
                        "source payload record root",
                    )
                ),
            ),
            (
                3,
                _TYPE_HASH256,
                bytes.fromhex(_digest(payload_sha256, "source payload SHA-256")),
            ),
            (4, _TYPE_BYTES, name.encode("ascii", "strict")),
            (5, _TYPE_BYTES, source_namespace.encode("ascii", "strict")),
            (6, _TYPE_U64, _u64(tensor_bytes, "excluded tensor bytes")),
            (7, _TYPE_BYTES, _EXCLUDED_ACTION.encode("ascii")),
            (8, _TYPE_BYTES, _DISABLED_EXCLUSION_REASON.encode("ascii")),
        ),
    )


def _namespace_root(value: dict[str, object]) -> str:
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-target-disposition-namespace:v1",
        (
            (1, _TYPE_BYTES, value["namespace"].encode("ascii", "strict")),
            (2, _TYPE_U32, _u32(value["source_tensor_count"], "namespace source count")),
            (3, _TYPE_U64, _u64(value["source_tensor_bytes"], "namespace source bytes")),
            (4, _TYPE_U32, _u32(value["target_tensor_count"], "namespace target count")),
            (5, _TYPE_U64, _u64(value["target_tensor_bytes"], "namespace target bytes")),
            (6, _TYPE_U32, _u32(value["excluded_tensor_count"], "namespace excluded count")),
            (7, _TYPE_U64, _u64(value["excluded_tensor_bytes"], "namespace excluded bytes")),
            (
                8,
                _TYPE_HASH256,
                bytes.fromhex(
                    _digest(
                        value["source_disposition_set_root"],
                        "source disposition set root",
                    )
                ),
            ),
            (
                9,
                _TYPE_HASH256,
                bytes.fromhex(
                    _digest(
                        value["target_record_set_root"],
                        "target record set root",
                    )
                ),
            ),
            (
                10,
                _TYPE_HASH256,
                bytes.fromhex(
                    _digest(
                        value["excluded_record_set_root"],
                        "excluded record set root",
                    )
                ),
            ),
        ),
    )


def _owner_namespace_root(
    rank: int,
    namespace: str,
    tensor_count: int,
    tensor_bytes: int,
    record_set_root: str,
) -> str:
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-owner-namespace:v1",
        (
            (1, _TYPE_U32, _u32(rank, "owner rank")),
            (2, _TYPE_BYTES, namespace.encode("ascii", "strict")),
            (3, _TYPE_U32, _u32(tensor_count, "owner namespace count")),
            (4, _TYPE_U64, _u64(tensor_bytes, "owner namespace bytes")),
            (5, _TYPE_HASH256, bytes.fromhex(_digest(record_set_root, "owner record set root"))),
        ),
    )


def _owner_namespace_summary_root(roots: Sequence[str]) -> str:
    return _root_set(
        "pih:deepseek-v4-flash-0731-owner-namespace-summary:v1",
        roots,
        "owner namespace root",
    )


def _owner_root(value: dict[str, object]) -> str:
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-target-owner:v1",
        (
            (1, _TYPE_U32, _u32(value["rank"], "owner rank")),
            (2, _TYPE_U32, _u32(value["first_layer"], "owner first layer")),
            (3, _TYPE_U32, _u32(value["last_layer"], "owner last layer")),
            (4, _TYPE_U32, _u32(int(value["owns_embedding"]), "owns embedding")),
            (5, _TYPE_U32, _u32(int(value["owns_final_head"]), "owns final head")),
            (6, _TYPE_U32, _u32(int(value["owns_dspark"]), "owns DSpark")),
            (7, _TYPE_U32, _u32(value["tensor_count"], "owner tensor count")),
            (8, _TYPE_U64, _u64(value["tensor_bytes"], "owner tensor bytes")),
            (
                9,
                _TYPE_HASH256,
                bytes.fromhex(
                    _digest(
                        value["owner_namespace_summary_root"],
                        "owner namespace summary root",
                    )
                ),
            ),
        ),
    )


def _disposition_root(
    *,
    source_inventory_root: str,
    source_payload_closure_root: str,
    dspark_enabled: bool,
    world_size: int,
    target_tensor_count: int,
    target_tensor_bytes: int,
    excluded_tensor_count: int,
    excluded_tensor_bytes: int,
    namespace_roots: Sequence[str],
    owner_roots: Sequence[str],
) -> str:
    fields: list[tuple[int, int, bytes]] = [
        (1, _TYPE_BYTES, _ABI.encode("ascii")),
        (2, _TYPE_BYTES, _MODEL_FAMILY.encode("ascii")),
        (3, _TYPE_HASH256, bytes.fromhex(_digest(source_inventory_root, "source inventory root"))),
        (
            4,
            _TYPE_HASH256,
            bytes.fromhex(
                _digest(
                    source_payload_closure_root,
                    "source payload closure root",
                )
            ),
        ),
        (5, _TYPE_U32, _u32(int(dspark_enabled), "DSpark enabled")),
        (6, _TYPE_U32, _u32(world_size, "target world size")),
        (7, _TYPE_U32, _u32(_SOURCE_TENSOR_COUNT, "source tensor count")),
        (8, _TYPE_U64, _u64(_SOURCE_TENSOR_BYTES, "source tensor bytes")),
        (9, _TYPE_U32, _u32(target_tensor_count, "target tensor count")),
        (10, _TYPE_U64, _u64(target_tensor_bytes, "target tensor bytes")),
        (11, _TYPE_U32, _u32(excluded_tensor_count, "excluded tensor count")),
        (12, _TYPE_U64, _u64(excluded_tensor_bytes, "excluded tensor bytes")),
        (13, _TYPE_U32, _u32(len(namespace_roots), "namespace count")),
        (14, _TYPE_U32, _u32(len(owner_roots), "owner count")),
        (15, _TYPE_BYTES, _VERIFICATION_SCOPE.encode("ascii")),
        (16, _TYPE_BYTES, _SUPPORT_STATE.encode("ascii")),
    ]
    fields.extend(
        (100 + ordinal, _TYPE_HASH256, bytes.fromhex(_digest(root, "namespace disposition root")))
        for ordinal, root in enumerate(namespace_roots)
    )
    fields.extend(
        (200 + ordinal, _TYPE_HASH256, bytes.fromhex(_digest(root, "target owner root")))
        for ordinal, root in enumerate(owner_roots)
    )
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-target-disposition:v1",
        fields,
    )


def _validate_family(dspark_enabled: object, world_size: object) -> None:
    if type(dspark_enabled) is not bool:
        _fail("target family DSpark flag must be a concrete bool")
    if type(world_size) is not int or not 1 <= world_size <= 4:
        _fail("target family world size must be an integer in 1..4")


def _owner_for_record(
    name: str,
    namespace: str,
    dspark_enabled: bool,
    world_size: int,
) -> int | None:
    if namespace == "endpoint":
        return 0 if name == "embed.weight" else world_size - 1
    if namespace.startswith("layers."):
        layer_text = namespace[len("layers."):]
        if not layer_text.isascii() or not layer_text.isdecimal():
            _fail("main target namespace is not canonical")
        layer = int(layer_text)
        for rank, (first_layer, last_layer) in enumerate(
            _RANGES[(dspark_enabled, world_size)]
        ):
            if first_layer <= layer <= last_layer:
                return rank
        _fail("main target layer has no owner")
    if namespace.startswith("mtp."):
        return world_size - 1 if dspark_enabled else None
    _fail("source namespace is outside the frozen target family")


def _validate_payload_projection(value: object) -> dict[str, object]:
    if (
        type(value) is not dict
        or value.get("abi") != _PAYLOAD_ABI
        or value.get("model_family") != _MODEL_FAMILY
        or value.get("verification_scope") != _PAYLOAD_SCOPE
        or value.get("support_state") != _SUPPORT_STATE
        or value.get("tensor_count") != _SOURCE_TENSOR_COUNT
        or value.get("tensor_bytes") != _SOURCE_TENSOR_BYTES
        or value.get("shard_count") != 48
    ):
        _fail("payload projection has invalid ABI or geometry")
    for key in (
        "artifact_model_digest",
        "semantic_root",
        "source_inventory_root",
        "payload_closure_root",
    ):
        _digest(value.get(key), f"payload projection {key}")
    return value


def _compile_target_disposition_projection(
    payload_projection: object,
    payload_details: object,
    expected_disposition_root: object,
    dspark_enabled: object,
    world_size: object,
    _details_sink: dict[str, object] | None = None,
) -> dict[str, object]:
    _validate_family(dspark_enabled, world_size)
    payload_value = _validate_payload_projection(payload_projection)
    expected_root = _digest(
        expected_disposition_root, "expected target disposition root"
    )
    if _details_sink is not None and (
        type(_details_sink) is not dict or _details_sink
    ):
        _fail("private disposition detail sink must be an empty concrete dict")
    if type(payload_details) is not dict or set(payload_details) != {
        "artifact_model_digest",
        "payload_closure_root",
        "records",
        "semantic_root",
        "source_inventory_root",
    }:
        _fail("payload verifier returned an invalid private detail set")
    for key in (
        "artifact_model_digest",
        "payload_closure_root",
        "semantic_root",
        "source_inventory_root",
    ):
        if payload_details[key] != payload_value[key]:
            _fail("payload private details differ from the public antecedent")
    raw_records = payload_details["records"]
    if (
        type(raw_records) is not tuple
        or len(raw_records) != _SOURCE_TENSOR_COUNT
        or tuple(
            record[0]
            for record in raw_records
            if type(record) is tuple and len(record) == 15
        )
        != tuple(
            sorted(
                record[0]
                for record in raw_records
                if type(record) is tuple and len(record) == 15
            )
        )
    ):
        _fail("payload private record set is invalid or noncanonical")
    namespace_names = tuple(
        namespace
        for namespace, _names in _payload._source._expected_name_groups()
    )
    by_namespace: dict[str, list[dict[str, object]]] = {
        namespace: [] for namespace in namespace_names
    }
    by_owner: list[dict[str, list[dict[str, object]]]] = [
        {namespace: [] for namespace in namespace_names}
        for _rank in range(world_size)
    ]
    names_seen: set[str] = set()
    private_records: list[tuple[object, ...]] = []
    for raw in raw_records:
        if type(raw) is not tuple or len(raw) != 15:
            _fail("payload private record is malformed")
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
            artifact_object_root,
            payload_sha256,
            payload_record_root,
        ) = raw
        if (
            type(name) is not str
            or not name
            or name in names_seen
            or type(shard_name) is not str
            or shard_name not in _payload._EXPECTED_SHARD_SET
            or type(dtype) is not str
            or dtype not in _payload._source._DTYPE_ELEMENT_BYTES
            or type(shape) is not tuple
            or not 1 <= len(shape) <= 8
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
            or namespace not in by_namespace
            or type(role_suffix) is not str
            or not role_suffix
            or type(data_begin) is not int
            or type(data_end) is not int
            or type(file_begin) is not int
            or type(file_end) is not int
            or not 0 <= data_begin < data_end
            or not 0 <= file_begin < file_end
            or data_end - data_begin != file_end - file_begin
        ):
            _fail("payload private logical record is invalid")
        source_draft = _payload._source._SourceRecord(
            name=name,
            shard_name=shard_name,
            dtype=dtype,
            shape=shape,
            data_begin=data_begin,
            data_end=data_end,
            semantic_tensor_root=_digest(semantic_tensor_root, "semantic tensor root"),
            namespace=namespace,
            role_suffix=role_suffix,
            file_begin=file_begin,
            file_end=file_end,
            record_root="0" * 64,
        )
        if source_record_root != _payload._source._source_record_root(source_draft):
            _fail("payload private source record root differs")
        replayed_payload_root = _payload._payload_record_root(
            source_record_root=_digest(source_record_root, "source record root"),
            artifact_object_root=_digest(artifact_object_root, "artifact object root"),
            name=name,
            shard_name=shard_name,
            data_begin=data_begin,
            data_end=data_end,
            file_begin=file_begin,
            file_end=file_end,
            payload_sha256=_digest(payload_sha256, "payload SHA-256"),
        )
        if payload_record_root != replayed_payload_root:
            _fail("payload private record root differs from independent replay")
        names_seen.add(name)
        tensor_bytes = file_end - file_begin
        owner_rank = _owner_for_record(
            name, namespace, dspark_enabled, world_size
        )
        if owner_rank is None:
            action = _EXCLUDED_ACTION
            transform = ""
            exclusion_reason = _DISABLED_EXCLUSION_REASON
            target_name = ""
            target_shard_key = ""
            target_logical_root = None
            disposition_record_root = _excluded_disposition_root(
                source_record_root=source_record_root,
                source_payload_record_root=payload_record_root,
                payload_sha256=payload_sha256,
                name=name,
                source_namespace=namespace,
                tensor_bytes=tensor_bytes,
            )
        else:
            action = _IDENTITY_ACTION
            transform = _IDENTITY_TRANSFORM
            exclusion_reason = ""
            target_name = name
            target_shard_key = namespace
            target_logical_root = _target_logical_root(
                source_record_root=source_record_root,
                source_payload_record_root=payload_record_root,
                payload_sha256=payload_sha256,
                name=name,
                dtype=dtype,
                shape=shape,
                target_shard_key=namespace,
                tensor_bytes=tensor_bytes,
            )
            disposition_record_root = _selected_disposition_root(
                source_record_root=source_record_root,
                source_payload_record_root=payload_record_root,
                target_logical_root=target_logical_root,
                name=name,
                source_namespace=namespace,
                tensor_bytes=tensor_bytes,
                owner_rank=owner_rank,
            )
        record = {
            "name": name,
            "namespace": namespace,
            "tensor_bytes": tensor_bytes,
            "owner_rank": owner_rank,
            "target_logical_root": target_logical_root,
            "disposition_record_root": disposition_record_root,
        }
        by_namespace[namespace].append(record)
        if owner_rank is not None:
            by_owner[owner_rank][namespace].append(record)
        private_records.append(
            (
                name,
                dtype,
                shape,
                namespace,
                role_suffix,
                shard_name,
                data_begin,
                data_end,
                file_begin,
                file_end,
                semantic_tensor_root,
                source_record_root,
                artifact_object_root,
                payload_record_root,
                payload_sha256,
                tensor_bytes,
                action,
                transform,
                exclusion_reason,
                target_name,
                target_shard_key,
                owner_rank,
                target_logical_root,
                disposition_record_root,
            )
        )
    if len(names_seen) != _SOURCE_TENSOR_COUNT:
        _fail("payload private records are not a total source set")

    expected_geometry = {
        namespace: (count, byte_count)
        for namespace, count, byte_count in _payload._source._namespace_geometry()
    }
    namespace_receipts: list[dict[str, object]] = []
    for namespace in namespace_names:
        records = by_namespace[namespace]
        selected = [record for record in records if record["owner_rank"] is not None]
        excluded = [record for record in records if record["owner_rank"] is None]
        source_count = len(records)
        source_bytes = sum(record["tensor_bytes"] for record in records)
        if (source_count, source_bytes) != expected_geometry[namespace]:
            _fail("source namespace differs from frozen geometry")
        value: dict[str, object] = {
            "namespace": namespace,
            "source_tensor_count": source_count,
            "source_tensor_bytes": source_bytes,
            "target_tensor_count": len(selected),
            "target_tensor_bytes": sum(record["tensor_bytes"] for record in selected),
            "excluded_tensor_count": len(excluded),
            "excluded_tensor_bytes": sum(record["tensor_bytes"] for record in excluded),
            "source_disposition_set_root": _root_set(
                "pih:deepseek-v4-flash-0731-source-disposition-set:v1",
                tuple(record["disposition_record_root"] for record in records),
                "source disposition record root",
            ),
            "target_record_set_root": _root_set(
                "pih:deepseek-v4-flash-0731-target-record-set:v1",
                tuple(record["target_logical_root"] for record in selected),
                "target logical record root",
            ),
            "excluded_record_set_root": _root_set(
                "pih:deepseek-v4-flash-0731-excluded-record-set:v1",
                tuple(record["disposition_record_root"] for record in excluded),
                "excluded disposition record root",
            ),
        }
        value["namespace_disposition_root"] = _namespace_root(value)
        namespace_receipts.append(value)

    owner_receipts: list[dict[str, object]] = []
    for rank, (first_layer, last_layer) in enumerate(
        _RANGES[(dspark_enabled, world_size)]
    ):
        owner_records: list[dict[str, object]] = []
        namespace_roots: list[str] = []
        for namespace in namespace_names:
            records = by_owner[rank][namespace]
            owner_records.extend(records)
            set_root = _root_set(
                "pih:deepseek-v4-flash-0731-owner-record-set:v1",
                tuple(record["disposition_record_root"] for record in records),
                "owner disposition record root",
            )
            namespace_roots.append(
                _owner_namespace_root(
                    rank,
                    namespace,
                    len(records),
                    sum(record["tensor_bytes"] for record in records),
                    set_root,
                )
            )
        count = len(owner_records)
        byte_count = sum(record["tensor_bytes"] for record in owner_records)
        if (count, byte_count) != _OWNER_GEOMETRY[
            (dspark_enabled, world_size)
        ][rank]:
            _fail("target owner differs from frozen geometry")
        owner: dict[str, object] = {
            "rank": rank,
            "first_layer": first_layer,
            "last_layer": last_layer,
            "owns_embedding": rank == 0,
            "owns_final_head": rank + 1 == world_size,
            "owns_dspark": dspark_enabled and rank + 1 == world_size,
            "tensor_count": count,
            "tensor_bytes": byte_count,
            "owner_namespace_summary_root": _owner_namespace_summary_root(
                namespace_roots
            ),
        }
        owner["owner_root"] = _owner_root(owner)
        owner_receipts.append(owner)

    target_count = sum(owner["tensor_count"] for owner in owner_receipts)
    target_bytes = sum(owner["tensor_bytes"] for owner in owner_receipts)
    excluded_count = _SOURCE_TENSOR_COUNT - target_count
    excluded_bytes = _SOURCE_TENSOR_BYTES - target_bytes
    expected_target = (
        (_ENABLED_TARGET_COUNT, _ENABLED_TARGET_BYTES, 0, 0)
        if dspark_enabled
        else (
            _DISABLED_TARGET_COUNT,
            _DISABLED_TARGET_BYTES,
            _DISABLED_EXCLUDED_COUNT,
            _DISABLED_EXCLUDED_BYTES,
        )
    )
    if (target_count, target_bytes, excluded_count, excluded_bytes) != expected_target:
        _fail("target and excluded totals differ from frozen family")
    disposition_root = _disposition_root(
        source_inventory_root=payload_value["source_inventory_root"],
        source_payload_closure_root=payload_value["payload_closure_root"],
        dspark_enabled=dspark_enabled,
        world_size=world_size,
        target_tensor_count=target_count,
        target_tensor_bytes=target_bytes,
        excluded_tensor_count=excluded_count,
        excluded_tensor_bytes=excluded_bytes,
        namespace_roots=tuple(
            row["namespace_disposition_root"] for row in namespace_receipts
        ),
        owner_roots=tuple(owner["owner_root"] for owner in owner_receipts),
    )
    if disposition_root != expected_root:
        _fail(
            "expected target disposition root differs from independent replay"
        )
    result = {
        "schema": _SCHEMA,
        "abi": _ABI,
        "model_family": _MODEL_FAMILY,
        "disposition_root": disposition_root,
        "source_inventory_root": payload_value["source_inventory_root"],
        "source_payload_closure_root": payload_value["payload_closure_root"],
        "dspark_enabled": dspark_enabled,
        "world_size": world_size,
        "source_tensor_count": _SOURCE_TENSOR_COUNT,
        "source_tensor_bytes": _SOURCE_TENSOR_BYTES,
        "target_tensor_count": target_count,
        "target_tensor_bytes": target_bytes,
        "excluded_tensor_count": excluded_count,
        "excluded_tensor_bytes": excluded_bytes,
        "namespace_count": len(namespace_receipts),
        "namespaces": namespace_receipts,
        "owner_count": len(owner_receipts),
        "owners": owner_receipts,
        "verification_scope": _VERIFICATION_SCOPE,
        "support_state": _SUPPORT_STATE,
    }
    if _details_sink is not None:
        _details_sink.update(
            {
                "artifact_model_digest": payload_value[
                    "artifact_model_digest"
                ],
                "semantic_root": payload_value["semantic_root"],
                "source_inventory_root": payload_value[
                    "source_inventory_root"
                ],
                "source_payload_closure_root": payload_value[
                    "payload_closure_root"
                ],
                "disposition_root": disposition_root,
                "dspark_enabled": dspark_enabled,
                "world_size": world_size,
                "records": tuple(private_records),
            }
        )
    return result


def _verify_deepseek_v4_flash_0731_target_disposition(
    *,
    model_root: Path,
    expected_model_digest: object,
    expected_semantic_root: object,
    expected_source_inventory_root: object,
    expected_payload_closure_root: object,
    expected_disposition_root: object,
    dspark_enabled: object,
    world_size: object,
    leases: list[tuple[Path, int, tuple[object, ...]]],
    details: dict[str, object],
) -> dict[str, object]:
    _validate_family(dspark_enabled, world_size)
    model_digest = _digest(expected_model_digest, "expected model digest")
    semantic_root = _digest(expected_semantic_root, "expected semantic root")
    source_inventory_root = _digest(
        expected_source_inventory_root, "expected source inventory root"
    )
    payload_closure_root = _digest(
        expected_payload_closure_root, "expected payload closure root"
    )
    disposition_root = _digest(
        expected_disposition_root, "expected target disposition root"
    )
    if type(leases) is not list or leases:
        _fail("private retained lease sink must be an empty concrete list")
    if type(details) is not dict or details:
        _fail("private disposition detail sink must be an empty concrete dict")
    payload_details: dict[str, object] = {}
    payload_projection = (
        _payload._verify_deepseek_v4_flash_0731_source_payload_closure(
            model_root=model_root,
            expected_model_digest=model_digest,
            expected_semantic_root=semantic_root,
            expected_source_inventory_root=source_inventory_root,
            expected_payload_closure_root=payload_closure_root,
            leases=leases,
            details=payload_details,
        )
    )
    _payload._source._semantic._require_exact_safetensors_set(model_root)
    disposition_root_before = (
        _payload._source._semantic._artifact._root_identity(model_root)
    )
    _payload._source._semantic._artifact._revalidate_leases(leases)
    candidate_details: dict[str, object] = {}
    result = _compile_target_disposition_projection(
        payload_projection,
        payload_details,
        disposition_root,
        dspark_enabled,
        world_size,
        candidate_details,
    )
    _payload._source._semantic._require_exact_safetensors_set(model_root)
    _payload._source._semantic._artifact._revalidate_leases(leases)
    if (
        _payload._source._semantic._artifact._root_identity(model_root)
        != disposition_root_before
    ):
        _fail("artifact root or parent identity changed during disposition replay")
    details.update(candidate_details)
    return result


def verify_deepseek_v4_flash_0731_target_disposition(
    *,
    model_root: Path,
    expected_model_digest: object,
    expected_semantic_root: object,
    expected_source_inventory_root: object,
    expected_payload_closure_root: object,
    expected_disposition_root: object,
    dspark_enabled: object,
    world_size: object,
) -> dict[str, object]:
    leases: list[tuple[Path, int, tuple[object, ...]]] = []
    details: dict[str, object] = {}
    try:
        return _verify_deepseek_v4_flash_0731_target_disposition(
            model_root=model_root,
            expected_model_digest=expected_model_digest,
            expected_semantic_root=expected_semantic_root,
            expected_source_inventory_root=expected_source_inventory_root,
            expected_payload_closure_root=expected_payload_closure_root,
            expected_disposition_root=expected_disposition_root,
            dspark_enabled=dspark_enabled,
            world_size=world_size,
            leases=leases,
            details=details,
        )
    finally:
        _payload._source._semantic._artifact._close_leases(leases)


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
            "Independently verify a DeepSeek V4 Flash 0731 target disposition."
        )
    )
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--expected-model-digest", required=True)
    parser.add_argument("--expected-semantic-root", required=True)
    parser.add_argument("--expected-source-inventory-root", required=True)
    parser.add_argument("--expected-payload-closure-root", required=True)
    parser.add_argument("--expected-disposition-root", required=True)
    parser.add_argument(
        "--dspark-enabled",
        required=True,
        choices=("false", "true"),
    )
    parser.add_argument("--world-size", required=True, type=int)
    arguments = parser.parse_args(argv)
    result = verify_deepseek_v4_flash_0731_target_disposition(
        model_root=Path(arguments.model_root),
        expected_model_digest=arguments.expected_model_digest,
        expected_semantic_root=arguments.expected_semantic_root,
        expected_source_inventory_root=arguments.expected_source_inventory_root,
        expected_payload_closure_root=arguments.expected_payload_closure_root,
        expected_disposition_root=arguments.expected_disposition_root,
        dspark_enabled=arguments.dspark_enabled == "true",
        world_size=arguments.world_size,
    )
    sys.stdout.buffer.write(_canonical(result).encode("ascii") + b"\n")
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
