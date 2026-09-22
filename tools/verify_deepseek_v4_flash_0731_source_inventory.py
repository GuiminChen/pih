#!/usr/bin/env python3
"""Independently verify the pinned DeepSeek V4 Flash 0731 source inventory."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import NoReturn, Sequence


def _load_semantic_verifier() -> ModuleType:
    name = "_pih_deepseek_semantic_verifier_for_source_inventory"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = Path(__file__).with_name(
        "verify_deepseek_safetensors_semantic_closure.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load independent semantic verifier: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


_semantic = _load_semantic_verifier()

_SCHEMA = "pih.deepseek_v4_flash_0731_source_inventory_verification.v1"
_ABI = "deepseek_v4_flash_0731_source_inventory_v1"
_MODEL_FAMILY = "deepseek_v4_flash_0731"
_VERIFICATION_SCOPE = (
    "exact_pinned_source_name_set_and_header_semantics_non_authorizing"
)
_SUPPORT_STATE = "hardware_evidence_open"
_GRAMMAR_ROOT = "cebbc15b457d8302b6bd5a39abf72ee71d897f52eb394b5994cf6daf5fb9364f"

_TENSOR_COUNT = 72_317
_TENSOR_BYTES = 166_878_536_440
_ENDPOINT_COUNT = 6
_ENDPOINT_BYTES = 2_118_393_876
_MAIN_COUNT = 67_606
_MAIN_BYTES = 153_897_304_264
_MTP_COUNT = 4_705
_MTP_BYTES = 10_862_838_300
_MTP_STAGE_GEOMETRY = (
    (1_568, 3_610_287_448),
    (1_565, 3_559_944_536),
    (1_572, 3_692_606_316),
)
_SHARD_COUNT = 48
_EXPECTED_SHARDS = tuple(
    f"model-{ordinal:05d}-of-00048.safetensors"
    for ordinal in range(1, _SHARD_COUNT + 1)
)
_EXPECTED_SHARD_SET = frozenset(_EXPECTED_SHARDS)
_DTYPE_ELEMENT_BYTES = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E8M0": 1,
    "I32": 4,
    "I64": 8,
    "BOOL": 1,
}

_MAX_U32 = (1 << 32) - 1
_MAX_U64 = (1 << 64) - 1
_TYPE_U32 = 1
_TYPE_U64 = 2
_TYPE_BYTES = 3
_TYPE_HASH256 = 4


def _fail(message: str) -> NoReturn:
    raise ValueError(
        f"invalid DeepSeek V4 Flash 0731 source inventory verification: {message}"
    )


def _u32(value: int, name: str) -> bytes:
    if type(value) is not int or not 0 <= value <= _MAX_U32:
        _fail(f"{name} is outside uint32")
    return value.to_bytes(4, "little")


def _u64(value: int, name: str) -> bytes:
    if type(value) is not int or not 0 <= value <= _MAX_U64:
        _fail(f"{name} is outside uint64")
    return value.to_bytes(8, "little")


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


def _common_source_suffixes() -> tuple[str, ...]:
    suffixes = [
        "attn_norm.weight",
        "attn.attn_sink",
        "attn.kv_norm.weight",
        "attn.q_norm.weight",
        "ffn_norm.weight",
        "ffn.gate.weight",
        "hc_attn_base",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_ffn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
    ]
    suffixes.extend(
        f"attn.{projection}.{component}"
        for projection in ("wkv", "wo_a", "wo_b", "wq_a", "wq_b")
        for component in ("weight", "scale")
    )
    suffixes.extend(
        f"ffn.shared_experts.{projection}.{component}"
        for projection in ("w1", "w2", "w3")
        for component in ("weight", "scale")
    )
    return tuple(sorted(suffixes))


def _expert_source_suffixes() -> tuple[str, ...]:
    return tuple(
        f"ffn.experts.{expert}.{projection}.{component}"
        for expert in range(256)
        for projection in ("w1", "w2", "w3")
        for component in ("weight", "scale")
    )


def _main_layer_suffixes(layer: int) -> tuple[str, ...]:
    suffixes = [*_common_source_suffixes(), *_expert_source_suffixes()]
    suffixes.append("ffn.gate.tid2eid" if layer <= 2 else "ffn.gate.bias")
    if layer >= 2:
        suffixes.extend(
            (
                "attn.compressor.ape",
                "attn.compressor.norm.weight",
                "attn.compressor.wgate.weight",
                "attn.compressor.wkv.weight",
            )
        )
    if layer >= 2 and layer % 2 == 0:
        suffixes.extend(
            (
                "attn.indexer.wq_b.weight",
                "attn.indexer.wq_b.scale",
                "attn.indexer.weights_proj.weight",
                "attn.indexer.compressor.ape",
                "attn.indexer.compressor.norm.weight",
                "attn.indexer.compressor.wgate.weight",
                "attn.indexer.compressor.wkv.weight",
            )
        )
    return tuple(sorted(suffixes))


def _mtp_suffixes(stage: int) -> tuple[str, ...]:
    suffixes = [
        *_common_source_suffixes(),
        *_expert_source_suffixes(),
        "ffn.gate.bias",
    ]
    if stage == 0:
        suffixes.extend(
            ("main_norm.weight", "main_proj.weight", "main_proj.scale")
        )
    elif stage == 2:
        suffixes.extend(
            (
                "confidence_head.proj.weight",
                "hc_head_base",
                "hc_head_fn",
                "hc_head_scale",
                "markov_head.markov_w1.weight",
                "markov_head.markov_w2.weight",
                "norm.weight",
            )
        )
    return tuple(sorted(suffixes))


@lru_cache(maxsize=1)
def _expected_name_groups() -> tuple[tuple[str, tuple[str, ...]], ...]:
    groups: list[tuple[str, tuple[str, ...]]] = [
        (
            "endpoint",
            tuple(
                sorted(
                    (
                        "embed.weight",
                        "head.weight",
                        "norm.weight",
                        "hc_head_fn",
                        "hc_head_scale",
                        "hc_head_base",
                    )
                )
            ),
        )
    ]
    groups.extend(
        (
            f"layers.{layer}",
            tuple(
                f"layers.{layer}.{suffix}"
                for suffix in _main_layer_suffixes(layer)
            ),
        )
        for layer in range(43)
    )
    groups.extend(
        (
            f"mtp.{stage}",
            tuple(f"mtp.{stage}.{suffix}" for suffix in _mtp_suffixes(stage)),
        )
        for stage in range(3)
    )
    result = tuple(groups)
    if len(result) != 47 or sum(len(names) for _key, names in result) != _TENSOR_COUNT:
        _fail("internal exact source name grammar has invalid geometry")
    if any(names != tuple(sorted(names)) for _key, names in result):
        _fail("internal exact source name grammar is not canonical")
    return result


def _source_name_root(name: str) -> str:
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-source-name:v1",
        ((1, _TYPE_BYTES, name.encode("ascii", "strict")),),
    )


def _source_name_group_root(key: str, names: Sequence[str]) -> str:
    fields: list[tuple[int, int, bytes]] = [
        (1, _TYPE_BYTES, key.encode("ascii", "strict")),
        (2, _TYPE_U32, _u32(len(names), "source name group count")),
    ]
    fields.extend(
        (100 + ordinal, _TYPE_HASH256, bytes.fromhex(_source_name_root(name)))
        for ordinal, name in enumerate(names)
    )
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-source-name-group:v1", fields
    )


def _source_name_grammar_root(
    groups: Sequence[tuple[str, Sequence[str]]],
) -> str:
    fields: list[tuple[int, int, bytes]] = [
        (1, _TYPE_BYTES, _ABI.encode("ascii")),
        (2, _TYPE_BYTES, _MODEL_FAMILY.encode("ascii")),
        (3, _TYPE_U32, _u32(len(groups), "source name group count")),
    ]
    fields.extend(
        (
            100 + ordinal,
            _TYPE_HASH256,
            bytes.fromhex(_source_name_group_root(key, names)),
        )
        for ordinal, (key, names) in enumerate(groups)
    )
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-source-name-grammar:v1", fields
    )


@lru_cache(maxsize=1)
def _source_roles() -> dict[str, tuple[str, str]]:
    groups = _expected_name_groups()
    if _source_name_grammar_root(groups) != _GRAMMAR_ROOT:
        _fail("internal exact source name grammar root drifted")
    roles: dict[str, tuple[str, str]] = {}
    for namespace, names in groups:
        prefix = "" if namespace == "endpoint" else namespace + "."
        for name in names:
            if prefix and not name.startswith(prefix):
                _fail("internal source name is spliced across namespaces")
            suffix = name[len(prefix):] if prefix else name
            if not suffix or name in roles:
                _fail("internal source role table is invalid or duplicated")
            roles[name] = (namespace, suffix)
    if len(roles) != _TENSOR_COUNT:
        _fail("internal source role table has invalid geometry")
    return roles


def _source_role(name: object) -> tuple[str, str]:
    if type(name) is not str:
        _fail("source tensor name must be a concrete string")
    role = _source_roles().get(name)
    if role is None:
        _fail("source tensor name is outside the exact pinned grammar")
    return role


@lru_cache(maxsize=1)
def _namespace_geometry() -> tuple[tuple[str, int, int], ...]:
    rows: list[tuple[str, int, int]] = [
        ("endpoint", _ENDPOINT_COUNT, _ENDPOINT_BYTES)
    ]
    for layer in range(43):
        if layer <= 1:
            count, byte_count = 1_565, 3_566_148_952
        elif layer == 2:
            count, byte_count = 1_576, 3_596_055_640
        elif layer % 2:
            count, byte_count = 1_569, 3_568_596_312
        else:
            count, byte_count = 1_576, 3_589_851_224
        rows.append((f"layers.{layer}", count, byte_count))
    rows.extend(
        (
            ("mtp.0", *_MTP_STAGE_GEOMETRY[0]),
            ("mtp.1", *_MTP_STAGE_GEOMETRY[1]),
            ("mtp.2", *_MTP_STAGE_GEOMETRY[2]),
        )
    )
    result = tuple(rows)
    if (
        len(result) != 47
        or sum(count for _key, count, _bytes in result) != _TENSOR_COUNT
        or sum(byte_count for _key, _count, byte_count in result)
        != _TENSOR_BYTES
        or sum(count for _key, count, _bytes in result[1:44])
        != _MAIN_COUNT
        or sum(byte_count for _key, _count, byte_count in result[1:44])
        != _MAIN_BYTES
        or sum(count for _key, count, _bytes in result[44:]) != _MTP_COUNT
        or sum(byte_count for _key, _count, byte_count in result[44:])
        != _MTP_BYTES
    ):
        _fail("internal source namespace geometry is inconsistent")
    return result


@dataclass(frozen=True, slots=True)
class _SourceRecord:
    name: str
    shard_name: str
    dtype: str
    shape: tuple[int, ...]
    data_begin: int
    data_end: int
    semantic_tensor_root: str
    namespace: str
    role_suffix: str
    file_begin: int
    file_end: int
    record_root: str

    @property
    def tensor_bytes(self) -> int:
        return self.data_end - self.data_begin


def _semantic_tensor_root(
    *,
    name: str,
    shard_name: str,
    dtype: str,
    shape: tuple[int, ...],
    data_begin: int,
    data_end: int,
) -> str:
    packed_shape = b"".join(
        _u64(dimension, "semantic tensor shape dimension")
        for dimension in shape
    )
    return _typed_sha256(
        "pih:deepseek-safetensors-tensor:v1",
        (
            (1, _TYPE_BYTES, name.encode("ascii", "strict")),
            (2, _TYPE_BYTES, shard_name.encode("ascii", "strict")),
            (3, _TYPE_BYTES, dtype.encode("ascii", "strict")),
            (4, _TYPE_U32, _u32(len(shape), "semantic tensor rank")),
            (5, _TYPE_BYTES, packed_shape),
            (6, _TYPE_U64, _u64(data_begin, "semantic data begin")),
            (7, _TYPE_U64, _u64(data_end, "semantic data end")),
        ),
    )


def _semantic_tensor_set_root(
    shard_name: str, tensors: Sequence[tuple[object, ...]]
) -> str:
    fields: list[tuple[int, int, bytes]] = [
        (1, _TYPE_U32, _u32(len(tensors), "semantic shard tensor count"))
    ]
    fields.extend(
        (
            100 + ordinal,
            _TYPE_HASH256,
            bytes.fromhex(_digest(tensor[5], "semantic tensor root")),
        )
        for ordinal, tensor in enumerate(tensors)
    )
    return _typed_sha256(
        "pih:deepseek-safetensors-tensor-set:v1", fields
    )


def _semantic_shard_root(
    *,
    artifact_object_root: str,
    shard_name: str,
    file_bytes: int,
    header_bytes: int,
    data_bytes: int,
    header_prefix_sha256: str,
    tensor_count: int,
    tensor_set_root: str,
) -> str:
    return _typed_sha256(
        "pih:deepseek-safetensors-shard:v1",
        (
            (1, _TYPE_HASH256, bytes.fromhex(_digest(artifact_object_root, "artifact object root"))),
            (2, _TYPE_BYTES, shard_name.encode("ascii", "strict")),
            (3, _TYPE_U64, _u64(file_bytes, "semantic shard file bytes")),
            (4, _TYPE_U64, _u64(header_bytes, "semantic shard header bytes")),
            (5, _TYPE_U64, _u64(data_bytes, "semantic shard data bytes")),
            (6, _TYPE_HASH256, bytes.fromhex(_digest(header_prefix_sha256, "header prefix SHA-256"))),
            (7, _TYPE_U32, _u32(tensor_count, "semantic shard tensor count")),
            (8, _TYPE_HASH256, bytes.fromhex(_digest(tensor_set_root, "semantic tensor-set root"))),
        ),
    )


def _source_record_root(record: _SourceRecord) -> str:
    packed_shape = b"".join(
        _u64(dimension, "source tensor shape dimension")
        for dimension in record.shape
    )
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-source-tensor:v1",
        (
            (1, _TYPE_BYTES, record.name.encode("ascii", "strict")),
            (2, _TYPE_BYTES, record.shard_name.encode("ascii", "strict")),
            (3, _TYPE_BYTES, record.dtype.encode("ascii", "strict")),
            (4, _TYPE_U32, _u32(len(record.shape), "source tensor rank")),
            (5, _TYPE_BYTES, packed_shape),
            (6, _TYPE_U64, _u64(record.data_begin, "source data begin")),
            (7, _TYPE_U64, _u64(record.data_end, "source data end")),
            (
                8,
                _TYPE_HASH256,
                bytes.fromhex(
                    _digest(record.semantic_tensor_root, "semantic tensor root")
                ),
            ),
            (9, _TYPE_BYTES, record.namespace.encode("ascii", "strict")),
            (10, _TYPE_BYTES, record.role_suffix.encode("ascii", "strict")),
            (11, _TYPE_U64, _u64(record.file_begin, "source file begin")),
            (12, _TYPE_U64, _u64(record.file_end, "source file end")),
        ),
    )


def _shard_record_set_root(records: Sequence[_SourceRecord]) -> str:
    fields: list[tuple[int, int, bytes]] = [
        (1, _TYPE_U32, _u32(len(records), "source shard record count"))
    ]
    fields.extend(
        (
            100 + ordinal,
            _TYPE_HASH256,
            bytes.fromhex(_digest(record.record_root, "source record root")),
        )
        for ordinal, record in enumerate(records)
    )
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-source-shard-record-set:v1",
        fields,
    )


def _shard_inventory_root(
    *,
    shard_name: str,
    semantic_shard_root: str,
    tensor_count: int,
    tensor_bytes: int,
    record_set_root: str,
    file_bytes: int,
    header_bytes: int,
    data_bytes: int,
) -> str:
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-source-shard-inventory:v1",
        (
            (1, _TYPE_BYTES, shard_name.encode("ascii", "strict")),
            (2, _TYPE_HASH256, bytes.fromhex(_digest(semantic_shard_root, "semantic shard root"))),
            (3, _TYPE_U32, _u32(tensor_count, "source shard tensor count")),
            (4, _TYPE_U64, _u64(tensor_bytes, "source shard tensor bytes")),
            (5, _TYPE_HASH256, bytes.fromhex(_digest(record_set_root, "source shard record-set root"))),
            (6, _TYPE_U64, _u64(file_bytes, "source shard file bytes")),
            (7, _TYPE_U64, _u64(header_bytes, "source shard header bytes")),
            (8, _TYPE_U64, _u64(data_bytes, "source shard data bytes")),
        ),
    )


def _namespace_root(
    *,
    namespace: str,
    tensor_count: int,
    tensor_bytes: int,
    name_group_root: str,
    records: Sequence[_SourceRecord],
) -> str:
    fields: list[tuple[int, int, bytes]] = [
        (1, _TYPE_BYTES, namespace.encode("ascii", "strict")),
        (2, _TYPE_U32, _u32(tensor_count, "namespace tensor count")),
        (3, _TYPE_U64, _u64(tensor_bytes, "namespace tensor bytes")),
        (4, _TYPE_HASH256, bytes.fromhex(_digest(name_group_root, "source name-group root"))),
    ]
    fields.extend(
        (
            100 + ordinal,
            _TYPE_HASH256,
            bytes.fromhex(_digest(record.record_root, "source record root")),
        )
        for ordinal, record in enumerate(records)
    )
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-source-namespace:v1", fields
    )


def _namespace_summary_root(namespace_roots: Sequence[str]) -> str:
    fields: list[tuple[int, int, bytes]] = [
        (1, _TYPE_U32, _u32(len(namespace_roots), "source namespace count"))
    ]
    fields.extend(
        (
            100 + ordinal,
            _TYPE_HASH256,
            bytes.fromhex(_digest(root, "source namespace root")),
        )
        for ordinal, root in enumerate(namespace_roots)
    )
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-source-namespace-set:v1", fields
    )


def _source_inventory_root(
    *,
    artifact_model_digest: str,
    semantic_root: str,
    grammar_root: str,
    namespace_summary_root: str,
    shard_roots: Sequence[str],
) -> str:
    fields: list[tuple[int, int, bytes]] = [
        (1, _TYPE_BYTES, _ABI.encode("ascii")),
        (2, _TYPE_BYTES, _MODEL_FAMILY.encode("ascii")),
        (3, _TYPE_HASH256, bytes.fromhex(_digest(artifact_model_digest, "artifact model digest"))),
        (4, _TYPE_HASH256, bytes.fromhex(_digest(semantic_root, "semantic root"))),
        (5, _TYPE_HASH256, bytes.fromhex(_digest(grammar_root, "source grammar root"))),
        (6, _TYPE_HASH256, bytes.fromhex(_digest(namespace_summary_root, "namespace summary root"))),
        (7, _TYPE_U32, _u32(len(shard_roots), "source shard count")),
        (8, _TYPE_U64, _u64(_TENSOR_COUNT, "source tensor count")),
        (9, _TYPE_U64, _u64(_TENSOR_BYTES, "source tensor bytes")),
        (10, _TYPE_U32, _u32(_ENDPOINT_COUNT, "endpoint tensor count")),
        (11, _TYPE_U64, _u64(_ENDPOINT_BYTES, "endpoint tensor bytes")),
        (12, _TYPE_U64, _u64(_MAIN_COUNT, "main tensor count")),
        (13, _TYPE_U64, _u64(_MAIN_BYTES, "main tensor bytes")),
        (14, _TYPE_U32, _u32(_MTP_COUNT, "MTP tensor count")),
        (15, _TYPE_U64, _u64(_MTP_BYTES, "MTP tensor bytes")),
        (16, _TYPE_BYTES, _VERIFICATION_SCOPE.encode("ascii")),
        (17, _TYPE_BYTES, _SUPPORT_STATE.encode("ascii")),
    ]
    fields.extend(
        (
            100 + ordinal,
            _TYPE_HASH256,
            bytes.fromhex(_digest(root, "source shard inventory root")),
        )
        for ordinal, root in enumerate(shard_roots)
    )
    return _typed_sha256(
        "pih:deepseek-v4-flash-0731-source-inventory:v1", fields
    )


def _validate_semantic_projection(value: object) -> dict[str, object]:
    expected_keys = {
        "schema",
        "abi",
        "model_family",
        "semantic_root",
        "artifact_model_digest",
        "artifact_index_object_root",
        "shard_count",
        "tensor_count",
        "tensor_bytes",
        "total_header_bytes",
        "dtype_bytes",
        "verification_scope",
        "support_state",
    }
    if type(value) is not dict or set(value) != expected_keys:
        _fail("semantic verifier returned an invalid public projection")
    if (
        value["schema"]
        != "pih.deepseek_safetensors_semantic_verification.v1"
        or value["abi"] != "deepseek_safetensors_semantic_closure_v1"
        or value["model_family"] != _MODEL_FAMILY
        or value["verification_scope"]
        != "exact_safetensors_header_semantics_non_authorizing"
        or value["support_state"] != _SUPPORT_STATE
        or value["shard_count"] != _SHARD_COUNT
        or value["tensor_count"] != _TENSOR_COUNT
        or value["tensor_bytes"] != _TENSOR_BYTES
    ):
        _fail("semantic verifier projection differs from the required antecedent")
    _digest(value["semantic_root"], "semantic root")
    _digest(value["artifact_model_digest"], "artifact model digest")
    _digest(value["artifact_index_object_root"], "artifact index object root")
    if (
        type(value["total_header_bytes"]) is not int
        or value["total_header_bytes"] <= 0
        or type(value["dtype_bytes"]) is not dict
        or any(
            type(dtype) is not str
            or dtype not in _DTYPE_ELEMENT_BYTES
            or type(byte_count) is not int
            or byte_count <= 0
            for dtype, byte_count in value["dtype_bytes"].items()
        )
        or sum(value["dtype_bytes"].values()) != _TENSOR_BYTES
    ):
        _fail("semantic verifier returned invalid header or dtype geometry")
    return value


def _compile_source_inventory_projection(
    semantic_projection: object,
    semantic_details: object,
    expected_source_inventory_root: object,
    _details_sink: dict[str, object] | None = None,
) -> dict[str, object]:
    projection = _validate_semantic_projection(semantic_projection)
    expected_root = _digest(
        expected_source_inventory_root, "expected source inventory root"
    )
    if _details_sink is not None and (
        type(_details_sink) is not dict or _details_sink
    ):
        _fail("private source detail sink must be an empty concrete dict")
    if type(semantic_details) is not dict or set(semantic_details) != {
        "artifact_index_object_root",
        "artifact_model_digest",
        "semantic_root",
        "shards",
        "weight_map",
    }:
        _fail("semantic verifier returned an invalid private detail set")
    if (
        semantic_details["artifact_model_digest"]
        != projection["artifact_model_digest"]
        or semantic_details["artifact_index_object_root"]
        != projection["artifact_index_object_root"]
        or semantic_details["semantic_root"] != projection["semantic_root"]
    ):
        _fail("semantic private details differ from the public antecedent")

    weight_map_value = semantic_details["weight_map"]
    if (
        type(weight_map_value) is not tuple
        or len(weight_map_value) != _TENSOR_COUNT
        or weight_map_value != tuple(sorted(weight_map_value))
        or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not str
            or item[1] not in _EXPECTED_SHARD_SET
            for item in weight_map_value
        )
    ):
        _fail("semantic private weight map is invalid")
    weight_map = dict(weight_map_value)
    if len(weight_map) != len(weight_map_value):
        _fail("semantic private weight map contains duplicate tensor names")

    shards_value = semantic_details["shards"]
    if (
        type(shards_value) is not tuple
        or len(shards_value) != _SHARD_COUNT
        or tuple(
            item[0]
            for item in shards_value
            if type(item) is tuple and len(item) == 9
        )
        != _EXPECTED_SHARDS
    ):
        _fail("semantic private shard set is invalid or noncanonical")

    all_records: list[_SourceRecord] = []
    observed_weight_map: dict[str, str] = {}
    shard_receipts: list[dict[str, object]] = []
    shard_inventory_roots: list[str] = []
    private_shards: list[tuple[object, ...]] = []
    observed_tensor_bytes = 0
    for raw_shard in shards_value:
        if type(raw_shard) is not tuple or len(raw_shard) != 9:
            _fail("semantic private shard record is malformed")
        (
            shard_name,
            artifact_object_root,
            file_bytes,
            header_bytes,
            data_bytes,
            header_prefix_sha256,
            tensor_set_root,
            semantic_shard_root,
            tensors,
        ) = raw_shard
        if (
            type(shard_name) is not str
            or shard_name not in _EXPECTED_SHARD_SET
            or type(file_bytes) is not int
            or type(header_bytes) is not int
            or type(data_bytes) is not int
            or not 0 < header_bytes <= _MAX_U64
            or not 0 < data_bytes <= _MAX_U64
            or file_bytes
            != _checked_add(
                _checked_add(8, header_bytes, "source payload begin"),
                data_bytes,
                "source shard file bytes",
            )
            or type(tensors) is not tuple
            or not tensors
            or len(tensors) > 4_096
        ):
            _fail("semantic private shard geometry is invalid")
        _digest(artifact_object_root, "artifact shard object root")
        _digest(header_prefix_sha256, "header prefix SHA-256")
        _digest(tensor_set_root, "semantic tensor-set root")
        _digest(semantic_shard_root, "semantic shard root")

        normalized_tensors: list[tuple[object, ...]] = []
        shard_records: list[_SourceRecord] = []
        data_ranges: list[tuple[int, int, str]] = []
        for raw_tensor in tensors:
            if type(raw_tensor) is not tuple or len(raw_tensor) != 6:
                _fail("semantic private tensor record is malformed")
            name, dtype, shape, data_begin, data_end, supplied_tensor_root = raw_tensor
            if (
                type(name) is not str
                or type(dtype) is not str
                or dtype not in _DTYPE_ELEMENT_BYTES
                or type(shape) is not tuple
                or not 1 <= len(shape) <= 8
                or type(data_begin) is not int
                or type(data_end) is not int
                or not 0 <= data_begin < data_end <= data_bytes
            ):
                _fail("semantic private tensor geometry is invalid")
            elements = 1
            for dimension in shape:
                if type(dimension) is not int or not 0 < dimension <= _MAX_U64:
                    _fail("semantic private tensor shape is invalid")
                elements = _checked_mul(
                    elements, dimension, "semantic tensor shape product"
                )
            tensor_bytes = _checked_mul(
                elements,
                _DTYPE_ELEMENT_BYTES[dtype],
                "semantic tensor byte count",
            )
            if data_end - data_begin != tensor_bytes:
                _fail("semantic tensor range differs from dtype and shape")
            expected_tensor_root = _semantic_tensor_root(
                name=name,
                shard_name=shard_name,
                dtype=dtype,
                shape=shape,
                data_begin=data_begin,
                data_end=data_end,
            )
            if supplied_tensor_root != expected_tensor_root:
                _fail("semantic tensor root differs from independent replay")
            if name in observed_weight_map:
                _fail("semantic tensor set contains a cross-shard duplicate")
            observed_weight_map[name] = shard_name
            if weight_map.get(name) != shard_name:
                _fail("semantic index/header bijection differs at source replay")
            namespace, role_suffix = _source_role(name)
            payload_begin = _checked_add(8, header_bytes, "source payload begin")
            file_begin = _checked_add(
                payload_begin, data_begin, "source absolute file begin"
            )
            file_end = _checked_add(
                payload_begin, data_end, "source absolute file end"
            )
            if file_end > file_bytes:
                _fail("source absolute tensor range exceeds the shard")
            draft = _SourceRecord(
                name=name,
                shard_name=shard_name,
                dtype=dtype,
                shape=shape,
                data_begin=data_begin,
                data_end=data_end,
                semantic_tensor_root=expected_tensor_root,
                namespace=namespace,
                role_suffix=role_suffix,
                file_begin=file_begin,
                file_end=file_end,
                record_root="0" * 64,
            )
            record = _SourceRecord(
                name=draft.name,
                shard_name=draft.shard_name,
                dtype=draft.dtype,
                shape=draft.shape,
                data_begin=draft.data_begin,
                data_end=draft.data_end,
                semantic_tensor_root=draft.semantic_tensor_root,
                namespace=draft.namespace,
                role_suffix=draft.role_suffix,
                file_begin=draft.file_begin,
                file_end=draft.file_end,
                record_root=_source_record_root(draft),
            )
            shard_records.append(record)
            all_records.append(record)
            normalized_tensors.append(raw_tensor)
            data_ranges.append((data_begin, data_end, name))

        if tuple(record.name for record in shard_records) != tuple(
            sorted(record.name for record in shard_records)
        ):
            _fail("semantic shard tensor names are not canonical")
        cursor = 0
        for data_begin, data_end, _name in sorted(data_ranges):
            if data_begin != cursor:
                _fail("semantic shard tensor ranges overlap or contain a gap")
            cursor = data_end
        if cursor != data_bytes:
            _fail("semantic shard tensor ranges do not cover the payload")
        independent_tensor_set_root = _semantic_tensor_set_root(
            shard_name, tuple(normalized_tensors)
        )
        if tensor_set_root != independent_tensor_set_root:
            _fail("semantic tensor-set root differs from independent replay")
        independent_semantic_shard_root = _semantic_shard_root(
            artifact_object_root=artifact_object_root,
            shard_name=shard_name,
            file_bytes=file_bytes,
            header_bytes=header_bytes,
            data_bytes=data_bytes,
            header_prefix_sha256=header_prefix_sha256,
            tensor_count=len(shard_records),
            tensor_set_root=independent_tensor_set_root,
        )
        if semantic_shard_root != independent_semantic_shard_root:
            _fail("semantic shard root differs from independent replay")
        record_set_root = _shard_record_set_root(shard_records)
        shard_tensor_bytes = sum(
            record.tensor_bytes for record in shard_records
        )
        if shard_tensor_bytes != data_bytes:
            _fail("source shard record bytes differ from the semantic payload")
        shard_inventory_root = _shard_inventory_root(
            shard_name=shard_name,
            semantic_shard_root=semantic_shard_root,
            tensor_count=len(shard_records),
            tensor_bytes=shard_tensor_bytes,
            record_set_root=record_set_root,
            file_bytes=file_bytes,
            header_bytes=header_bytes,
            data_bytes=data_bytes,
        )
        shard_inventory_roots.append(shard_inventory_root)
        shard_receipts.append(
            {
                "shard_name": shard_name,
                "semantic_shard_root": semantic_shard_root,
                "tensor_count": len(shard_records),
                "tensor_bytes": shard_tensor_bytes,
                "record_set_root": record_set_root,
                "file_bytes": file_bytes,
                "header_bytes": header_bytes,
                "data_bytes": data_bytes,
                "shard_inventory_root": shard_inventory_root,
            }
        )
        private_shards.append(
            (
                shard_name,
                artifact_object_root,
                shard_inventory_root,
                len(shard_records),
                shard_tensor_bytes,
            )
        )
        observed_tensor_bytes = _checked_add(
            observed_tensor_bytes,
            shard_tensor_bytes,
            "source tensor byte total",
        )

    if observed_weight_map != weight_map:
        _fail("source replay does not form the exact index/header bijection")
    if len(all_records) != _TENSOR_COUNT or observed_tensor_bytes != _TENSOR_BYTES:
        _fail("source record geometry differs from the exact pinned inventory")
    all_records.sort(key=lambda record: record.name)
    if tuple(record.name for record in all_records) != tuple(sorted(weight_map)):
        _fail("source record names differ from the semantic antecedent")

    groups = _expected_name_groups()
    geometry = _namespace_geometry()
    if (
        tuple(namespace for namespace, _names in groups)
        != tuple(namespace for namespace, _count, _bytes in geometry)
        or _source_name_grammar_root(groups) != _GRAMMAR_ROOT
    ):
        _fail("source grammar and namespace geometry disagree")
    records_by_namespace: dict[str, list[_SourceRecord]] = {
        namespace: [] for namespace, _names in groups
    }
    for record in all_records:
        records_by_namespace[record.namespace].append(record)

    namespace_receipts: list[dict[str, object]] = []
    namespace_roots: list[str] = []
    for (namespace, expected_names), (
        geometry_namespace,
        expected_count,
        expected_bytes,
    ) in zip(groups, geometry, strict=True):
        if namespace != geometry_namespace:
            _fail("source namespace order differs from the frozen geometry")
        records = records_by_namespace[namespace]
        observed_names = tuple(record.name for record in records)
        tensor_bytes = sum(record.tensor_bytes for record in records)
        if (
            observed_names != tuple(expected_names)
            or len(records) != expected_count
            or tensor_bytes != expected_bytes
        ):
            _fail(
                f"source namespace {namespace} differs from exact names or bytes"
            )
        name_group_root = _source_name_group_root(namespace, expected_names)
        namespace_root = _namespace_root(
            namespace=namespace,
            tensor_count=len(records),
            tensor_bytes=tensor_bytes,
            name_group_root=name_group_root,
            records=records,
        )
        namespace_roots.append(namespace_root)
        namespace_receipts.append(
            {
                "namespace": namespace,
                "tensor_count": len(records),
                "tensor_bytes": tensor_bytes,
                "name_group_root": name_group_root,
                "namespace_root": namespace_root,
            }
        )

    endpoint_count = len(records_by_namespace["endpoint"])
    endpoint_bytes = sum(
        record.tensor_bytes for record in records_by_namespace["endpoint"]
    )
    main_records = [
        record for record in all_records if record.namespace.startswith("layers.")
    ]
    mtp_records = [
        record for record in all_records if record.namespace.startswith("mtp.")
    ]
    main_bytes = sum(record.tensor_bytes for record in main_records)
    mtp_bytes = sum(record.tensor_bytes for record in mtp_records)
    if (
        endpoint_count != _ENDPOINT_COUNT
        or endpoint_bytes != _ENDPOINT_BYTES
        or len(main_records) != _MAIN_COUNT
        or main_bytes != _MAIN_BYTES
        or len(mtp_records) != _MTP_COUNT
        or mtp_bytes != _MTP_BYTES
    ):
        _fail("source endpoint/main/MTP partition geometry differs")

    namespace_summary_root = _namespace_summary_root(namespace_roots)
    source_inventory_root = _source_inventory_root(
        artifact_model_digest=projection["artifact_model_digest"],
        semantic_root=projection["semantic_root"],
        grammar_root=_GRAMMAR_ROOT,
        namespace_summary_root=namespace_summary_root,
        shard_roots=shard_inventory_roots,
    )
    if source_inventory_root != expected_root:
        _fail(
            "expected source inventory root differs from independently replayed root"
        )
    result = {
        "schema": _SCHEMA,
        "abi": _ABI,
        "model_family": _MODEL_FAMILY,
        "source_inventory_root": source_inventory_root,
        "semantic_root": projection["semantic_root"],
        "artifact_model_digest": projection["artifact_model_digest"],
        "grammar_root": _GRAMMAR_ROOT,
        "namespace_summary_root": namespace_summary_root,
        "namespace_count": len(namespace_receipts),
        "namespaces": namespace_receipts,
        "shard_count": len(shard_receipts),
        "shards": shard_receipts,
        "tensor_count": len(all_records),
        "tensor_bytes": observed_tensor_bytes,
        "endpoint_tensor_count": endpoint_count,
        "endpoint_tensor_bytes": endpoint_bytes,
        "main_tensor_count": len(main_records),
        "main_tensor_bytes": main_bytes,
        "mtp_tensor_count": len(mtp_records),
        "mtp_tensor_bytes": mtp_bytes,
        "verification_scope": _VERIFICATION_SCOPE,
        "support_state": _SUPPORT_STATE,
    }
    if _details_sink is not None:
        _details_sink.update(
            {
                "artifact_model_digest": projection["artifact_model_digest"],
                "semantic_root": projection["semantic_root"],
                "source_inventory_root": source_inventory_root,
                "shards": tuple(private_shards),
                "records": tuple(
                    (
                        record.name,
                        record.shard_name,
                        record.dtype,
                        record.shape,
                        record.data_begin,
                        record.data_end,
                        record.semantic_tensor_root,
                        record.namespace,
                        record.role_suffix,
                        record.file_begin,
                        record.file_end,
                        record.record_root,
                    )
                    for record in all_records
                ),
            }
        )
    return result


def _verify_deepseek_v4_flash_0731_source_inventory(
    *,
    model_root: Path,
    expected_model_digest: object,
    expected_semantic_root: object,
    expected_source_inventory_root: object,
    leases: list[tuple[Path, int, tuple[object, ...]]],
    details: dict[str, object],
) -> dict[str, object]:
    model_digest = _digest(expected_model_digest, "expected model digest")
    semantic_root = _digest(expected_semantic_root, "expected semantic root")
    source_inventory_root = _digest(
        expected_source_inventory_root, "expected source inventory root"
    )
    if type(leases) is not list or leases:
        _fail("private retained lease sink must be an empty concrete list")
    if type(details) is not dict or details:
        _fail("private source detail sink must be an empty concrete dict")
    semantic_details: dict[str, object] = {}
    semantic_projection = _semantic._verify_safetensors_semantic_closure(
        model_root=model_root,
        expected_model_digest=model_digest,
        expected_semantic_root=semantic_root,
        leases=leases,
        details=semantic_details,
    )
    _semantic._require_exact_safetensors_set(model_root)
    source_root_before = _semantic._artifact._root_identity(model_root)
    _semantic._artifact._revalidate_leases(leases)
    candidate_details: dict[str, object] = {}
    result = _compile_source_inventory_projection(
        semantic_projection,
        semantic_details,
        source_inventory_root,
        candidate_details,
    )
    _semantic._require_exact_safetensors_set(model_root)
    _semantic._artifact._revalidate_leases(leases)
    _semantic._require_exact_safetensors_set(model_root)
    if _semantic._artifact._root_identity(model_root) != source_root_before:
        _fail("artifact root or parent identity changed during source replay")
    details.update(candidate_details)
    return result


def verify_deepseek_v4_flash_0731_source_inventory(
    *,
    model_root: Path,
    expected_model_digest: object,
    expected_semantic_root: object,
    expected_source_inventory_root: object,
) -> dict[str, object]:
    leases: list[tuple[Path, int, tuple[object, ...]]] = []
    details: dict[str, object] = {}
    try:
        return _verify_deepseek_v4_flash_0731_source_inventory(
            model_root=model_root,
            expected_model_digest=expected_model_digest,
            expected_semantic_root=expected_semantic_root,
            expected_source_inventory_root=expected_source_inventory_root,
            leases=leases,
            details=details,
        )
    finally:
        _semantic._artifact._close_leases(leases)


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
            "Independently verify a DeepSeek V4 Flash 0731 source inventory."
        )
    )
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--expected-model-digest", required=True)
    parser.add_argument("--expected-semantic-root", required=True)
    parser.add_argument("--expected-source-inventory-root", required=True)
    arguments = parser.parse_args(argv)
    result = verify_deepseek_v4_flash_0731_source_inventory(
        model_root=Path(arguments.model_root),
        expected_model_digest=arguments.expected_model_digest,
        expected_semantic_root=arguments.expected_semantic_root,
        expected_source_inventory_root=arguments.expected_source_inventory_root,
    )
    sys.stdout.buffer.write(_canonical(result).encode("ascii") + b"\n")
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
