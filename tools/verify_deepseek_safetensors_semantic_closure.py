#!/usr/bin/env python3
"""Independently verify one non-authorizing DeepSeek Safetensors closure."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import NoReturn, Sequence


def _load_artifact_verifier() -> ModuleType:
    """Load the sibling verifier without depending on the Python package."""
    name = "_pih_deepseek_artifact_verifier_for_semantics"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = Path(__file__).with_name("verify_deepseek_artifact_closure.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load independent artifact verifier: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


_artifact = _load_artifact_verifier()

_SCHEMA = "pih.deepseek_safetensors_semantic_verification.v1"
_ABI = "deepseek_safetensors_semantic_closure_v1"
_MODEL_FAMILY = "deepseek_v4_flash_0731"
_VERIFICATION_SCOPE = "exact_safetensors_header_semantics_non_authorizing"
_SUPPORT_STATE = "hardware_evidence_open"
_INDEX_NAME = "model.safetensors.index.json"
_TENSOR_COUNT = 72_317
_SHARD_COUNT = 48
_DECLARED_TENSOR_BYTES = 166_878_536_440
_EXPECTED_SHARDS = tuple(
    f"model-{ordinal:05d}-of-00048.safetensors"
    for ordinal in range(1, _SHARD_COUNT + 1)
)
_EXPECTED_SHARD_SET = frozenset(_EXPECTED_SHARDS)

_MAX_HEADER_BYTES = 16 << 20
_MAX_TENSORS_PER_SHARD = 4096
_MAX_TENSOR_NAME_BYTES = 512
_MAX_RANK = 8
_MAX_HEADER_JSON_NODES = 65_536
_MAX_INDEX_JSON_NODES = 1_000_016
_MAX_HEADER_JSON_STRING_BYTES = 16 << 20
_MAX_INDEX_JSON_STRING_BYTES = 64 << 20
_MAX_INDEX_BYTES = 64 << 20
_READ_CHUNK_BYTES = 64 << 10
_MAX_U32 = (1 << 32) - 1
_MAX_U64 = (1 << 64) - 1
_TENSOR_NAME = re.compile(r"[A-Za-z0-9_.]+\Z")

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
_EXPECTED_DTYPE_BYTES = {
    "BF16": 2_967_134_976,
    "I64": 18_616_320,
    "F32": 150_966_520,
    "F8_E8M0": 9_261_408_000,
    "F8_E4M3": 6_304_038_912,
    "I8": 148_176_371_712,
}

_TYPE_U32 = 1
_TYPE_U64 = 2
_TYPE_BYTES = 3
_TYPE_HASH256 = 4


def _fail(message: str) -> NoReturn:
    raise ValueError(
        f"invalid DeepSeek Safetensors semantic verification: {message}"
    )


def _digest(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or value == "0" * 64
    ):
        _fail(f"{name} must be a nonzero canonical lowercase SHA-256")
    return value


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


def _strict_json(
    source: bytes,
    name: str,
    *,
    max_nodes: int,
    max_string_bytes: int,
    max_source_bytes: int,
) -> object:
    if (
        type(source) is not bytes
        or type(max_source_bytes) is not int
        or not 0 <= len(source) <= max_source_bytes
    ):
        _fail(f"{name} exceeds its source byte bound")
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{name} contains duplicate JSON field {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> NoReturn:
        _fail(f"{name} contains non-finite number {value!r}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            _fail(f"{name} contains non-finite number {value!r}")
        return parsed

    try:
        value = json.loads(
            source.decode("utf-8", "strict"),
            object_pairs_hook=unique,
            parse_constant=reject_constant,
            parse_float=finite_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        _fail(f"{name} is malformed bounded UTF-8 JSON: {error}")
    nodes = 0
    strings = 0
    pending: list[tuple[object, int]] = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > max_nodes or depth > 16:
            _fail(f"{name} exceeds JSON structure bounds")
        if item is None or type(item) in {bool, int}:
            if type(item) is int and not -(1 << 63) <= item <= _MAX_U64:
                _fail(f"{name} contains an out-of-range JSON integer")
            continue
        if type(item) is float:
            if not math.isfinite(item):
                _fail(f"{name} contains a non-finite JSON number")
            continue
        if type(item) is str:
            try:
                strings += len(item.encode("utf-8", "strict"))
            except UnicodeEncodeError as error:
                _fail(f"{name} contains invalid Unicode: {error}")
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is dict:
            for key, child in item.items():
                try:
                    strings += len(key.encode("utf-8", "strict"))
                except UnicodeEncodeError as error:
                    _fail(f"{name} contains invalid key Unicode: {error}")
                pending.append((child, depth + 1))
        else:
            _fail(f"{name} contains a non-JSON value")
        if strings > max_string_bytes:
            _fail(f"{name} exceeds JSON string bounds")
    return value


def _nonnegative_u64(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_U64:
        _fail(f"{name} must be a nonnegative uint64")
    return value


@dataclass(frozen=True, slots=True)
class _TensorSemantic:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_begin: int
    data_end: int

    def __post_init__(self) -> None:
        if type(self.name) is not str:
            _fail("tensor name must be a string")
        try:
            name_bytes = self.name.encode("ascii", "strict")
        except UnicodeEncodeError as error:
            _fail(f"tensor name is not canonical ASCII: {error}")
        if (
            not name_bytes
            or len(name_bytes) > _MAX_TENSOR_NAME_BYTES
            or _TENSOR_NAME.fullmatch(self.name) is None
        ):
            _fail("tensor name is outside the canonical runtime grammar")
        if type(self.dtype) is not str or self.dtype not in _DTYPE_ELEMENT_BYTES:
            _fail("tensor dtype is unsupported")
        if type(self.shape) is not tuple or not 1 <= len(self.shape) <= _MAX_RANK:
            _fail("tensor rank is outside the admitted non-scalar range")
        elements = 1
        for dimension in self.shape:
            if type(dimension) is not int or not 0 < dimension <= _MAX_U64:
                _fail("tensor dimension must be a positive uint64")
            elements = _checked_mul(elements, dimension, "tensor shape product")
        byte_count = _checked_mul(
            elements, _DTYPE_ELEMENT_BYTES[self.dtype], "tensor byte count"
        )
        if (
            type(self.data_begin) is not int
            or type(self.data_end) is not int
            or not 0 <= self.data_begin < self.data_end <= _MAX_U64
        ):
            _fail("tensor data offsets are invalid or empty")
        if self.data_end - self.data_begin != byte_count:
            _fail("tensor byte count differs from dtype and shape")


@dataclass(frozen=True, slots=True)
class _ParsedShardSemantic:
    shard_name: str
    file_bytes: int
    header_bytes: int
    data_bytes: int
    header_prefix_sha256: str
    tensors: tuple[_TensorSemantic, ...]


def _validate_contiguous_ranges(
    tensors: Sequence[_TensorSemantic], data_bytes: int
) -> None:
    ordered = sorted(
        tensors,
        key=lambda tensor: (tensor.data_begin, tensor.data_end, tensor.name),
    )
    cursor = 0
    for tensor in ordered:
        if tensor.data_begin < cursor:
            _fail("Safetensors tensor data ranges overlap")
        if tensor.data_begin > cursor:
            _fail("Safetensors tensor data ranges contain a gap")
        cursor = tensor.data_end
    if cursor != data_bytes:
        _fail("Safetensors tensor data ranges do not exactly cover the file")


def _parse_safetensors_header(
    prefix: bytes,
    file_bytes: int,
    shard_name: str,
) -> _ParsedShardSemantic:
    if (
        type(prefix) is not bytes
        or len(prefix) < 8
        or type(file_bytes) is not int
        or file_bytes < 8
    ):
        _fail("Safetensors length prefix is missing")
    if type(shard_name) is not str or shard_name not in _EXPECTED_SHARD_SET:
        _fail("Safetensors shard name is outside the exact shard set")
    header_bytes = int.from_bytes(prefix[:8], "little")
    if not 0 < header_bytes <= _MAX_HEADER_BYTES:
        _fail("Safetensors header length is outside the exact runtime bound")
    prefix_bytes = _checked_add(8, header_bytes, "Safetensors prefix bytes")
    if len(prefix) < prefix_bytes or file_bytes < prefix_bytes:
        _fail("Safetensors header prefix is truncated")
    if len(prefix) != prefix_bytes:
        _fail("Safetensors parser received bytes beyond the header prefix")
    data_bytes = file_bytes - prefix_bytes
    if data_bytes <= 0:
        _fail("Safetensors shard has no tensor payload")
    value = _strict_json(
        prefix[8:],
        f"Safetensors header {shard_name}",
        max_nodes=_MAX_HEADER_JSON_NODES,
        max_string_bytes=_MAX_HEADER_JSON_STRING_BYTES,
        max_source_bytes=_MAX_HEADER_BYTES,
    )
    if type(value) is not dict or not value:
        _fail("Safetensors header root must be a nonempty object")
    if "__metadata__" in value:
        metadata = value["__metadata__"]
        if type(metadata) is not dict or any(
            type(key) is not str or type(item) is not str
            for key, item in metadata.items()
        ):
            _fail("Safetensors metadata must be map[str,str]")
        for key, item in metadata.items():
            if (
                len(key.encode("utf-8", "strict")) > _MAX_TENSOR_NAME_BYTES
                or len(item.encode("utf-8", "strict"))
                > _MAX_TENSOR_NAME_BYTES
            ):
                _fail("Safetensors metadata string exceeds the runtime bound")
    tensors: list[_TensorSemantic] = []
    for tensor_name, record in value.items():
        if tensor_name == "__metadata__":
            continue
        if len(tensors) == _MAX_TENSORS_PER_SHARD:
            _fail("Safetensors tensor count exceeds the per-shard bound")
        if type(record) is not dict or set(record) != {
            "dtype",
            "shape",
            "data_offsets",
        }:
            _fail("Safetensors tensor record must have exactly three fields")
        dtype = record["dtype"]
        shape_value = record["shape"]
        offsets = record["data_offsets"]
        if type(dtype) is not str or dtype not in _DTYPE_ELEMENT_BYTES:
            _fail("Safetensors tensor dtype is unsupported")
        if (
            type(shape_value) is not list
            or not 1 <= len(shape_value) <= _MAX_RANK
        ):
            _fail("Safetensors tensor rank is outside the admitted non-scalar range")
        shape: list[int] = []
        for dimension in shape_value:
            if type(dimension) is not int or not 0 < dimension <= _MAX_U64:
                _fail("Safetensors tensor dimension must be a positive uint64")
            shape.append(dimension)
        if type(offsets) is not list or len(offsets) != 2:
            _fail("Safetensors data offsets must contain two integers")
        tensors.append(
            _TensorSemantic(
                tensor_name,
                dtype,
                tuple(shape),
                _nonnegative_u64(offsets[0], "Safetensors data offset"),
                _nonnegative_u64(offsets[1], "Safetensors data offset"),
            )
        )
    if not tensors:
        _fail("Safetensors header contains no tensors")
    tensors.sort(key=lambda tensor: tensor.name)
    if len({tensor.name for tensor in tensors}) != len(tensors):
        _fail("Safetensors header contains duplicate tensor names")
    _validate_contiguous_ranges(tensors, data_bytes)
    return _ParsedShardSemantic(
        shard_name=shard_name,
        file_bytes=file_bytes,
        header_bytes=header_bytes,
        data_bytes=data_bytes,
        header_prefix_sha256=hashlib.sha256(prefix).hexdigest(),
        tensors=tuple(tensors),
    )


def _parse_weight_index(source: bytes) -> dict[str, str]:
    value = _strict_json(
        source,
        "weight index semantic replay",
        max_nodes=_MAX_INDEX_JSON_NODES,
        max_string_bytes=_MAX_INDEX_JSON_STRING_BYTES,
        max_source_bytes=_MAX_INDEX_BYTES,
    )
    if type(value) is not dict or set(value) != {"metadata", "weight_map"}:
        _fail("weight index root fields are invalid")
    metadata = value["metadata"]
    weight_map = value["weight_map"]
    if (
        type(metadata) is not dict
        or set(metadata) != {"total_size"}
        or type(metadata["total_size"]) is not int
        or metadata["total_size"] != _DECLARED_TENSOR_BYTES
    ):
        _fail("weight index total_size differs from exact DeepSeek geometry")
    if type(weight_map) is not dict or len(weight_map) != _TENSOR_COUNT:
        _fail("weight index tensor count differs from exact DeepSeek geometry")
    result: dict[str, str] = {}
    for tensor_name, shard_name in weight_map.items():
        if type(tensor_name) is not str:
            _fail("weight index tensor name is not a string")
        try:
            name_bytes = tensor_name.encode("ascii", "strict")
        except UnicodeEncodeError as error:
            _fail(f"weight index tensor name is not canonical ASCII: {error}")
        if (
            not name_bytes
            or len(name_bytes) > _MAX_TENSOR_NAME_BYTES
            or _TENSOR_NAME.fullmatch(tensor_name) is None
        ):
            _fail("weight index tensor name is outside the runtime grammar")
        if type(shard_name) is not str or shard_name not in _EXPECTED_SHARD_SET:
            _fail("weight index names a shard outside the exact shard set")
        result[tensor_name] = shard_name
    if frozenset(result.values()) != _EXPECTED_SHARD_SET:
        _fail("weight index does not reference the exact 48-shard set")
    return result


def _tensor_root(shard_name: str, tensor: _TensorSemantic) -> str:
    shape = b"".join(
        _u64(dimension, "tensor shape dimension")
        for dimension in tensor.shape
    )
    return _typed_sha256(
        "pih:deepseek-safetensors-tensor:v1",
        (
            (1, _TYPE_BYTES, tensor.name.encode("ascii")),
            (2, _TYPE_BYTES, shard_name.encode("ascii")),
            (3, _TYPE_BYTES, tensor.dtype.encode("ascii")),
            (4, _TYPE_U32, _u32(len(tensor.shape), "tensor rank")),
            (5, _TYPE_BYTES, shape),
            (6, _TYPE_U64, _u64(tensor.data_begin, "tensor data begin")),
            (7, _TYPE_U64, _u64(tensor.data_end, "tensor data end")),
        ),
    )


def _tensor_set_root(
    shard_name: str, tensors: Sequence[_TensorSemantic]
) -> str:
    fields: list[tuple[int, int, bytes]] = [
        (1, _TYPE_U32, _u32(len(tensors), "shard tensor count"))
    ]
    fields.extend(
        (
            100 + ordinal,
            _TYPE_HASH256,
            bytes.fromhex(_tensor_root(shard_name, tensor)),
        )
        for ordinal, tensor in enumerate(tensors)
    )
    return _typed_sha256(
        "pih:deepseek-safetensors-tensor-set:v1", fields
    )


def _shard_root(
    *,
    artifact_object_root: str,
    shard_name: str,
    file_bytes: int,
    header_bytes: int,
    data_bytes: int,
    header_prefix_sha256: str,
    tensor_count: int,
    tensor_root: str,
) -> str:
    return _typed_sha256(
        "pih:deepseek-safetensors-shard:v1",
        (
            (
                1,
                _TYPE_HASH256,
                bytes.fromhex(_digest(artifact_object_root, "artifact object root")),
            ),
            (2, _TYPE_BYTES, shard_name.encode("ascii")),
            (3, _TYPE_U64, _u64(file_bytes, "shard file bytes")),
            (4, _TYPE_U64, _u64(header_bytes, "shard header bytes")),
            (5, _TYPE_U64, _u64(data_bytes, "shard data bytes")),
            (
                6,
                _TYPE_HASH256,
                bytes.fromhex(
                    _digest(header_prefix_sha256, "header prefix SHA-256")
                ),
            ),
            (7, _TYPE_U32, _u32(tensor_count, "shard tensor count")),
            (
                8,
                _TYPE_HASH256,
                bytes.fromhex(_digest(tensor_root, "tensor-set root")),
            ),
        ),
    )


def _dtype_summary_root(dtype_bytes: Sequence[tuple[str, int]]) -> str:
    fields: list[tuple[int, int, bytes]] = [
        (1, _TYPE_U32, _u32(len(dtype_bytes), "dtype count"))
    ]
    for ordinal, (dtype, byte_count) in enumerate(dtype_bytes):
        if type(dtype) is not str or dtype not in _DTYPE_ELEMENT_BYTES:
            _fail("dtype summary contains an unsupported dtype")
        fields.append(
            (
                100 + ordinal,
                _TYPE_BYTES,
                dtype.encode("ascii")
                + b"\0"
                + _u64(byte_count, "dtype bytes"),
            )
        )
    return _typed_sha256("pih:deepseek-safetensors-dtypes:v1", fields)


def _semantic_root(
    *,
    artifact_model_digest: str,
    artifact_index_object_root: str,
    shard_roots: Sequence[str],
    tensor_count: int,
    tensor_bytes: int,
    total_header_bytes: int,
    dtype_bytes: Sequence[tuple[str, int]],
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
            bytes.fromhex(
                _digest(artifact_index_object_root, "artifact index object root")
            ),
        ),
        (5, _TYPE_U32, _u32(len(shard_roots), "semantic shard count")),
        (6, _TYPE_U64, _u64(tensor_count, "semantic tensor count")),
        (7, _TYPE_U64, _u64(tensor_bytes, "semantic tensor bytes")),
        (8, _TYPE_U64, _u64(total_header_bytes, "semantic header bytes")),
        (9, _TYPE_HASH256, bytes.fromhex(_dtype_summary_root(dtype_bytes))),
    ]
    fields.extend(
        (100 + ordinal, _TYPE_HASH256, bytes.fromhex(root))
        for ordinal, root in enumerate(shard_roots)
    )
    return _typed_sha256(
        "pih:deepseek-safetensors-semantic-closure:v1", fields
    )


def _read_exact_at(
    descriptor: int,
    offset: int,
    byte_count: int,
    name: str,
) -> bytes:
    if (
        type(descriptor) is not int
        or descriptor < 0
        or type(offset) is not int
        or type(byte_count) is not int
        or offset < 0
        or byte_count < 0
    ):
        _fail("descriptor range is invalid")
    try:
        os.lseek(descriptor, offset, os.SEEK_SET)
    except OSError as error:
        _fail(f"cannot seek retained descriptor for {name}: {error}")
    chunks: list[bytes] = []
    total = 0
    while total < byte_count:
        requested = min(_READ_CHUNK_BYTES, byte_count - total)
        try:
            chunk = os.read(descriptor, requested)
        except InterruptedError:
            continue
        except OSError as error:
            _fail(f"cannot read retained descriptor for {name}: {error}")
        if not chunk or len(chunk) > requested:
            _fail(f"retained descriptor ended during bounded read: {name}")
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


def _read_shard_header(
    descriptor: int, file_bytes: int, shard_name: str
) -> _ParsedShardSemantic:
    length_prefix = _read_exact_at(descriptor, 0, 8, shard_name)
    header_bytes = int.from_bytes(length_prefix, "little")
    if not 0 < header_bytes <= _MAX_HEADER_BYTES:
        _fail("Safetensors header length is outside the exact runtime bound")
    prefix_bytes = _checked_add(8, header_bytes, "Safetensors prefix bytes")
    if prefix_bytes >= file_bytes:
        _fail("Safetensors header is truncated or has no payload")
    header = _read_exact_at(descriptor, 8, header_bytes, shard_name)
    return _parse_safetensors_header(
        length_prefix + header, file_bytes, shard_name
    )


@dataclass(frozen=True, slots=True)
class _ArtifactAntecedent:
    model_digest: str
    index_object_root: str
    index_source: bytes
    shard_objects: tuple[tuple[str, int, str], ...]


def _require_exact_safetensors_set(model_root: Path) -> None:
    if _artifact._listed_shards(model_root) != _EXPECTED_SHARD_SET:
        _fail("Safetensors shard set changed during semantic verification")


def _verify_artifact_antecedent(
    *,
    model_root: Path,
    expected_model_digest: str,
    leases: list[tuple[Path, int, tuple[object, ...]]],
) -> _ArtifactAntecedent:
    details: dict[str, object] = {}
    projection = _artifact._verify_artifact_closure(
        model_root=model_root,
        expected_model_digest=expected_model_digest,
        leases=leases,
        details=details,
    )
    if (
        type(projection) is not dict
        or projection.get("model_digest") != expected_model_digest
        or set(details)
        != {"index_source", "index_object_root", "shard_objects"}
    ):
        _fail("artifact verifier returned an invalid identity projection")
    index_source = details["index_source"]
    index_object_root = details["index_object_root"]
    shard_objects = details["shard_objects"]
    if type(index_source) is not bytes:
        _fail("artifact verifier returned an invalid index source")
    if type(index_object_root) is not str:
        _fail("artifact verifier returned an invalid index object root")
    _digest(index_object_root, "artifact index object root")
    if (
        type(shard_objects) is not tuple
        or len(shard_objects) != _SHARD_COUNT
        or any(
            type(item) is not tuple
            or len(item) != 3
            or type(item[0]) is not str
            or type(item[1]) is not int
            or not 0 < item[1] <= _MAX_U64
            or type(item[2]) is not str
            for item in shard_objects
        )
    ):
        _fail("artifact verifier returned invalid shard identities")
    typed_shards = tuple(shard_objects)
    if tuple(item[0] for item in typed_shards) != _EXPECTED_SHARDS:
        _fail("artifact verifier returned a noncanonical shard identity set")
    for _name, _length, object_root in typed_shards:
        _digest(object_root, "artifact shard object root")
    return _ArtifactAntecedent(
        model_digest=expected_model_digest,
        index_object_root=index_object_root,
        index_source=index_source,
        shard_objects=typed_shards,
    )

def _compile_semantic_projection(
    artifact: _ArtifactAntecedent,
    weight_map: dict[str, str],
    parsed_shards: tuple[_ParsedShardSemantic, ...],
    expected_semantic_root: str,
) -> dict[str, object]:
    expected = _digest(expected_semantic_root, "expected semantic root")
    if len(weight_map) != _TENSOR_COUNT:
        _fail("weight index map has the wrong tensor count")
    canonical = tuple(sorted(parsed_shards, key=lambda shard: shard.shard_name))
    if tuple(shard.shard_name for shard in canonical) != _EXPECTED_SHARDS:
        _fail("parsed shard set differs from the exact 48-shard set")
    artifact_shards = {
        name: (length, object_root)
        for name, length, object_root in artifact.shard_objects
    }
    if tuple(sorted(artifact_shards)) != _EXPECTED_SHARDS:
        _fail("artifact antecedent shard set is invalid")

    observed: dict[str, str] = {}
    dtype_totals: dict[str, int] = {}
    shard_roots: list[str] = []
    tensor_total = 0
    payload_total = 0
    header_total = 0
    for parsed in canonical:
        artifact_length, artifact_object_root = artifact_shards[parsed.shard_name]
        if parsed.file_bytes != artifact_length:
            _fail("parsed shard bytes differ from artifact object identity")
        for tensor in parsed.tensors:
            if tensor.name in observed:
                _fail("header tensor set contains a cross-shard duplicate")
            observed[tensor.name] = parsed.shard_name
            if weight_map.get(tensor.name) != parsed.shard_name:
                _fail("index/header bijection differs at the declared shard")
            tensor_bytes = tensor.data_end - tensor.data_begin
            dtype_totals[tensor.dtype] = _checked_add(
                dtype_totals.get(tensor.dtype, 0),
                tensor_bytes,
                "dtype byte total",
            )
        tensor_total = _checked_add(
            tensor_total, len(parsed.tensors), "tensor count"
        )
        payload_total = _checked_add(
            payload_total, parsed.data_bytes, "payload byte total"
        )
        header_total = _checked_add(
            header_total, parsed.header_bytes, "header byte total"
        )
        tensor_set_root = _tensor_set_root(parsed.shard_name, parsed.tensors)
        shard_roots.append(
            _shard_root(
                artifact_object_root=artifact_object_root,
                shard_name=parsed.shard_name,
                file_bytes=parsed.file_bytes,
                header_bytes=parsed.header_bytes,
                data_bytes=parsed.data_bytes,
                header_prefix_sha256=parsed.header_prefix_sha256,
                tensor_count=len(parsed.tensors),
                tensor_root=tensor_set_root,
            )
        )
    if observed != weight_map:
        _fail("index/header tensor sets do not form an exact reverse bijection")
    if tensor_total != _TENSOR_COUNT:
        _fail("header tensor count differs from exact DeepSeek geometry")
    if payload_total != _DECLARED_TENSOR_BYTES:
        _fail("header payload bytes differ from exact DeepSeek geometry")
    if dtype_totals != _EXPECTED_DTYPE_BYTES:
        _fail("header dtype bytes differ from exact DeepSeek geometry")
    dtype_items = tuple(sorted(dtype_totals.items()))
    semantic_root = _semantic_root(
        artifact_model_digest=artifact.model_digest,
        artifact_index_object_root=artifact.index_object_root,
        shard_roots=shard_roots,
        tensor_count=tensor_total,
        tensor_bytes=payload_total,
        total_header_bytes=header_total,
        dtype_bytes=dtype_items,
    )
    if semantic_root != expected:
        _fail("expected semantic root differs from independently replayed root")
    return {
        "schema": _SCHEMA,
        "abi": _ABI,
        "model_family": _MODEL_FAMILY,
        "semantic_root": semantic_root,
        "artifact_model_digest": artifact.model_digest,
        "artifact_index_object_root": artifact.index_object_root,
        "shard_count": len(shard_roots),
        "tensor_count": tensor_total,
        "tensor_bytes": payload_total,
        "total_header_bytes": header_total,
        "dtype_bytes": dict(dtype_items),
        "verification_scope": _VERIFICATION_SCOPE,
        "support_state": _SUPPORT_STATE,
    }


def _verify_safetensors_semantic_closure(
    *,
    model_root: Path,
    expected_model_digest: str,
    expected_semantic_root: str,
    leases: list[tuple[Path, int, tuple[object, ...]]],
    details: dict[str, object],
) -> dict[str, object]:
    expected = _digest(expected_semantic_root, "expected semantic root")
    if type(leases) is not list or leases:
        _fail("private retained lease sink must be an empty list")
    if type(details) is not dict or details:
        _fail("private semantic detail sink must be an empty dict")
    artifact = _verify_artifact_antecedent(
        model_root=model_root,
        expected_model_digest=expected_model_digest,
        leases=leases,
    )
    _require_exact_safetensors_set(model_root)
    semantic_root_before = _artifact._root_identity(model_root)
    _artifact._revalidate_leases(leases)
    descriptors: dict[str, int] = {}
    for path, descriptor, _identity in leases:
        if path.name in descriptors:
            _fail("retained artifact descriptor set contains a duplicate basename")
        descriptors[path.name] = descriptor
    expected_members = {
        _artifact._CONFIG_NAME,
        _INDEX_NAME,
        *_EXPECTED_SHARDS,
    }
    if set(descriptors) != expected_members:
        _fail("retained artifact descriptor set differs from exact closure members")
    weight_map = _parse_weight_index(artifact.index_source)
    artifact_shards = {
        name: (length, object_root)
        for name, length, object_root in artifact.shard_objects
    }
    parsed: list[_ParsedShardSemantic] = []
    for shard_name in _EXPECTED_SHARDS:
        descriptor = descriptors[shard_name]
        try:
            file_bytes = int(os.fstat(descriptor).st_size)
        except OSError as error:
            _fail(
                f"retained shard descriptor is unavailable: {shard_name}: {error}"
            )
        if file_bytes != artifact_shards[shard_name][0]:
            _fail("retained shard size differs from artifact object identity")
        parsed.append(_read_shard_header(descriptor, file_bytes, shard_name))
    parsed_shards = tuple(parsed)
    result = _compile_semantic_projection(
        artifact,
        weight_map,
        parsed_shards,
        expected,
    )
    _require_exact_safetensors_set(model_root)
    _artifact._revalidate_leases(leases)
    _require_exact_safetensors_set(model_root)
    if _artifact._root_identity(model_root) != semantic_root_before:
        _fail("artifact root or parent identity changed during semantic replay")

    shard_details: list[tuple[object, ...]] = []
    for shard in parsed_shards:
        artifact_length, artifact_object_root = artifact_shards[shard.shard_name]
        tensor_set_root = _tensor_set_root(shard.shard_name, shard.tensors)
        semantic_shard_root = _shard_root(
            artifact_object_root=artifact_object_root,
            shard_name=shard.shard_name,
            file_bytes=artifact_length,
            header_bytes=shard.header_bytes,
            data_bytes=shard.data_bytes,
            header_prefix_sha256=shard.header_prefix_sha256,
            tensor_count=len(shard.tensors),
            tensor_root=tensor_set_root,
        )
        tensor_details = tuple(
            (
                tensor.name,
                tensor.dtype,
                tensor.shape,
                tensor.data_begin,
                tensor.data_end,
                _tensor_root(shard.shard_name, tensor),
            )
            for tensor in shard.tensors
        )
        shard_details.append(
            (
                shard.shard_name,
                artifact_object_root,
                shard.file_bytes,
                shard.header_bytes,
                shard.data_bytes,
                shard.header_prefix_sha256,
                tensor_set_root,
                semantic_shard_root,
                tensor_details,
            )
        )
    candidate_details = {
        "artifact_index_object_root": artifact.index_object_root,
        "artifact_model_digest": artifact.model_digest,
        "semantic_root": result["semantic_root"],
        "shards": tuple(shard_details),
        "weight_map": tuple(sorted(weight_map.items())),
    }
    details.update(candidate_details)
    return result


def verify_safetensors_semantic_closure(
    *,
    model_root: Path,
    expected_model_digest: str,
    expected_semantic_root: str,
) -> dict[str, object]:
    leases: list[tuple[Path, int, tuple[object, ...]]] = []
    details: dict[str, object] = {}
    try:
        return _verify_safetensors_semantic_closure(
            model_root=model_root,
            expected_model_digest=expected_model_digest,
            expected_semantic_root=expected_semantic_root,
            leases=leases,
            details=details,
        )
    finally:
        _artifact._close_leases(leases)


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
        description="Independently verify a DeepSeek Safetensors semantic closure."
    )
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--expected-model-digest", required=True)
    parser.add_argument("--expected-semantic-root", required=True)
    arguments = parser.parse_args(argv)
    result = verify_safetensors_semantic_closure(
        model_root=Path(arguments.model_root),
        expected_model_digest=arguments.expected_model_digest,
        expected_semantic_root=arguments.expected_semantic_root,
    )
    sys.stdout.buffer.write(_canonical(result).encode("ascii") + b"\n")
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
