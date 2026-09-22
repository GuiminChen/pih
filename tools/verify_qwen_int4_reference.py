from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import mmap
from pathlib import Path
import struct
import sys
from typing import Mapping, NoReturn, Sequence


_METADATA_BYTES = 1_048_576
_OFFICIAL_FILE_BYTES = 539_758_592
_MAX_SOURCE_BYTES = 1_520_041_984
_MAX_HEADER_BYTES = 16 << 20
_MAGIC = b"XINGW4A1"
_FORMAT = "xing-w4a16-sym-g128-v1"
_LAYOUT = "canonical-nk-low-nibble-k-v1"


def _official_shapes() -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (151936, 1024),
        "lm_head.weight": (151936, 1024),
        "model.norm.weight": (1024,),
    }
    linears = {
        "self_attn.q_proj.weight": (2048, 1024),
        "self_attn.k_proj.weight": (1024, 1024),
        "self_attn.v_proj.weight": (1024, 1024),
        "self_attn.o_proj.weight": (1024, 2048),
        "mlp.gate_proj.weight": (3072, 1024),
        "mlp.up_proj.weight": (3072, 1024),
        "mlp.down_proj.weight": (1024, 3072),
    }
    norms = {
        "input_layernorm.weight": (1024,),
        "post_attention_layernorm.weight": (1024,),
        "self_attn.q_norm.weight": (128,),
        "self_attn.k_norm.weight": (128,),
    }
    for layer in range(28):
        prefix = f"model.layers.{layer}."
        result.update({prefix + name: shape for name, shape in linears.items()})
        result.update({prefix + name: shape for name, shape in norms.items()})
    assert len(result) == 311
    return result


_OFFICIAL_SHAPES = _official_shapes()


def _fail(message: str) -> NoReturn:
    raise ValueError(f"invalid Qwen INT4 reference input: {message}")


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON field {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class _Tensor:
    dtype: str
    shape: tuple[int, ...]
    first: int
    last: int


@dataclass(frozen=True)
class _Record:
    kind: int
    identity: str
    source: str
    alias: str
    logical_bytes: int
    offset: int
    extent: int
    digest: bytes


def _safetensors(mm: mmap.mmap) -> tuple[int, dict[str, _Tensor]]:
    if len(mm) < 8:
        _fail("source Safetensors header is truncated")
    header_bytes = struct.unpack_from("<Q", mm, 0)[0]
    if header_bytes == 0 or header_bytes > _MAX_HEADER_BYTES or 8 + header_bytes > len(mm):
        _fail("source Safetensors header length is invalid")
    try:
        root = json.loads(mm[8 : 8 + header_bytes], object_pairs_hook=_unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(str(error))
    if type(root) is not dict:
        _fail("source Safetensors header is not an object")
    data_start = 8 + header_bytes
    tensors: dict[str, _Tensor] = {}
    spans: list[tuple[int, int]] = []
    for name, raw in root.items():
        if name == "__metadata__":
            continue
        if type(name) is not str or not name or type(raw) is not dict or set(raw) != {"dtype", "shape", "data_offsets"}:
            _fail("source tensor schema is invalid")
        dtype, shape, offsets = raw["dtype"], raw["shape"], raw["data_offsets"]
        if dtype != "BF16" or type(shape) is not list or not shape or any(type(x) is not int or x <= 0 for x in shape):
            _fail(f"source tensor {name} is not bounded BF16")
        if type(offsets) is not list or len(offsets) != 2 or any(type(x) is not int or x < 0 for x in offsets):
            _fail(f"source tensor {name} offsets are invalid")
        first, last = offsets
        elements = math.prod(shape)
        if last - first != elements * 2 or last > len(mm) - data_start:
            _fail(f"source tensor {name} extent is invalid")
        spans.append((first, last))
        tensors[name] = _Tensor(dtype, tuple(shape), data_start + first, data_start + last)
    ordered = sorted(spans)
    if (not tensors or ordered[0][0] != 0 or ordered[-1][1] != len(mm) - data_start or
            any(left[1] != right[0] for left, right in zip(ordered, ordered[1:]))):
        _fail("source tensor payload coverage is not exact")
    return data_start, tensors


def _metadata(mm: mmap.mmap) -> tuple[dict[str, str], tuple[_Record, ...]]:
    if len(mm) != _OFFICIAL_FILE_BYTES or mm[:8] != _MAGIC:
        _fail("artifact size or magic is not canonical")
    version, encoded, count, reserved = struct.unpack_from("<IIII", mm, 8)
    if version != 1 or reserved != 0 or not 160 <= encoded <= _METADATA_BYTES or count != 507:
        _fail("artifact metadata header is invalid")
    wire = bytearray(mm[:encoded])
    stored_checksum = bytes(wire[120:152])
    wire[120:152] = bytes(32)
    if hashlib.sha256(wire).digest() != stored_checksum or any(mm[encoded:_METADATA_BYTES]):
        _fail("artifact metadata checksum or zero tail is invalid")
    roots = {
        "source_artifact_root": mm[24:56].hex(),
        "source_binding_root": mm[56:88].hex(),
        "disposition_root": mm[88:120].hex(),
    }
    if any(value == "0" * 64 for value in roots.values()):
        _fail("artifact authority root is empty")
    format_len, layout_len, reserved2 = struct.unpack_from("<HHI", mm, 152)
    position = 160
    def text(length: int) -> str:
        nonlocal position
        if length > 65535 or position + length > encoded:
            _fail("artifact metadata string is truncated")
        try:
            value = mm[position : position + length].decode("utf-8")
        except UnicodeDecodeError as error:
            _fail(str(error))
        position += length
        return value
    if reserved2 != 0 or text(format_len) != _FORMAT or text(layout_len) != _LAYOUT:
        _fail("artifact format or layout identifier drifted")
    records: list[_Record] = []
    previous = ""
    next_offset = _METADATA_BYTES
    for _ in range(count):
        if position + 64 > encoded:
            _fail("artifact record is truncated")
        kind, zero, il, sl, al, logical, offset, extent = struct.unpack_from("<BBHHHQQQ", mm, position)
        position += 32
        digest = bytes(mm[position : position + 32]); position += 32
        identity, source, alias = text(il), text(sl), text(al)
        if zero != 0 or kind not in {1, 2, 3, 4, 5} or not identity or identity <= previous:
            _fail("artifact record order or kind is invalid")
        previous = identity
        if kind == 5:
            if logical != 0 or offset != 0 or extent != 0 or digest != bytes(32) or not alias:
                _fail("artifact alias record is invalid")
        elif offset != next_offset or offset % 4096 or extent % 4096 or logical == 0 or logical > extent or digest == bytes(32):
            _fail("artifact payload record is invalid")
        if kind != 5:
            next_offset += extent
        records.append(_Record(kind, identity, source, alias, logical, offset, extent, digest))
    if position != encoded or sum(record.kind != 5 for record in records) != 506 or next_offset != len(mm):
        _fail("artifact encoded record coverage drifted")
    return roots, tuple(records)


def _bf16(mm: mmap.mmap, offset: int) -> float:
    bits = struct.unpack_from("<H", mm, offset)[0]
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def _quantized_group(values: Sequence[float]) -> tuple[int, tuple[int, ...]]:
    if not values or len(values) > 128 or any(not math.isfinite(value) for value in values):
        _fail("sample group is invalid")
    maximum = max(abs(value) for value in values)
    scale = 1.0 if maximum == 0.0 else maximum / 7.0
    scale_bits = struct.unpack("<H", struct.pack("<e", scale))[0]
    restored = struct.unpack("<e", struct.pack("<H", scale_bits))[0]
    if maximum != 0.0 and (not math.isfinite(restored) or restored == 0.0):
        _fail("sample scale is outside FP16")
    quantized = tuple(0 if maximum == 0.0 else max(-7, min(7, round(value / restored))) for value in values)
    return scale_bits, quantized


def _verify_samples(source: mmap.mmap, artifact: mmap.mmap, tensors: Mapping[str, _Tensor], records: Sequence[_Record]) -> int:
    by_source: dict[str, dict[int, _Record]] = {}
    for record in records:
        if record.kind in {1, 2}:
            pair = by_source.setdefault(record.source, {})
            if record.kind in pair:
                _fail(f"linear disposition for {record.source} is duplicated")
            pair[record.kind] = record
    checked = 0
    for name, pair in by_source.items():
        if set(pair) != {1, 2} or name not in tensors:
            _fail(f"linear disposition for {name} is incomplete")
        tensor = tensors[name]
        if len(tensor.shape) != 2:
            _fail(f"linear source {name} is not rank two")
        rows, columns = tensor.shape
        groups = (columns + 127) // 128
        packed_columns = (columns + 1) // 2
        if pair[1].logical_bytes != rows * packed_columns or pair[2].logical_bytes != rows * groups * 2:
            _fail(f"linear artifact shape for {name} drifted")
        candidates = {(0, 0), (rows - 1, groups - 1), (int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little") % rows, int.from_bytes(hashlib.sha256(name.encode()).digest()[8:16], "little") % groups)}
        for row, group in sorted(candidates):
            first = group * 128; last = min(columns, first + 128)
            values = [_bf16(source, tensor.first + 2 * (row * columns + column)) for column in range(first, last)]
            scale_bits, expected = _quantized_group(values)
            observed_scale = struct.unpack_from("<H", artifact, pair[2].offset + 2 * (row * groups + group))[0]
            if scale_bits != observed_scale:
                _fail(f"sample scale mismatch for {name}")
            for relative, column in enumerate(range(first, last)):
                packed = artifact[pair[1].offset + row * packed_columns + column // 2]
                nibble = (packed >> (4 if column & 1 else 0)) & 15
                observed = nibble - 16 if nibble >= 8 else nibble
                if observed == -8 or observed != expected[relative]:
                    _fail(f"sample nibble mismatch for {name}")
            checked += 1
    return checked


def _digest_range(mm: mmap.mmap, first: int, count: int) -> bytes:
    digest = hashlib.sha256()
    position = first
    remaining = count
    while remaining:
        size = min(1 << 20, remaining)
        digest.update(mm[position : position + size])
        position += size
        remaining -= size
    return digest.digest()


def _equal_ranges(left: mmap.mmap, left_first: int, right: mmap.mmap,
                  right_first: int, count: int) -> bool:
    position = 0
    while position < count:
        size = min(1 << 20, count - position)
        if left[left_first + position : left_first + position + size] != right[
            right_first + position : right_first + position + size
        ]:
            return False
        position += size
    return True


def run(source_path: Path, artifact_path: Path) -> dict[str, object]:
    if not source_path.is_absolute() or not artifact_path.is_absolute() or source_path == artifact_path:
        _fail("paths must be distinct and absolute")
    if not source_path.is_file() or not artifact_path.is_file() or source_path.stat().st_size > _MAX_SOURCE_BYTES:
        _fail("source or artifact file bound is invalid")
    with source_path.open("rb") as source_file, artifact_path.open("rb") as artifact_file:
        with mmap.mmap(source_file.fileno(), 0, access=mmap.ACCESS_READ) as source, mmap.mmap(artifact_file.fileno(), 0, access=mmap.ACCESS_READ) as artifact:
            _, tensors = _safetensors(source)
            roots, records = _metadata(artifact)
            if set(tensors) != set(_OFFICIAL_SHAPES) or any(
                tensors[name].shape != shape
                for name, shape in _OFFICIAL_SHAPES.items()
            ):
                _fail("source tensor names or shapes are not the official closure")
            source_names = {record.source for record in records}
            if source_names != set(_OFFICIAL_SHAPES):
                _fail("artifact disposition does not exactly cover source tensors")
            counts = {kind: sum(record.kind == kind for record in records)
                      for kind in range(1, 6)}
            if counts != {1: 196, 2: 196, 3: 1, 4: 113, 5: 1}:
                _fail("artifact disposition kind counts drifted")
            for record in records:
                if record.kind == 1 and record.identity != record.source + ".packed_values":
                    _fail(f"packed identity drifted for {record.source}")
                if record.kind == 2 and record.identity != record.source + ".scales":
                    _fail(f"scale identity drifted for {record.source}")
                if record.kind in {3, 4} and record.identity != record.source:
                    _fail(f"BF16 identity drifted for {record.source}")
                if record.kind == 5 and (
                    record.identity != "lm_head.weight" or
                    record.source != "lm_head.weight" or
                    record.alias != "model.embed_tokens.weight"
                ):
                    _fail("tied LM-head alias drifted")
                if record.kind != 5 and _digest_range(artifact, record.offset, record.logical_bytes) != record.digest:
                    _fail(f"payload digest mismatch for {record.identity}")
                if record.kind != 5 and any(artifact[
                    record.offset + record.logical_bytes : record.offset + record.extent
                ]):
                    _fail(f"payload padding is not zero for {record.identity}")
                if record.kind in {3, 4}:
                    tensor = tensors[record.source]
                    if record.logical_bytes != tensor.last - tensor.first or not _equal_ranges(
                        artifact, record.offset, source, tensor.first,
                        record.logical_bytes,
                    ):
                        _fail(f"BF16 copy mismatch for {record.identity}")
            head = tensors["lm_head.weight"]
            embedding = tensors["model.embed_tokens.weight"]
            if not _equal_ranges(
                source, head.first, source, embedding.first,
                head.last - head.first,
            ):
                _fail("tied LM-head source differs from its embedding owner")
            samples = _verify_samples(source, artifact, tensors, records)
            source_digest = hashlib.sha256(source).hexdigest()
            if roots["source_artifact_root"] != source_digest:
                _fail("artifact source root does not match the supplied source")
            return {"schema": "pih.qwen3_int4_independent_reference.v1", **roots, "source_tensor_count": len(tensors), "artifact_record_count": len(records), "sampled_group_count": samples, "source_sha256": source_digest, "artifact_sha256": hashlib.sha256(artifact).hexdigest()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independently verify the canonical Qwen INT4 artifact against its BF16 Safetensors source")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        receipt = run(arguments.source.resolve(), arguments.artifact.resolve())
    except (OSError, ValueError, MemoryError) as error:
        print(str(error), file=sys.stderr); return 2
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
