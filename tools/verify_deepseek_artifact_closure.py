#!/usr/bin/env python3
"""Independently replay one non-authorizing DeepSeek checkpoint byte closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import NoReturn, Sequence


if os.name == "nt":  # pragma: no cover - definitions are exercised on Windows
    import ctypes
    from ctypes import wintypes
    import msvcrt

    class _WindowsFileBasicInfo(ctypes.Structure):
        _fields_ = [
            ("CreationTime", ctypes.c_longlong),
            ("LastAccessTime", ctypes.c_longlong),
            ("LastWriteTime", ctypes.c_longlong),
            ("ChangeTime", ctypes.c_longlong),
            ("FileAttributes", wintypes.DWORD),
        ]

    _WINDOWS_KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _WINDOWS_CREATE_FILE = _WINDOWS_KERNEL32.CreateFileW
    _WINDOWS_CREATE_FILE.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    _WINDOWS_CREATE_FILE.restype = wintypes.HANDLE
    _WINDOWS_GET_FILE_INFORMATION = (
        _WINDOWS_KERNEL32.GetFileInformationByHandleEx
    )
    _WINDOWS_GET_FILE_INFORMATION.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    _WINDOWS_GET_FILE_INFORMATION.restype = wintypes.BOOL
    _WINDOWS_CLOSE_HANDLE = _WINDOWS_KERNEL32.CloseHandle
    _WINDOWS_CLOSE_HANDLE.argtypes = (wintypes.HANDLE,)
    _WINDOWS_CLOSE_HANDLE.restype = wintypes.BOOL
    _WINDOWS_INVALID_HANDLE = wintypes.HANDLE(-1).value


_SCHEMA = "pih.deepseek_artifact_closure_verification.v1"
_ABI = "deepseek_artifact_closure_v1"
_MODEL_FAMILY = "deepseek_v4_flash_0731"
_VERIFICATION_SCOPE = "exact_checkpoint_bytes_non_authorizing"
_SUPPORT_STATE = "hardware_evidence_open"
_CONFIG_NAME = "config.json"
_INDEX_NAME = "model.safetensors.index.json"
_TENSOR_COUNT = 72_317
_SHARD_COUNT = 48
_DECLARED_TENSOR_BYTES = 166_878_536_440
_SHARDS = tuple(
    f"model-{index:05d}-of-00048.safetensors"
    for index in range(1, _SHARD_COUNT + 1)
)
_SHARD_SET = frozenset(_SHARDS)
_SHARD_PATTERN = re.compile(r"[A-Za-z0-9._-]+\.safetensors\Z")

_MAX_CONFIG_BYTES = 1 << 20
_MAX_INDEX_BYTES = 64 << 20
_MAX_SHARD_BYTES = 16 << 30
_MAX_TOTAL_BYTES = 256 << 30
_CONTROL_CHUNK_BYTES = 64 << 10
_SHARD_CHUNK_BYTES = 1 << 20
_MAX_JSON_DEPTH = 16
_MAX_JSON_NODES = 1_000_016
_MAX_JSON_STRING_BYTES = 128 << 20
_MAX_TENSOR_NAME_BYTES = 1024
_MAX_SHARD_NAME_BYTES = 255
_MAX_U32 = (1 << 32) - 1
_MAX_U64 = (1 << 64) - 1
_TYPE_U32 = 1
_TYPE_U64 = 2
_TYPE_BYTES = 3
_TYPE_HASH256 = 4


def _fail(message: str) -> NoReturn:
    raise ValueError(f"invalid DeepSeek artifact verification: {message}")


def _digest(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{name} must be canonical lowercase SHA-256")
    if value == "0" * 64:
        _fail(f"{name} must not use the reserved zero content digest")
    return value


def _integer_bytes(value: int, width: int, name: str) -> bytes:
    maximum = _MAX_U32 if width == 4 else _MAX_U64
    if type(value) is not int or not 0 <= value <= maximum:
        _fail(f"{name} is outside unsigned {width * 8}-bit range")
    return value.to_bytes(width, "little")


def _typed_root(
    domain: str,
    fields: Sequence[tuple[int, int, bytes]],
) -> str:
    try:
        domain_bytes = domain.encode("ascii", "strict")
    except UnicodeEncodeError as error:
        _fail(f"typed domain is not ASCII: {error}")
    if (
        not domain_bytes
        or len(domain_bytes) > 127
        or not domain.startswith("pih:")
        or any(byte < 0x20 or byte > 0x7E for byte in domain_bytes)
    ):
        _fail("typed domain is invalid")
    result = hashlib.sha256()
    result.update(domain_bytes)
    result.update(b"\0")
    result.update(_integer_bytes(len(fields), 4, "field count"))
    previous = 0
    for field_id, field_type, value in fields:
        if (
            type(field_id) is not int
            or not previous < field_id <= 0xFFFF
            or field_type not in {_TYPE_U32, _TYPE_U64, _TYPE_BYTES, _TYPE_HASH256}
            or type(value) is not bytes
        ):
            _fail("typed field is invalid or unordered")
        expected = {
            _TYPE_U32: 4,
            _TYPE_U64: 8,
            _TYPE_HASH256: 32,
        }.get(field_type)
        if expected is not None and len(value) != expected:
            _fail("typed fixed-width field has invalid length")
        result.update(field_id.to_bytes(2, "little"))
        result.update(bytes((field_type,)))
        result.update(_integer_bytes(len(value), 8, "field length"))
        result.update(value)
        previous = field_id
    return result.hexdigest()


def _object_root(role: str, name: str, length: int, media_sha256: str) -> str:
    try:
        role_bytes = role.encode("ascii", "strict")
        name_bytes = name.encode("ascii", "strict")
    except UnicodeEncodeError as error:
        _fail(f"object identity is not ASCII: {error}")
    return _typed_root(
        "pih:deepseek-artifact-object:v1",
        (
            (1, _TYPE_BYTES, role_bytes),
            (2, _TYPE_BYTES, name_bytes),
            (3, _TYPE_U64, _integer_bytes(length, 8, "object length")),
            (4, _TYPE_HASH256, bytes.fromhex(_digest(media_sha256, "media digest"))),
        ),
    )


def _strict_json(source: bytes, name: str) -> object:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                _fail(f"{name} contains duplicate JSON field {key!r}")
            result[key] = item
        return result

    def reject_constant(value: str) -> NoReturn:
        _fail(f"{name} contains non-finite JSON number {value!r}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            _fail(f"{name} contains non-finite JSON number {value!r}")
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
    string_bytes = 0
    pending: list[tuple[object, int]] = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
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
                string_bytes += len(item.encode("utf-8", "strict"))
            except UnicodeEncodeError as error:
                _fail(f"{name} contains invalid Unicode: {error}")
            if string_bytes > _MAX_JSON_STRING_BYTES:
                _fail(f"{name} exceeds JSON string bounds")
            continue
        if type(item) is list:
            pending.extend((child, depth + 1) for child in item)
            continue
        if type(item) is dict:
            for key, child in item.items():
                try:
                    string_bytes += len(key.encode("utf-8", "strict"))
                except UnicodeEncodeError as error:
                    _fail(f"{name} contains invalid JSON key Unicode: {error}")
                if string_bytes > _MAX_JSON_STRING_BYTES:
                    _fail(f"{name} exceeds JSON string bounds")
                pending.append((child, depth + 1))
            continue
        _fail(f"{name} contains a non-JSON value")
    return value


def _node(info: object, name: str) -> tuple[int, int, int]:
    inode = int(getattr(info, "st_ino"))
    if inode == 0:
        _fail(f"filesystem identity is unavailable for {name}")
    return int(getattr(info, "st_dev")), inode, int(getattr(info, "st_mode"))


def _directory_identity(info: object, name: str) -> tuple[object, ...]:
    mode = int(getattr(info, "st_mode"))
    if not stat.S_ISDIR(mode):
        _fail(f"artifact root component is not a directory: {name}")
    node = _node(info, name)
    if os.name == "nt":
        return node
    return (
        *node,
        int(getattr(info, "st_mtime_ns")),
        getattr(info, "st_ctime_ns", None),
    )


def _file_identity(info: object, name: str) -> tuple[int, ...]:
    mode = int(getattr(info, "st_mode"))
    if not stat.S_ISREG(mode):
        _fail(f"artifact member must be a regular non-symlink file: {name}")
    links = int(getattr(info, "st_nlink"))
    if links != 1:
        _fail(f"artifact member has an external hard-link alias: {name}")
    return (
        *_node(info, name),
        int(getattr(info, "st_size")),
        int(getattr(info, "st_mtime_ns")),
        links,
    )


def _windows_handle_change_time(handle: int, name: str) -> int:
    basic = _WindowsFileBasicInfo()
    if not _WINDOWS_GET_FILE_INFORMATION(
        handle,
        0,
        ctypes.byref(basic),
        ctypes.sizeof(basic),
    ):
        error = ctypes.get_last_error()
        _fail(f"Win32 ChangeTime is unavailable for {name}: error {error}")
    return int(basic.ChangeTime)


def _path_change_token(path: Path, info: object, name: str) -> int:
    if os.name != "nt":
        return int(getattr(info, "st_ctime_ns"))
    handle = _WINDOWS_CREATE_FILE(
        str(path),
        0x0080,
        0x0001 | 0x0002 | 0x0004,
        None,
        3,
        0x02000000 | 0x00200000,
        None,
    )
    if handle in (None, _WINDOWS_INVALID_HANDLE):
        error = ctypes.get_last_error()
        _fail(f"artifact member ChangeTime cannot be opened: {name}: error {error}")
    try:
        return _windows_handle_change_time(handle, name)
    finally:
        if not _WINDOWS_CLOSE_HANDLE(handle):
            error = ctypes.get_last_error()
            _fail(f"artifact member ChangeTime handle cannot be closed: {name}: error {error}")


def _descriptor_change_token(descriptor: int, info: object, name: str) -> int:
    if os.name != "nt":
        return int(getattr(info, "st_ctime_ns"))
    try:
        handle = msvcrt.get_osfhandle(descriptor)
    except OSError as error:
        _fail(f"artifact member descriptor handle is unavailable: {name}: {error}")
    return _windows_handle_change_time(handle, name)


def _lexical_file_identity(
    path: Path,
    info: object,
    name: str,
) -> tuple[object, ...]:
    return (*_file_identity(info, name), _path_change_token(path, info, name))


def _descriptor_identity(
    descriptor: int,
    info: object,
    name: str,
) -> tuple[object, ...]:
    return (
        *_file_identity(info, name),
        _descriptor_change_token(descriptor, info, name),
    )


def _open_stable_member(path: Path, flags: int, name: str) -> int:
    if os.name != "nt":
        try:
            return os.open(path, flags)
        except OSError as error:
            _fail(f"artifact member cannot be opened safely: {name}: {error}")
    handle = _WINDOWS_CREATE_FILE(
        str(path),
        0x80000000,
        0x0001,
        None,
        3,
        0x08000000 | 0x00200000,
        None,
    )
    if handle in (None, _WINDOWS_INVALID_HANDLE):
        error = ctypes.get_last_error()
        _fail(f"artifact member cannot be opened safely: {name}: error {error}")
    try:
        return msvcrt.open_osfhandle(
            handle,
            os.O_RDONLY | getattr(os, "O_BINARY", 0),
        )
    except OSError as error:
        if not _WINDOWS_CLOSE_HANDLE(handle):
            close_error = ctypes.get_last_error()
            _fail(
                f"artifact member native handle conversion and close failed: "
                f"{name}: {error}; close error {close_error}"
            )
        _fail(f"artifact member native handle cannot become a descriptor: {name}: {error}")


def _junction(path: Path) -> bool:
    function = getattr(os.path, "isjunction", None)
    if function is not None:
        return bool(function(path))
    method = getattr(path, "is_junction", None)
    return bool(method()) if method is not None else False


def _reject_link(path: Path, info: object, name: str) -> None:
    if stat.S_ISLNK(int(getattr(info, "st_mode"))) or _junction(path):
        _fail(f"artifact path contains a symlink or junction: {name}")


def _parents(path: Path) -> tuple[tuple[Path, tuple[object, ...]], ...]:
    result: list[tuple[Path, tuple[object, ...]]] = []
    for parent in path.parents:
        try:
            info = os.lstat(parent)
        except OSError as error:
            _fail(f"artifact parent path is unavailable: {error}")
        _reject_link(parent, info, parent.name or str(parent))
        if not stat.S_ISDIR(info.st_mode):
            _fail(f"artifact parent path is not a directory: {parent}")
        result.append((parent, _directory_identity(info, parent.name or str(parent))))
    return tuple(result)


def _root_identity(
    root: Path,
) -> tuple[tuple[object, ...], tuple[tuple[Path, tuple[object, ...]], ...]]:
    try:
        info = os.lstat(root)
    except OSError as error:
        _fail(f"artifact root is unavailable: {error}")
    _reject_link(root, info, root.name or str(root))
    return _directory_identity(info, root.name or str(root)), _parents(root)


def _reject_sparse(info: object, name: str) -> None:
    size = int(getattr(info, "st_size"))
    blocks = getattr(info, "st_blocks", None)
    if size and blocks is not None and int(blocks) * 512 < size:
        _fail(f"artifact member must not be sparse: {name}")
    attributes = int(getattr(info, "st_file_attributes", 0))
    sparse = int(getattr(stat, "FILE_ATTRIBUTE_SPARSE_FILE", 0))
    if sparse and attributes & sparse:
        _fail(f"artifact member must not be sparse: {name}")


_MemberLease = tuple[Path, int, tuple[object, ...]]


def _revalidate_leases(leases: Sequence[_MemberLease]) -> None:
    for path, descriptor, identity in leases:
        try:
            descriptor_info = os.fstat(descriptor)
        except OSError as error:
            _fail(f"artifact member lease is unavailable: {path.name}: {error}")
        if _descriptor_identity(descriptor, descriptor_info, path.name) != identity:
            _fail(f"artifact member changed after its read: {path.name}")
        _reject_sparse(descriptor_info, path.name)
        try:
            lexical_info = os.lstat(path)
        except OSError as error:
            _fail(f"artifact member path changed after its read: {path.name}: {error}")
        _reject_link(path, lexical_info, path.name)
        if _lexical_file_identity(path, lexical_info, path.name) != identity:
            _fail(f"artifact member path changed after its read: {path.name}")


def _close_leases(leases: Sequence[_MemberLease]) -> None:
    first_error: OSError | None = None
    for _path, descriptor, _identity in reversed(leases):
        try:
            os.close(descriptor)
        except OSError as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        _fail(f"artifact member lease cannot be closed: {first_error}")


def _read_member(
    path: Path,
    *,
    maximum_bytes: int,
    materialize: bool,
    chunk_bytes: int,
    leases: list[_MemberLease] | None = None,
) -> tuple[int, str, tuple[int, int], bytes | None]:
    parents_before = _parents(path)
    try:
        lexical_before = os.lstat(path)
    except OSError as error:
        _fail(f"artifact member is unavailable: {path.name}: {error}")
    _reject_link(path, lexical_before, path.name)
    lexical_snapshot = _lexical_file_identity(path, lexical_before, path.name)
    _reject_sparse(lexical_before, path.name)
    size = lexical_before.st_size
    if not 0 < size <= maximum_bytes:
        _fail(f"artifact member exceeds its byte bound: {path.name}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOINHERIT", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = _open_stable_member(path, flags, path.name)
    retained = False
    try:
        digest = hashlib.sha256()
        chunks: list[bytes] | None = [] if materialize else None
        total = 0
        descriptor_before = os.fstat(descriptor)
        descriptor_snapshot = _descriptor_identity(
            descriptor, descriptor_before, path.name
        )
        if descriptor_snapshot != lexical_snapshot:
            _fail(f"artifact member identity changed before open: {path.name}")
        _reject_sparse(descriptor_before, path.name)
        while total < size:
            requested = min(chunk_bytes, size - total)
            try:
                chunk = os.read(descriptor, requested)
            except InterruptedError:
                continue
            except (BlockingIOError, OSError) as error:
                _fail(f"artifact member cannot be read safely: {path.name}: {error}")
            if not chunk or len(chunk) > requested:
                _fail(f"artifact member changed or ended during read: {path.name}")
            total += len(chunk)
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        descriptor_after = os.fstat(descriptor)
        if (
            _descriptor_identity(descriptor, descriptor_after, path.name)
            != descriptor_snapshot
            or total != size
        ):
            _fail(f"artifact member identity changed during read: {path.name}")
        try:
            lexical_after = os.lstat(path)
        except OSError as error:
            _fail(
                f"artifact member identity changed after read: {path.name}: {error}"
            )
        _reject_link(path, lexical_after, path.name)
        if _lexical_file_identity(path, lexical_after, path.name) != lexical_snapshot:
            _fail(f"artifact member identity changed after read: {path.name}")
        if _parents(path) != parents_before:
            _fail(f"artifact parent identity changed during read: {path.name}")
        result = (
            total,
            digest.hexdigest(),
            (lexical_before.st_dev, lexical_before.st_ino),
            b"".join(chunks) if chunks is not None else None,
        )
        if leases is not None:
            leases.append((path, descriptor, descriptor_snapshot))
            retained = True
        return result
    finally:
        if not retained:
            os.close(descriptor)


def _preflight_member(path: Path, maximum_bytes: int) -> tuple[int, tuple[int, int]]:
    try:
        info = os.lstat(path)
    except OSError as error:
        _fail(f"artifact member is unavailable during preflight: {path.name}: {error}")
    _reject_link(path, info, path.name)
    _file_identity(info, path.name)
    _reject_sparse(info, path.name)
    if not 0 < info.st_size <= maximum_bytes:
        _fail(f"artifact member exceeds its byte bound: {path.name}")
    return info.st_size, (info.st_dev, info.st_ino)


def _parse_index(source: bytes) -> tuple[int, tuple[str, ...]]:
    value = _strict_json(source, "weight index")
    if type(value) is not dict or set(value) != {"metadata", "weight_map"}:
        _fail("weight index root fields are invalid")
    metadata = value["metadata"]
    weight_map = value["weight_map"]
    if type(metadata) is not dict or set(metadata) != {"total_size"}:
        _fail("weight index metadata fields are invalid")
    total_size = metadata["total_size"]
    if type(total_size) is not int or total_size != _DECLARED_TENSOR_BYTES:
        _fail("weight index total_size is outside official geometry")
    if type(weight_map) is not dict or len(weight_map) != _TENSOR_COUNT:
        _fail("weight index tensor count is outside official geometry")
    names: set[str] = set()
    for tensor_name, shard_name in weight_map.items():
        try:
            tensor_bytes = tensor_name.encode("utf-8", "strict")
        except UnicodeEncodeError as error:
            _fail(f"tensor name is invalid UTF-8: {error}")
        if not tensor_bytes or len(tensor_bytes) > _MAX_TENSOR_NAME_BYTES:
            _fail("tensor name exceeds its byte bound")
        if type(shard_name) is not str:
            _fail("shard path must be a string basename")
        try:
            shard_bytes = shard_name.encode("ascii", "strict")
        except UnicodeEncodeError as error:
            _fail(f"shard path is not ASCII: {error}")
        if (
            not shard_bytes
            or len(shard_bytes) > _MAX_SHARD_NAME_BYTES
            or _SHARD_PATTERN.fullmatch(shard_name) is None
            or Path(shard_name).name != shard_name
            or "/" in shard_name
            or "\\" in shard_name
            or ".." in shard_name
        ):
            _fail("shard path must be a canonical basename")
        names.add(shard_name)
    if names != _SHARD_SET:
        _fail("weight index shard set is outside official geometry")
    return total_size, tuple(sorted(names))


def _listed_shards(root: Path) -> frozenset[str]:
    try:
        with os.scandir(root) as entries:
            return frozenset(
                entry.name for entry in entries if entry.name.endswith(".safetensors")
            )
    except OSError as error:
        _fail(f"Safetensors file set cannot be enumerated: {error}")


def _closure_root(
    config_root: str,
    index_root: str,
    shard_roots: Sequence[str],
    total_file_bytes: int,
) -> str:
    fields: list[tuple[int, int, bytes]] = [
        (1, _TYPE_BYTES, _ABI.encode("ascii")),
        (2, _TYPE_BYTES, _MODEL_FAMILY.encode("ascii")),
        (3, _TYPE_HASH256, bytes.fromhex(config_root)),
        (4, _TYPE_HASH256, bytes.fromhex(index_root)),
        (5, _TYPE_U32, _integer_bytes(len(shard_roots), 4, "shard count")),
        (6, _TYPE_U64, _integer_bytes(_TENSOR_COUNT, 8, "tensor count")),
        (
            7,
            _TYPE_U64,
            _integer_bytes(_DECLARED_TENSOR_BYTES, 8, "declared tensor bytes"),
        ),
        (8, _TYPE_U64, _integer_bytes(total_file_bytes, 8, "total file bytes")),
    ]
    fields.extend(
        (100 + index, _TYPE_HASH256, bytes.fromhex(root))
        for index, root in enumerate(shard_roots)
    )
    return _typed_root("pih:deepseek-artifact-closure:v1", fields)


def _verify_artifact_closure(
    *,
    model_root: Path,
    expected_model_digest: str,
    leases: list[_MemberLease],
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    expected = _digest(expected_model_digest, "expected model digest")
    if details is not None and (type(details) is not dict or details):
        _fail("artifact identity details sink must be an empty concrete dict")
    if not isinstance(model_root, Path) or not model_root.is_absolute():
        _fail("model root must be a lexical absolute Path")
    if any(part in {".", ".."} for part in model_root.parts):
        _fail("model root must not contain dot path components")
    root_before = _root_identity(model_root)
    inodes: set[tuple[int, int]] = set()

    def admit(inode: tuple[int, int], name: str) -> None:
        if inode in inodes:
            _fail(f"artifact members contain a hard-link identity alias: {name}")
        inodes.add(inode)

    config_length, config_digest, config_inode, config_source = _read_member(
        model_root / _CONFIG_NAME,
        maximum_bytes=_MAX_CONFIG_BYTES,
        materialize=True,
        chunk_bytes=_CONTROL_CHUNK_BYTES,
        leases=leases,
    )
    admit(config_inode, _CONFIG_NAME)
    assert config_source is not None
    config_value = _strict_json(config_source, "model config")
    if type(config_value) is not dict or config_value.get("model_type") != "deepseek_v4":
        _fail("model config does not identify deepseek_v4")
    config_root = _object_root(
        "model_config", _CONFIG_NAME, config_length, config_digest
    )

    index_length, index_digest, index_inode, index_source = _read_member(
        model_root / _INDEX_NAME,
        maximum_bytes=_MAX_INDEX_BYTES,
        materialize=True,
        chunk_bytes=_CONTROL_CHUNK_BYTES,
        leases=leases,
    )
    admit(index_inode, _INDEX_NAME)
    assert index_source is not None
    declared_tensor_bytes, names = _parse_index(index_source)
    index_root = _object_root(
        "weight_index", _INDEX_NAME, index_length, index_digest
    )

    if _listed_shards(model_root) != _SHARD_SET:
        _fail("Safetensors shard set is missing or contains unreferenced files")
    preflight_total = config_length + index_length
    preflight_inodes = set(inodes)
    for name in names:
        length, inode = _preflight_member(model_root / name, _MAX_SHARD_BYTES)
        if inode in preflight_inodes:
            _fail(f"artifact members contain a hard-link identity alias: {name}")
        preflight_inodes.add(inode)
        if preflight_total > _MAX_TOTAL_BYTES - length:
            _fail("artifact aggregate bytes exceed 256GiB")
        preflight_total += length
    total_file_bytes = config_length + index_length
    shard_roots: list[str] = []
    shard_objects: list[tuple[str, int, str]] = []
    for name in names:
        length, media_digest, inode, source = _read_member(
            model_root / name,
            maximum_bytes=_MAX_SHARD_BYTES,
            materialize=False,
            chunk_bytes=_SHARD_CHUNK_BYTES,
            leases=leases,
        )
        assert source is None
        admit(inode, name)
        if total_file_bytes > _MAX_TOTAL_BYTES - length:
            _fail("artifact aggregate bytes exceed 256GiB")
        total_file_bytes += length
        object_root = _object_root("weight_shard", name, length, media_digest)
        shard_roots.append(object_root)
        shard_objects.append((name, length, object_root))
    if _listed_shards(model_root) != _SHARD_SET:
        _fail("Safetensors shard set changed during verification")
    _revalidate_leases(leases)
    if _root_identity(model_root) != root_before:
        _fail("artifact root or parent identity changed during verification")

    observed = _closure_root(
        config_root,
        index_root,
        shard_roots,
        total_file_bytes,
    )
    if observed != expected:
        _fail("expected model digest differs from independently replayed digest")
    if details is not None:
        details.update(
            {
                "index_source": index_source,
                "index_object_root": index_root,
                "shard_objects": tuple(shard_objects),
            }
        )
    return {
        "schema": _SCHEMA,
        "abi": _ABI,
        "model_family": _MODEL_FAMILY,
        "model_digest": observed,
        "object_count": 2 + len(shard_roots),
        "shard_count": len(shard_roots),
        "tensor_count": _TENSOR_COUNT,
        "declared_tensor_bytes": declared_tensor_bytes,
        "total_file_bytes": total_file_bytes,
        "verification_scope": _VERIFICATION_SCOPE,
        "support_state": _SUPPORT_STATE,
    }


def verify_artifact_closure(
    *,
    model_root: Path,
    expected_model_digest: str,
) -> dict[str, object]:
    leases: list[_MemberLease] = []
    try:
        return _verify_artifact_closure(
            model_root=model_root,
            expected_model_digest=expected_model_digest,
            leases=leases,
        )
    finally:
        _close_leases(leases)


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
        description="Independently verify a DeepSeek checkpoint byte closure."
    )
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--expected-model-digest", required=True)
    arguments = parser.parse_args(argv)
    result = verify_artifact_closure(
        model_root=Path(arguments.model_root),
        expected_model_digest=arguments.expected_model_digest,
    )
    sys.stdout.buffer.write(_canonical(result).encode("ascii") + b"\n")
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
