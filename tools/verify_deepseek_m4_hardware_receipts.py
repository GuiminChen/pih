#!/usr/bin/env python3
"""Independently replay one non-authorizing DeepSeek M4 receipt generation."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
from typing import Sequence


_PROFILES = frozenset({"rtx4090d", "h100-pcie"})
_MAX_DEVICE_ORDINAL = 2_147_483_647
_PROFILE_GPU_IDENTITY = {
    "rtx4090d": (
        "NVIDIA GeForce RTX 4090 D", "8.9", 24_000_000_000, 30_000_000_000,
    ),
    "h100-pcie": (
        "NVIDIA H100 PCIe", "9.0", 80_000_000_000, 90_000_000_000,
    ),
}
_FULL_GPU_UUID = re.compile(
    r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}"
)
_MAX_RECEIPT_BYTES = 1 << 20
_VERIFICATION_SCHEMA = "pih.deepseek.m4.hardware_receipt_verification.v1"
_FIELDS = frozenset({
    "schema", "support_state", "hardware_profile", "devices", "world_size",
    "model_digest", "commit_sha", "repetitions", "started_ns", "finished_ns",
    "boundary_snapshot", "first_token_ids", "fault_injection_observed",
    "clean_teardown", "gpu_inventory", "driver_version", "cuda_version",
    "nccl_version", "evidence_root",
})
_SNAPSHOT_FIELDS = frozenset({
    "world_size", "expected_edge_count", "sealed_edge_count",
    "assembly_present", "generation_state",
})


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _digest(value: object, name: str, lengths: frozenset[int]) -> str:
    if (
        type(value) is not str or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be canonical lowercase hexadecimal")
    return value


def _nonempty_text(value: object, name: str) -> str:
    if type(value) is not str or not value or len(value) > 4096:
        raise ValueError(f"{name} must be bounded nonempty text")
    return value


def _verify_gpu_inventory_semantics(
    hardware_profile: str,
    devices: tuple[int, ...],
    inventory: tuple[str, ...],
) -> None:
    if (
        type(inventory) is not tuple
        or len(inventory) != len(devices)
        or any(type(row) is not str or not row or len(row) > 256 for row in inventory)
    ):
        raise ValueError("GPU inventory does not match the PP topology")
    name, compute_cap, minimum_bytes, maximum_bytes = _PROFILE_GPU_IDENTITY[
        hardware_profile
    ]
    unique_uuids: set[str] = set()
    for device, row in zip(devices, inventory, strict=True):
        columns = row.split(", ")
        if len(columns) != 5 or row != ", ".join(columns):
            raise ValueError("GPU inventory row schema is invalid")
        index_text, uuid, observed_name, observed_cap, memory_text = columns
        if (
            not index_text.isascii()
            or not index_text.isdecimal()
            or (len(index_text) > 1 and index_text.startswith("0"))
            or int(index_text) != device
        ):
            raise ValueError("GPU inventory device ordinal is invalid")
        if _FULL_GPU_UUID.fullmatch(uuid) is None or uuid in unique_uuids:
            raise ValueError("GPU inventory UUID is invalid or duplicated")
        unique_uuids.add(uuid)
        if observed_name != name or observed_cap != compute_cap:
            raise ValueError("GPU inventory device class differs from the profile")
        if (
            not memory_text.isascii()
            or not memory_text.isdecimal()
            or (len(memory_text) > 1 and memory_text.startswith("0"))
        ):
            raise ValueError("GPU inventory memory is invalid")
        memory_bytes = int(memory_text) * 1_048_576
        if not minimum_bytes <= memory_bytes < maximum_bytes:
            raise ValueError("GPU inventory memory class differs from the profile")


def build_verification_projection(
    *, hardware_profile: str, devices: tuple[int, ...], model_digest: str,
    commit_sha: str, repetitions: int, gpu_inventory: tuple[str, ...],
    driver_version: str, cuda_version: str, nccl_version: str,
    receipt_roots: tuple[str, ...],
) -> dict[str, object]:
    """Build the complete, closed M4 verification-root preimage."""
    if hardware_profile not in _PROFILES:
        raise ValueError("hardware_profile is outside the V1 target set")
    if (
        type(devices) is not tuple or not 1 <= len(devices) <= 4
        or any(
            type(device) is not int
            or device < 0
            or device > _MAX_DEVICE_ORDINAL
            for device in devices
        )
        or len(set(devices)) != len(devices)
    ):
        raise ValueError("devices must be a unique PP1-PP4 tuple")
    if type(repetitions) is not int or not 5 <= repetitions <= 100:
        raise ValueError("repetitions must be in [5, 100]")
    canonical_model_digest = _digest(
        model_digest, "model_digest", frozenset({64})
    )
    canonical_commit_sha = _digest(
        commit_sha, "commit_sha", frozenset({40, 64})
    )
    _verify_gpu_inventory_semantics(hardware_profile, devices, gpu_inventory)
    versions = {
        "driver_version": _nonempty_text(driver_version, "driver_version"),
        "cuda_version": _nonempty_text(cuda_version, "cuda_version"),
        "nccl_version": _nonempty_text(nccl_version, "nccl_version"),
    }
    if (
        type(receipt_roots) is not tuple
        or len(receipt_roots) != len(devices) * repetitions
    ):
        raise ValueError("receipt_roots must cover every PP repetition")
    canonical_receipt_roots = [
        _digest(root, "receipt_root", frozenset({64})) for root in receipt_roots
    ]
    if len(set(canonical_receipt_roots)) != len(canonical_receipt_roots):
        raise ValueError("receipt_roots must be unique")
    return {
        "schema": _VERIFICATION_SCHEMA,
        "hardware_profile": hardware_profile,
        "devices": list(devices),
        "world_sizes": list(range(1, len(devices) + 1)),
        "model_digest": canonical_model_digest,
        "commit_sha": canonical_commit_sha,
        "repetitions": repetitions,
        "gpu_inventory": list(gpu_inventory),
        **versions,
        "receipt_roots": canonical_receipt_roots,
        "support_state": "hardware_evidence_open",
    }


def _node_identity(info: os.stat_result, name: str) -> tuple[int, int, int]:
    if info.st_ino == 0:
        raise ValueError(f"receipt {name} identity is unavailable")
    return info.st_dev, info.st_ino, info.st_mode


def _file_identity_snapshot(info: os.stat_result, name: str) -> tuple[int, ...]:
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"receipt must be a regular non-symlink file: {name}")
    return (
        *_node_identity(info, name),
        info.st_size,
        info.st_mtime_ns,
    )


def _descriptor_snapshot(info: os.stat_result, name: str) -> tuple[int, ...]:
    return (
        *_file_identity_snapshot(info, name),
        info.st_ctime_ns,
    )


def _parent_identities(path: Path) -> tuple[tuple[Path, tuple[int, int, int]], ...]:
    identities: list[tuple[Path, tuple[int, int, int]]] = []
    junction_check = getattr(os.path, "isjunction", None)
    for parent in path.parents:
        try:
            info = os.lstat(parent)
        except OSError as error:
            raise ValueError(f"receipt parent path is unavailable: {path.name}") from error
        if (
            stat.S_ISLNK(info.st_mode)
            or (junction_check is not None and junction_check(parent))
        ):
            raise ValueError(f"receipt parent path contains a symlink: {path.name}")
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"receipt parent path is not a directory: {path.name}")
        identities.append((parent, _node_identity(info, parent.name or str(parent))))
    return tuple(identities)


def _bounded_fd_read(fd: int, name: str) -> bytes:
    chunks: list[bytes] = []
    remaining = _MAX_RECEIPT_BYTES + 1
    while remaining:
        try:
            chunk = os.read(fd, min(65_536, remaining))
        except InterruptedError:
            continue
        except (BlockingIOError, OSError) as error:
            raise ValueError(f"receipt cannot be read safely: {name}") from error
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    source = b"".join(chunks)
    if len(source) > _MAX_RECEIPT_BYTES:
        raise ValueError(f"receipt exceeds byte ceiling: {name}")
    return source


def _parse_receipt(path: Path) -> dict[str, object]:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("receipt path must be lexical absolute")
    parents_before = _parent_identities(path)
    try:
        lexical_before = os.lstat(path)
    except OSError as error:
        raise ValueError(f"receipt is unavailable: {path.name}") from error
    identity_before = _file_identity_snapshot(lexical_before, path.name)
    if lexical_before.st_size > _MAX_RECEIPT_BYTES:
        raise ValueError(f"receipt exceeds byte ceiling: {path.name}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOINHERIT", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"receipt cannot be opened safely: {path.name}") from error
    try:
        descriptor_before = os.fstat(fd)
        if _file_identity_snapshot(descriptor_before, path.name) != identity_before:
            raise ValueError(f"receipt identity changed before open: {path.name}")
        descriptor_snapshot = _descriptor_snapshot(descriptor_before, path.name)
        source = _bounded_fd_read(fd, path.name)
        descriptor_after = os.fstat(fd)
        if _descriptor_snapshot(descriptor_after, path.name) != descriptor_snapshot:
            raise ValueError(f"receipt changed during read: {path.name}")
        if len(source) != descriptor_after.st_size:
            raise ValueError(f"receipt changed during read: {path.name}")
    finally:
        os.close(fd)
    try:
        lexical_after = os.lstat(path)
    except OSError as error:
        raise ValueError(f"receipt identity changed after read: {path.name}") from error
    if _file_identity_snapshot(lexical_after, path.name) != identity_before:
        raise ValueError(f"receipt identity changed after read: {path.name}")
    if _parent_identities(path) != parents_before:
        raise ValueError(f"receipt parent identity changed during read: {path.name}")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"receipt contains duplicate field: {path.name}")
            value[key] = item
        return value

    try:
        value = json.loads(source.decode("ascii"), object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"receipt is not canonical ASCII JSON: {path.name}") from error
    if not isinstance(value, dict) or source != _canonical(value) + b"\n":
        raise ValueError(f"receipt is not canonical ASCII JSON: {path.name}")
    return value


def verify_receipts_projection(
    *, evidence_root: Path, expected_hardware_profile: str,
    expected_devices: tuple[int, ...], expected_model_digest: str,
    expected_commit_sha: str, expected_repetitions: int,
    expected_gpu_inventory: tuple[str, ...], expected_driver_version: str,
    expected_cuda_version: str, expected_nccl_version: str,
) -> dict[str, object]:
    if (
        not isinstance(evidence_root, Path) or not evidence_root.is_absolute()
        or evidence_root.is_symlink() or not evidence_root.is_dir()
    ):
        raise ValueError("evidence_root must be an absolute non-symlink directory")
    if expected_hardware_profile not in _PROFILES:
        raise ValueError("expected_hardware_profile is outside the V1 target set")
    if (
        type(expected_devices) is not tuple or not 1 <= len(expected_devices) <= 4
        or any(
            type(device) is not int
            or device < 0
            or device > _MAX_DEVICE_ORDINAL
            for device in expected_devices
        )
        or len(set(expected_devices)) != len(expected_devices)
    ):
        raise ValueError("expected_devices must be a unique PP1-PP4 tuple")
    if type(expected_repetitions) is not int or not 5 <= expected_repetitions <= 100:
        raise ValueError("expected_repetitions must be in [5, 100]")
    model_digest = _digest(expected_model_digest, "expected_model_digest", frozenset({64}))
    commit_sha = _digest(expected_commit_sha, "expected_commit_sha", frozenset({40, 64}))
    maximum_world_size = len(expected_devices)
    _verify_gpu_inventory_semantics(
        expected_hardware_profile, expected_devices, expected_gpu_inventory
    )
    versions = {
        "driver_version": _nonempty_text(expected_driver_version, "expected_driver_version"),
        "cuda_version": _nonempty_text(expected_cuda_version, "expected_cuda_version"),
        "nccl_version": _nonempty_text(expected_nccl_version, "expected_nccl_version"),
    }
    expected_names = tuple(
        f"{expected_hardware_profile}-pp{world_size}-run{ordinal:02d}.json"
        for world_size in range(1, maximum_world_size + 1)
        for ordinal in range(1, expected_repetitions + 1)
    )
    observed_names = frozenset(path.name for path in evidence_root.iterdir())
    if observed_names != frozenset(expected_names):
        raise ValueError("hardware receipt generation file set differs")

    receipt_roots: list[str] = []
    previous_finished = -1
    for index, name in enumerate(expected_names):
        world_size = index // expected_repetitions + 1
        expected_edges = world_size - 1
        value = _parse_receipt(evidence_root / name)
        if set(value) != _FIELDS:
            raise ValueError(f"hardware receipt fields differ: {name}")
        snapshot = value["boundary_snapshot"]
        if not isinstance(snapshot, dict) or set(snapshot) != _SNAPSHOT_FIELDS:
            raise ValueError(f"boundary snapshot fields differ: {name}")
        integer_axes = (
            value["world_size"], value["repetitions"], value["started_ns"],
            value["finished_ns"], snapshot["world_size"],
            snapshot["expected_edge_count"], snapshot["sealed_edge_count"],
        )
        if any(type(axis) is not int for axis in integer_axes):
            raise ValueError(f"hardware receipt integer axis is invalid: {name}")
        devices = value["devices"]
        if (
            not isinstance(devices, list)
            or any(type(device) is not int for device in devices)
        ):
            raise ValueError(f"hardware receipt device axis is invalid: {name}")
        receipt_inventory = value["gpu_inventory"]
        if not isinstance(receipt_inventory, list):
            raise ValueError(f"hardware receipt GPU inventory is invalid: {name}")
        try:
            _verify_gpu_inventory_semantics(
                expected_hardware_profile,
                expected_devices[:world_size],
                tuple(receipt_inventory),
            )
        except ValueError as error:
            raise ValueError(
                f"hardware receipt GPU inventory is invalid: {name}"
            ) from error
        started = value["started_ns"]
        finished = value["finished_ns"]
        assert isinstance(started, int) and isinstance(finished, int)
        if started < 0 or finished <= started or started < previous_finished:
            raise ValueError(f"hardware receipt time interval is invalid: {name}")
        previous_finished = finished
        expected_present = world_size > 1
        expected_state = "sealed" if expected_present else "not_applicable"
        if (
            value["schema"] != "pih.deepseek.nccl.hardware-evidence.v1"
            or value["support_state"] != "hardware_evidence_open"
            or value["hardware_profile"] != expected_hardware_profile
            or value["devices"] != list(expected_devices[:world_size])
            or value["world_size"] != world_size
            or value["model_digest"] != model_digest
            or value["commit_sha"] != commit_sha
            or value["repetitions"] != expected_repetitions
            or snapshot["world_size"] != world_size
            or snapshot["expected_edge_count"] != expected_edges
            or snapshot["sealed_edge_count"] != expected_edges
            or snapshot["assembly_present"] is not expected_present
            or snapshot["generation_state"] != expected_state
            or value["fault_injection_observed"] is not True
            or value["clean_teardown"] is not True
            or value["gpu_inventory"] != list(expected_gpu_inventory[:world_size])
            or any(value[key] != expected for key, expected in versions.items())
        ):
            raise ValueError(f"hardware receipt authority differs: {name}")
        tokens = value["first_token_ids"]
        if (
            not isinstance(tokens, list) or not tokens
            or any(type(token) is not int or not 0 <= token < 129_280 for token in tokens)
        ):
            raise ValueError(f"hardware receipt token evidence is invalid: {name}")
        evidence_root_value = _digest(
            value["evidence_root"], "evidence_root", frozenset({64})
        )
        payload = {key: item for key, item in value.items() if key != "evidence_root"}
        if sha256(_canonical(payload)).hexdigest() != evidence_root_value:
            raise ValueError(f"hardware receipt root does not replay: {name}")
        receipt_roots.append(evidence_root_value)

    return build_verification_projection(
        hardware_profile=expected_hardware_profile,
        devices=expected_devices,
        model_digest=model_digest,
        commit_sha=commit_sha,
        repetitions=expected_repetitions,
        gpu_inventory=expected_gpu_inventory,
        driver_version=versions["driver_version"],
        cuda_version=versions["cuda_version"],
        nccl_version=versions["nccl_version"],
        receipt_roots=tuple(receipt_roots),
    )


def verify_receipts(
    *, evidence_root: Path, expected_hardware_profile: str,
    expected_devices: tuple[int, ...], expected_model_digest: str,
    expected_commit_sha: str, expected_repetitions: int,
    expected_gpu_inventory: tuple[str, ...], expected_driver_version: str,
    expected_cuda_version: str, expected_nccl_version: str,
) -> str:
    projection = verify_receipts_projection(
        evidence_root=evidence_root,
        expected_hardware_profile=expected_hardware_profile,
        expected_devices=expected_devices,
        expected_model_digest=expected_model_digest,
        expected_commit_sha=expected_commit_sha,
        expected_repetitions=expected_repetitions,
        expected_gpu_inventory=expected_gpu_inventory,
        expected_driver_version=expected_driver_version,
        expected_cuda_version=expected_cuda_version,
        expected_nccl_version=expected_nccl_version,
    )
    return sha256(_canonical(projection)).hexdigest()


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--hardware-profile", required=True, choices=sorted(_PROFILES))
    parser.add_argument("--devices", required=True)
    parser.add_argument("--model-digest", required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--repetitions", required=True, type=int)
    parser.add_argument("--gpu-inventory", required=True, action="append")
    parser.add_argument("--driver-version", required=True)
    parser.add_argument("--cuda-version", required=True)
    parser.add_argument("--nccl-version", required=True)
    parsed = parser.parse_args(arguments)
    try:
        devices = tuple(int(value) for value in parsed.devices.split(","))
    except ValueError as error:
        raise SystemExit("--devices must be comma-separated integers") from error
    projection = verify_receipts_projection(
        evidence_root=parsed.evidence_root.absolute(),
        expected_hardware_profile=parsed.hardware_profile,
        expected_devices=devices, expected_model_digest=parsed.model_digest,
        expected_commit_sha=parsed.commit_sha,
        expected_repetitions=parsed.repetitions,
        expected_gpu_inventory=tuple(parsed.gpu_inventory),
        expected_driver_version=parsed.driver_version,
        expected_cuda_version=parsed.cuda_version,
        expected_nccl_version=parsed.nccl_version,
    )
    document = {
        **projection,
        "verification_root": sha256(_canonical(projection)).hexdigest(),
    }
    print(_canonical(document).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
