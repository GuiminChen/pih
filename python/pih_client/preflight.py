"""Client-side, non-authorizing readiness checks for an exact PIH GPU host.

This module is intentionally a deployment diagnostic, not a qualification
collector.  A successful receipt says only that the live host has the expected
Linux/SKU/topology namespace and reports its physical-memory baseline.  It
always retains ``hardware_evidence_open``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import subprocess
from types import MappingProxyType


TARGET_HOST_PREFLIGHT_ABI = "pih.target-host-preflight.v1"
_SUPPORT_STATE = "hardware_evidence_open"
_MIN_HOST_MEMORY_BYTES = 256 * (1 << 30)
_MAX_MEMINFO_BYTES = 1 << 20
_MAX_NVIDIA_SMI_BYTES = 64 << 10
_MAX_RECEIPT_DOCUMENT_BYTES = 128 << 10
_GPU_UUID = re.compile(
    r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}\Z"
)
_MEMTOTAL = re.compile(r"MemTotal:\s+([1-9][0-9]*) kB\Z")
_PROFILES = {
    "rtx4090d": (
        "NVIDIA GeForce RTX 4090 D", "8.9", 24_000_000_000, 30_000_000_000,
    ),
    "h100-pcie": (
        "NVIDIA H100 PCIe", "9.0", 80_000_000_000, 90_000_000_000,
    ),
    # NVIDIA Blackwell Ultra, compute capability 10.3, nominal 288 GB HBM3e.
    # This range tolerates driver-reserved memory and GB/GiB reporting; it is
    # an identity sanity check, not available-memory admission for a model.
    "b300": (
        "NVIDIA B300", "10.3", 280_000_000_000, 320_000_000_000,
    ),
}
_NVIDIA_SMI_QUERY = (
    "nvidia-smi",
    "--query-gpu=index,uuid,name,compute_cap,memory.total",
    "--format=csv,noheader,nounits",
)


@dataclass(frozen=True, slots=True)
class TargetHostPreflightReceipt:
    """A replayable, deliberately non-authorizing host diagnostic."""

    payload: Mapping[str, object]
    receipt_root: str

    def to_payload(self) -> dict[str, object]:
        return dict(self.payload)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _validate_ambient_namespace(environment: Mapping[str, str]) -> None:
    if "CUDA_VISIBLE_DEVICES" in environment:
        raise ValueError("CUDA_VISIBLE_DEVICES must be unset for target preflight")
    if environment.get("NVIDIA_VISIBLE_DEVICES", "all") != "all":
        raise ValueError(
            "NVIDIA_VISIBLE_DEVICES must be unset or exactly all for target preflight"
        )


def _validate_devices(devices: object, hardware_profile: str) -> tuple[int, ...]:
    maximum = 8 if hardware_profile == "b300" else 4
    if (
        type(devices) is not tuple
        or not 1 <= len(devices) <= maximum
        or any(type(device) is not int or not 0 <= device <= 2_147_483_647
               for device in devices)
        or len(set(devices)) != len(devices)
    ):
        raise ValueError(f"devices must be a unique exact 1-{maximum} device tuple")
    return devices


def _parse_memtotal(meminfo: object) -> int:
    if type(meminfo) is not str or not meminfo:
        raise ValueError("/proc/meminfo is invalid or exceeds its byte ceiling")
    try:
        encoded = meminfo.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("/proc/meminfo must be ASCII") from error
    if len(encoded) > _MAX_MEMINFO_BYTES:
        raise ValueError("/proc/meminfo is invalid or exceeds its byte ceiling")
    matches = [match for line in meminfo.splitlines()
               if (match := _MEMTOTAL.fullmatch(line)) is not None]
    if len(matches) != 1:
        raise ValueError("/proc/meminfo must contain one canonical MemTotal row")
    kibibytes = int(matches[0].group(1))
    if kibibytes > ((1 << 63) - 1) // 1024:
        raise ValueError("/proc/meminfo MemTotal overflows bytes")
    return kibibytes * 1024


def _parse_inventory(
    output: object, *, hardware_profile: str, devices: tuple[int, ...],
) -> tuple[str, ...]:
    if type(output) is not str:
        raise ValueError("nvidia-smi output must be text")
    try:
        encoded = output.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("nvidia-smi output must be canonical ASCII") from error
    if not output or len(encoded) > _MAX_NVIDIA_SMI_BYTES:
        raise ValueError("nvidia-smi output is empty or exceeds its byte ceiling")
    rows: dict[int, str] = {}
    for raw in output.splitlines():
        row = raw.strip()
        if not row:
            continue
        fields = row.split(", ")
        if len(fields) != 5 or row != ", ".join(fields):
            raise ValueError("nvidia-smi row schema is invalid")
        ordinal_text, uuid, _name, _capability, memory_text = fields
        if (
            not ordinal_text.isdecimal()
            or (len(ordinal_text) > 1 and ordinal_text.startswith("0"))
            or int(ordinal_text) > 2_147_483_647
            or _GPU_UUID.fullmatch(uuid) is None
            or not memory_text.isdecimal()
            or (len(memory_text) > 1 and memory_text.startswith("0"))
        ):
            raise ValueError("nvidia-smi row ordinal, UUID, or memory is invalid")
        ordinal = int(ordinal_text)
        if ordinal in rows:
            raise ValueError("nvidia-smi output contains a duplicate GPU ordinal")
        rows[ordinal] = row
    if any(device not in rows for device in devices):
        raise ValueError("nvidia-smi output omits a requested GPU ordinal")
    expected_name, expected_capability, minimum_bytes, maximum_bytes = _PROFILES[
        hardware_profile
    ]
    selected: list[str] = []
    selected_uuids: set[str] = set()
    for device in devices:
        row = rows[device]
        _ordinal, uuid, name, capability, memory_text = row.split(", ")
        memory_bytes = int(memory_text) * 1_048_576
        if uuid in selected_uuids:
            raise ValueError("nvidia-smi output duplicates a selected GPU UUID")
        selected_uuids.add(uuid)
        if name != expected_name or capability != expected_capability:
            raise ValueError("selected GPU SKU differs from the requested profile")
        if not minimum_bytes <= memory_bytes < maximum_bytes:
            raise ValueError("selected GPU memory class differs from the requested profile")
        selected.append(row)
    return tuple(selected)


def compile_target_host_preflight(
    *,
    hardware_profile: str,
    devices: tuple[int, ...],
    nvidia_smi_output: str,
    meminfo: str,
    system: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> TargetHostPreflightReceipt:
    """Compile a diagnostic receipt from bounded live-host observations."""

    observed_system = platform.system() if system is None else system
    if observed_system != "Linux":
        raise ValueError("target host preflight requires Linux")
    if type(hardware_profile) is not str or hardware_profile not in _PROFILES:
        raise ValueError("hardware_profile is outside the PIH V1 target set")
    selected_devices = _validate_devices(devices, hardware_profile)
    ambient = os.environ if environment is None else environment
    if not isinstance(ambient, Mapping) or any(
        type(key) is not str or type(value) is not str for key, value in ambient.items()
    ):
        raise ValueError("target preflight environment is invalid")
    _validate_ambient_namespace(ambient)
    inventory = _parse_inventory(
        nvidia_smi_output,
        hardware_profile=hardware_profile,
        devices=selected_devices,
    )
    host_memory_bytes = _parse_memtotal(meminfo)
    payload: dict[str, object] = {
        "abi": TARGET_HOST_PREFLIGHT_ABI,
        "devices": selected_devices,
        "gpu_inventory": inventory,
        "hardware_profile": hardware_profile,
        "host_memory_256_gib_sufficient": host_memory_bytes >= _MIN_HOST_MEMORY_BYTES,
        "host_memory_bytes": host_memory_bytes,
        "support_state": _SUPPORT_STATE,
        "world_size": len(selected_devices),
    }
    root = sha256(_canonical(payload)).hexdigest()
    return TargetHostPreflightReceipt(MappingProxyType(payload), root)


def verify_target_host_preflight(receipt: TargetHostPreflightReceipt) -> None:
    """Replay a persisted diagnostic before another deployment step uses it."""

    if not isinstance(receipt, TargetHostPreflightReceipt):
        raise TypeError("target host preflight receipt has an invalid type")
    if not isinstance(receipt.payload, Mapping):
        raise ValueError("target host preflight payload is invalid")
    payload = dict(receipt.payload)
    expected_keys = {
        "abi", "devices", "gpu_inventory", "hardware_profile",
        "host_memory_256_gib_sufficient", "host_memory_bytes",
        "support_state", "world_size",
    }
    if set(payload) != expected_keys or payload["abi"] != TARGET_HOST_PREFLIGHT_ABI:
        raise ValueError("target host preflight payload schema is invalid")
    profile = payload["hardware_profile"]
    if type(profile) is not str or profile not in _PROFILES:
        raise ValueError("target host preflight hardware_profile is invalid")
    devices = _validate_devices(payload["devices"], profile)
    inventory = payload["gpu_inventory"]
    if type(inventory) is not tuple or not all(type(row) is str for row in inventory):
        raise ValueError("target host preflight GPU inventory is invalid")
    if _parse_inventory(
        "\n".join(inventory), hardware_profile=profile, devices=devices,
    ) != inventory:
        raise ValueError("target host preflight GPU inventory does not replay")
    host_memory = payload["host_memory_bytes"]
    sufficient = payload["host_memory_256_gib_sufficient"]
    if (
        type(host_memory) is not int
        or not 0 < host_memory <= (1 << 63) - 1
        or type(sufficient) is not bool
        or sufficient is not (host_memory >= _MIN_HOST_MEMORY_BYTES)
        or type(payload["world_size"]) is not int
        or payload["world_size"] != len(devices)
        or payload["support_state"] != _SUPPORT_STATE
    ):
        raise ValueError("target host preflight payload values are invalid")
    if type(receipt.receipt_root) is not str or re.fullmatch(
        r"[0-9a-f]{64}", receipt.receipt_root,
    ) is None:
        raise ValueError("target host preflight receipt root is invalid")
    if sha256(_canonical(payload)).hexdigest() != receipt.receipt_root:
        raise ValueError("target host preflight receipt root does not replay")


def decode_target_host_preflight_document(
    document: object,
) -> TargetHostPreflightReceipt:
    """Decode the JSON-shaped preflight document emitted by :func:`main`."""

    if type(document) is not dict:
        raise ValueError("target host preflight document must be an object")
    expected_keys = {
        "abi", "devices", "gpu_inventory", "hardware_profile",
        "host_memory_256_gib_sufficient", "host_memory_bytes",
        "receipt_root", "support_state", "world_size",
    }
    if set(document) != expected_keys:
        raise ValueError("target host preflight document schema is invalid")
    devices = document["devices"]
    inventory = document["gpu_inventory"]
    if type(devices) is not list or type(inventory) is not list:
        raise ValueError("target host preflight document arrays are invalid")
    payload = {
        "abi": document["abi"],
        "devices": tuple(devices),
        "gpu_inventory": tuple(inventory),
        "hardware_profile": document["hardware_profile"],
        "host_memory_256_gib_sufficient": document[
            "host_memory_256_gib_sufficient"
        ],
        "host_memory_bytes": document["host_memory_bytes"],
        "support_state": document["support_state"],
        "world_size": document["world_size"],
    }
    receipt = TargetHostPreflightReceipt(
        MappingProxyType(payload), document["receipt_root"],
    )
    verify_target_host_preflight(receipt)
    return receipt


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("target host preflight document has a duplicate key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"target host preflight document constant is invalid: {value}")


def load_target_host_preflight_document(
    source: Path,
) -> TargetHostPreflightReceipt:
    """Load one bounded, strict JSON target-preflight receipt document."""

    if not isinstance(source, Path):
        raise TypeError("target host preflight document path is invalid")
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise ValueError("cannot read target host preflight document") from error
    if not raw or len(raw) > _MAX_RECEIPT_DOCUMENT_BYTES:
        raise ValueError("target host preflight document size is invalid")
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("target host preflight document JSON is invalid") from error
    return decode_target_host_preflight_document(document)


def _run_nvidia_smi(argv: tuple[str, ...]) -> str:
    try:
        completed = subprocess.run(
            argv, check=False, capture_output=True, text=False, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError("nvidia-smi target query failed") from error
    if completed.returncode != 0 or len(completed.stdout) > _MAX_NVIDIA_SMI_BYTES:
        raise ValueError("nvidia-smi target query failed")
    try:
        return completed.stdout.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("nvidia-smi output must be canonical ASCII") from error


def _read_meminfo() -> str:
    try:
        payload = Path("/proc/meminfo").read_bytes()
    except OSError as error:
        raise ValueError("cannot read /proc/meminfo") from error
    if not payload or len(payload) > _MAX_MEMINFO_BYTES:
        raise ValueError("/proc/meminfo is invalid or exceeds its byte ceiling")
    try:
        return payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("/proc/meminfo must be ASCII") from error


def run_target_host_preflight(
    *,
    hardware_profile: str,
    devices: tuple[int, ...],
    execute: Callable[[tuple[str, ...]], str] = _run_nvidia_smi,
    read_meminfo: Callable[[], str] = _read_meminfo,
    system: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> TargetHostPreflightReceipt:
    """Collect the two bounded host observations and compile their receipt."""

    if not callable(execute) or not callable(read_meminfo):
        raise TypeError("target preflight collectors must be callable")
    return compile_target_host_preflight(
        hardware_profile=hardware_profile,
        devices=devices,
        nvidia_smi_output=execute(_NVIDIA_SMI_QUERY),
        meminfo=read_meminfo(),
        system=system,
        environment=environment,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="diagnose an exact PIH GPU target host without authorizing support"
    )
    parser.add_argument("--hardware-profile", choices=sorted(_PROFILES))
    parser.add_argument("--devices")
    parser.add_argument("--verify-receipt", type=Path)
    namespace = parser.parse_args(arguments)
    try:
        if namespace.verify_receipt is not None:
            if namespace.hardware_profile is not None or namespace.devices is not None:
                raise ValueError(
                    "--verify-receipt cannot be combined with target collection options"
                )
            receipt = load_target_host_preflight_document(namespace.verify_receipt)
        else:
            if namespace.hardware_profile is None or namespace.devices is None:
                raise ValueError(
                    "--hardware-profile and --devices are required for target collection"
                )
            devices = tuple(int(value) for value in namespace.devices.split(","))
            receipt = run_target_host_preflight(
                hardware_profile=namespace.hardware_profile, devices=devices,
            )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(
        {**receipt.to_payload(), "receipt_root": receipt.receipt_root},
        ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ))
    return 0


__all__ = [
    "TARGET_HOST_PREFLIGHT_ABI",
    "TargetHostPreflightReceipt",
    "compile_target_host_preflight",
    "decode_target_host_preflight_document",
    "load_target_host_preflight_document",
    "run_target_host_preflight",
    "verify_target_host_preflight",
]


if __name__ == "__main__":
    raise SystemExit(main())
