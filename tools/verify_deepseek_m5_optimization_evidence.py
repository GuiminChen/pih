#!/usr/bin/env python3
"""Independently replay one non-authorizing DeepSeek M5 v2 receipt."""

from __future__ import annotations

import argparse
import base64
import binascii
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import struct
from typing import Sequence


_OPTIMIZATIONS = frozenset(
    {
        "chunked_prefill",
        "attention_physical_layout",
        "fused_kernels",
        "observed_miss_debit",
        "dspark",
    }
)
_ENVIRONMENT_FIELDS = frozenset(
    {"hardware", "driver", "cuda", "nccl", "os", "commit", "process"}
)
_POLICY_FIELDS = frozenset(
    {
        "architecture",
        "world_size",
        "chunked_prefill",
        "attention_layout",
        "fused_kernel_set",
        "observed_miss_debit",
        "dspark",
        "cuda_graph",
        "expert_locality_reorder",
    }
)
_POLICY_AXIS = {
    "chunked_prefill": "chunked_prefill",
    "attention_physical_layout": "attention_layout",
    "fused_kernels": "fused_kernel_set",
    "observed_miss_debit": "observed_miss_debit",
    "dspark": "dspark",
}
_POLICY_ENABLEMENT = {
    "chunked_prefill": (False, True),
    "attention_physical_layout": ("baseline", "v1"),
    "fused_kernels": ("unfused_control", "verified_v1"),
    "observed_miss_debit": (False, True),
    "dspark": (False, True),
}
_PROFILE_GPU_IDENTITY = {
    "rtx4090d": (
        "NVIDIA GeForce RTX 4090 D",
        "8.9",
        24_000_000_000,
        30_000_000_000,
    ),
    "h100-pcie": (
        "NVIDIA H100 PCIe",
        "9.0",
        80_000_000_000,
        90_000_000_000,
    ),
}
_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "capture_scope",
        "analysis_abi",
        "support_state",
        "plan",
        "plan_root",
        "executions",
        "derived_metrics",
        "correctness_state",
        "performance_state",
        "baseline_state",
        "evidence_root",
    }
)
_PLAN_FIELDS = frozenset(
    {
        "schema",
        "capture_scope",
        "analysis_abi",
        "support_state",
        "repetitions",
        "workload_instance_roots",
        "run_order",
        "control",
        "candidate",
        "baselines",
        "environment",
        "hardware",
        "m4_verification",
        "m4_verification_root",
        "runner_root",
        "working_directory",
        "per_command_timeout_seconds",
    }
)
_RUN_FIELDS = frozenset(
    {
        "role",
        "optimization",
        "model_revision",
        "model_digest",
        "profile_root",
        "input_distribution_root",
        "concurrency",
        "policy",
        "policy_root",
        "command_template",
        "command_files",
    }
)
_M4_VERIFICATION_FIELDS = frozenset(
    {
        "schema",
        "hardware_profile",
        "devices",
        "world_sizes",
        "model_digest",
        "commit_sha",
        "repetitions",
        "gpu_inventory",
        "driver_version",
        "cuda_version",
        "nccl_version",
        "receipt_roots",
        "support_state",
        "verification_root",
    }
)
_M4_PROJECTION_FIELDS = _M4_VERIFICATION_FIELDS - {"verification_root"}
_BASELINE_FIELDS = frozenset(
    {"name", "version", "command", "command_files"}
)
_CAPTURE_FIELDS = frozenset(
    {
        "execution_ordinal",
        "role",
        "pair_ordinal",
        "order_slot",
        "argv",
        "command_files_before",
        "command_files_after",
        "started_monotonic_ns",
        "finished_monotonic_ns",
        "exit_code",
        "stdout_bytes",
        "stdout_sha256",
        "stderr_bytes",
        "stderr_sha256",
        "stderr_base64",
        "gpu_pre",
        "gpu_post",
        "observation",
        "observation_root",
    }
)
_OBSERVATION_FIELDS = frozenset(
    {
        "schema",
        "invocation_id",
        "role",
        "repetition",
        "workload_instance_root",
        "clock_abi",
        "requests",
        "gpu_bytes_samples",
        "host_bytes_samples",
    }
)
_REQUEST_FIELDS = frozenset(
    {
        "ordinal",
        "input_root",
        "output_root",
        "outcome",
        "prompt_tokens",
        "output_tokens",
        "prefill_ns",
        "decode_ns",
        "ttft_ns",
        "itl_ns",
    }
)
_MAX_DEVICE_ORDINAL = 2_147_483_647
_MAX_REQUESTS = 100_000
_MAX_INTERVALS = 1_000_000
_MAX_STDOUT_BYTES = 16 << 20
_MAX_STDERR_BYTES = 16 << 20
_MAX_FILE_BYTES = 2 << 30
_MAX_RECEIPT_BYTES = 16 << 20
_MAX_U64 = (1 << 64) - 1


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _typed_json_root(domain: str, value: object) -> str:
    if (
        type(domain) is not str
        or not domain.startswith("pih:")
        or not 1 <= len(domain) <= 127
        or any(ord(character) < 0x20 or ord(character) > 0x7E for character in domain)
    ):
        raise ValueError("typed JSON domain is invalid")
    encoded = _canonical(value)
    return sha256(
        domain.encode("ascii") + b"\0" + struct.pack("<Q", len(encoded)) + encoded
    ).hexdigest()


def _fields(value: object, expected: frozenset[str], name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{name} fields differ")
    return value


def _hex(
    value: object,
    name: str,
    lengths: frozenset[int] = frozenset({64}),
) -> str:
    if (
        type(value) is not str
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be canonical lowercase hexadecimal")
    return value


def _text(value: object, name: str, maximum: int = 4096) -> str:
    if (
        type(value) is not str
        or not value
        or "\0" in value
        or len(value.encode("utf-8")) > maximum
    ):
        raise ValueError(f"{name} must be bounded nonempty text")
    return value


def _decimal(value: object, name: str, *, positive: bool = False) -> int:
    if (
        type(value) is not str
        or not value.isascii()
        or not value.isdecimal()
        or (len(value) > 1 and value.startswith("0"))
    ):
        raise ValueError(f"{name} must be a canonical uint64 decimal string")
    parsed = int(value)
    if parsed > _MAX_U64 or (positive and parsed == 0):
        raise ValueError(f"{name} is outside its uint64 bound")
    return parsed


def _u32(value: object, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or not 0 <= value <= 0xFFFFFFFF:
        raise ValueError(f"{name} must be an exact uint32")
    if positive and value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _command(value: object, name: str) -> list[str]:
    if (
        type(value) is not list
        or not 1 <= len(value) <= 256
        or any(
            type(item) is not str
            or not item
            or "\0" in item
            or len(item.encode("utf-8")) > 4096
            for item in value
        )
        or sum(len(item.encode("utf-8")) for item in value) > 65_536
    ):
        raise ValueError(f"{name} must be a bounded nonempty string array")
    return list(value)


def _descriptor(value: object, name: str) -> dict[str, object]:
    descriptor = _fields(
        value, frozenset({"path", "bytes", "sha256"}), f"{name} descriptor"
    )
    path = _text(descriptor["path"], f"{name} path", 16_384)
    if not Path(path).is_absolute():
        raise ValueError(f"{name} path must be absolute")
    size = descriptor["bytes"]
    if type(size) is not int or not 0 < size <= _MAX_FILE_BYTES:
        raise ValueError(f"{name} byte length is invalid")
    return {
        "path": path,
        "bytes": size,
        "sha256": _hex(descriptor["sha256"], f"{name} sha256"),
    }


def _descriptors(value: object, name: str) -> list[dict[str, object]]:
    if type(value) is not list or not 1 <= len(value) <= 64:
        raise ValueError(f"{name} must contain one to 64 file descriptors")
    result = [
        _descriptor(item, f"{name}[{index}]") for index, item in enumerate(value)
    ]
    paths = [str(item["path"]) for item in result]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{name} contains duplicate paths")
    return result


def _policy_root(values: Sequence[int]) -> str:
    hasher = sha256()
    hasher.update(b"pih:deepseek-optimization-policy:v1")
    hasher.update(b"\0")
    hasher.update(struct.pack("<I", 9))
    for field_id, value in enumerate(values, 1):
        hasher.update(struct.pack("<H", field_id))
        hasher.update(b"\x01")
        hasher.update(struct.pack("<Q", 4))
        hasher.update(struct.pack("<I", value))
    return hasher.hexdigest()


def _policy(value: object) -> tuple[dict[str, object], str]:
    policy = _fields(value, _POLICY_FIELDS, "optimization policy projection")
    architecture = policy["architecture"]
    world_size = policy["world_size"]
    chunked = policy["chunked_prefill"]
    layout = policy["attention_layout"]
    fused = policy["fused_kernel_set"]
    miss_debit = policy["observed_miss_debit"]
    dspark = policy["dspark"]
    graph = policy["cuda_graph"]
    locality = policy["expert_locality_reorder"]
    if architecture not in {"sm89", "sm90"}:
        raise ValueError("optimization architecture is outside SM89/SM90")
    if type(world_size) is not int or not 1 <= world_size <= 4:
        raise ValueError("optimization world_size is outside PP1-PP4")
    if layout not in {"baseline", "v1"}:
        raise ValueError("optimization attention layout is unknown")
    if fused not in {"unfused_control", "verified_v1"}:
        raise ValueError("optimization fused kernel set is unknown")
    for name, item in (
        ("chunked_prefill", chunked),
        ("observed_miss_debit", miss_debit),
        ("dspark", dspark),
        ("cuda_graph", graph),
        ("expert_locality_reorder", locality),
    ):
        if type(item) is not bool:
            raise ValueError(f"optimization policy {name} must be bool")
    if graph or locality:
        raise ValueError("DeepSeek V1 forbids CUDA Graph and locality reordering")
    if architecture == "sm89" and dspark:
        raise ValueError("DeepSeek SM89 policy forbids DSpark")
    normalized = {
        "architecture": architecture,
        "world_size": world_size,
        "chunked_prefill": chunked,
        "attention_layout": layout,
        "fused_kernel_set": fused,
        "observed_miss_debit": miss_debit,
        "dspark": dspark,
        "cuda_graph": graph,
        "expert_locality_reorder": locality,
    }
    values = (
        89 if architecture == "sm89" else 90,
        world_size,
        int(chunked),
        0 if layout == "baseline" else 1,
        0 if fused == "unfused_control" else 1,
        int(miss_debit),
        int(dspark),
        int(graph),
        int(locality),
    )
    return normalized, _policy_root(values)


def _run_spec(value: object, expected_role: str) -> dict[str, object]:
    spec = _fields(value, _RUN_FIELDS, f"{expected_role} run spec")
    if spec["role"] != expected_role:
        raise ValueError("paired run roles are invalid")
    optimization = spec["optimization"]
    if optimization not in _OPTIMIZATIONS:
        raise ValueError("optimization is outside the closed M5 set")
    concurrency = spec["concurrency"]
    if type(concurrency) is not int or not 1 <= concurrency <= 65_536:
        raise ValueError("concurrency is outside its bound")
    policy, policy_root = _policy(spec["policy"])
    if _hex(spec["policy_root"], f"{expected_role} policy_root") != policy_root:
        raise ValueError(f"{expected_role} policy_root does not replay")
    command = _command(spec["command_template"], f"{expected_role} command template")
    if command.count("{repetition}") != 1 or command.count(
        "{workload_instance_root}"
    ) != 1:
        raise ValueError("run command template placeholders differ")
    command_files = _descriptors(
        spec["command_files"], f"{expected_role} command files"
    )
    if command[0] != command_files[0]["path"]:
        raise ValueError("run executable differs from its first command file")
    return {
        "role": expected_role,
        "optimization": optimization,
        "model_revision": _hex(
            spec["model_revision"], "model_revision", frozenset({40, 64})
        ),
        "model_digest": _hex(spec["model_digest"], "model_digest"),
        "profile_root": _hex(spec["profile_root"], "profile_root"),
        "input_distribution_root": _hex(
            spec["input_distribution_root"], "input_distribution_root"
        ),
        "concurrency": concurrency,
        "policy": policy,
        "policy_root": policy_root,
        "command_template": command,
        "command_files": command_files,
    }


def _baseline(value: object) -> dict[str, object]:
    spec = _fields(value, _BASELINE_FIELDS, "baseline spec")
    if spec["name"] not in {"vllm", "sglang"}:
        raise ValueError("baseline name is outside the closed set")
    command = _command(spec["command"], "baseline command")
    if "{repetition}" in command or "{workload_instance_root}" in command:
        raise ValueError("baseline command cannot contain run placeholders")
    command_files = _descriptors(spec["command_files"], "baseline command files")
    if command[0] != command_files[0]["path"]:
        raise ValueError("baseline executable differs from its first command file")
    return {
        "name": spec["name"],
        "version": _text(spec["version"], "baseline version", 128),
        "command": command,
        "command_files": command_files,
    }


def _environment(value: object) -> dict[str, object]:
    environment = _fields(value, _ENVIRONMENT_FIELDS, "optimization environment")
    normalized: dict[str, object] = {}
    for key in sorted(_ENVIRONMENT_FIELDS - {"process"}):
        normalized[key] = _text(environment[key], f"environment {key}")
    _hex(normalized["commit"], "environment commit", frozenset({40, 64}))
    process = environment["process"]
    if type(process) is not dict or len(process) > 128:
        raise ValueError("process environment must be a bounded object")
    normalized_process: dict[str, str] = {}
    total = 0
    for key, item in process.items():
        if (
            type(key) is not str
            or not key
            or "=" in key
            or "\0" in key
            or len(key.encode("utf-8")) > 256
            or type(item) is not str
            or "\0" in item
            or len(item.encode("utf-8")) > 4096
        ):
            raise ValueError("process environment entry is invalid")
        total += len(key.encode("utf-8")) + len(item.encode("utf-8"))
        normalized_process[key] = item
    if total > 65_536:
        raise ValueError("process environment exceeds its byte ceiling")
    normalized["process"] = {
        key: normalized_process[key] for key in sorted(normalized_process)
    }
    return normalized


def _hardware(value: object) -> dict[str, object]:
    hardware = _fields(
        value,
        frozenset({"hardware_profile", "devices", "gpu_inventory"}),
        "optimization hardware",
    )
    profile = hardware["hardware_profile"]
    if profile not in _PROFILE_GPU_IDENTITY:
        raise ValueError("hardware profile is outside the V1 target set")
    devices = hardware["devices"]
    if (
        type(devices) is not list
        or not 1 <= len(devices) <= 4
        or any(
            type(device) is not int
            or not 0 <= device <= _MAX_DEVICE_ORDINAL
            for device in devices
        )
        or len(devices) != len(set(devices))
    ):
        raise ValueError("hardware devices must be a unique PP1-PP4 array")
    inventory = hardware["gpu_inventory"]
    if type(inventory) is not list or len(inventory) != len(devices):
        raise ValueError("GPU inventory does not match the PP topology")
    expected_name, expected_cap, minimum_bytes, maximum_bytes = (
        _PROFILE_GPU_IDENTITY[profile]
    )
    uuids: set[str] = set()
    normalized_inventory: list[str] = []
    for device, row in zip(devices, inventory, strict=True):
        if type(row) is not str or not row or len(row) > 256:
            raise ValueError("GPU inventory row is invalid")
        columns = row.split(", ")
        if len(columns) != 5 or row != ", ".join(columns):
            raise ValueError("GPU inventory row schema is invalid")
        ordinal, uuid, name, compute_cap, memory = columns
        if (
            not ordinal.isascii()
            or not ordinal.isdecimal()
            or (len(ordinal) > 1 and ordinal.startswith("0"))
            or int(ordinal) != device
        ):
            raise ValueError("GPU inventory ordinal is invalid")
        uuid_parts = uuid.split("-")
        if (
            len(uuid_parts) != 6
            or uuid_parts[0] != "GPU"
            or tuple(len(part) for part in uuid_parts[1:]) != (8, 4, 4, 4, 12)
            or any(
                character not in "0123456789abcdef"
                for part in uuid_parts[1:]
                for character in part
            )
            or uuid in uuids
        ):
            raise ValueError("GPU inventory UUID is invalid or duplicated")
        uuids.add(uuid)
        if name != expected_name or compute_cap != expected_cap:
            raise ValueError("GPU inventory class differs from the profile")
        if (
            not memory.isascii()
            or not memory.isdecimal()
            or (len(memory) > 1 and memory.startswith("0"))
        ):
            raise ValueError("GPU inventory memory is invalid")
        memory_bytes = int(memory) * 1_048_576
        if not minimum_bytes <= memory_bytes < maximum_bytes:
            raise ValueError("GPU inventory memory class differs from the profile")
        normalized_inventory.append(row)
    return {
        "hardware_profile": profile,
        "devices": list(devices),
        "gpu_inventory": normalized_inventory,
    }


def _m4_verification(value: object) -> dict[str, object]:
    verification = _fields(
        value, _M4_VERIFICATION_FIELDS, "M4 verification projection"
    )
    if (
        verification["schema"]
        != "pih.deepseek.m4.hardware_receipt_verification.v1"
        or verification["support_state"] != "hardware_evidence_open"
    ):
        raise ValueError("M4 verification projection authority differs")
    hardware = _hardware(
        {
            "hardware_profile": verification["hardware_profile"],
            "devices": verification["devices"],
            "gpu_inventory": verification["gpu_inventory"],
        }
    )
    devices = hardware["devices"]
    assert isinstance(devices, list)
    world_sizes = verification["world_sizes"]
    expected_world_sizes = list(range(1, len(devices) + 1))
    if type(world_sizes) is not list or world_sizes != expected_world_sizes:
        raise ValueError("M4 verification world_sizes do not replay from devices")
    repetitions = verification["repetitions"]
    if type(repetitions) is not int or not 5 <= repetitions <= 100:
        raise ValueError("M4 verification repetitions must be in [5, 100]")
    roots = verification["receipt_roots"]
    expected_root_count = repetitions * len(expected_world_sizes)
    if type(roots) is not list or len(roots) != expected_root_count:
        raise ValueError("M4 verification receipt roots differ from its generation axes")
    receipt_roots = [
        _hex(item, f"M4 receipt_roots[{index}]")
        for index, item in enumerate(roots)
    ]
    if len(receipt_roots) != len(set(receipt_roots)):
        raise ValueError("M4 verification receipt roots must be unique")
    projection: dict[str, object] = {
        "schema": "pih.deepseek.m4.hardware_receipt_verification.v1",
        "hardware_profile": hardware["hardware_profile"],
        "devices": devices,
        "world_sizes": expected_world_sizes,
        "model_digest": _hex(
            verification["model_digest"], "M4 verification model_digest"
        ),
        "commit_sha": _hex(
            verification["commit_sha"],
            "M4 verification commit_sha",
            frozenset({40, 64}),
        ),
        "repetitions": repetitions,
        "gpu_inventory": hardware["gpu_inventory"],
        "driver_version": _text(
            verification["driver_version"], "M4 verification driver_version"
        ),
        "cuda_version": _text(
            verification["cuda_version"], "M4 verification cuda_version"
        ),
        "nccl_version": _text(
            verification["nccl_version"], "M4 verification nccl_version"
        ),
        "receipt_roots": receipt_roots,
        "support_state": "hardware_evidence_open",
    }
    if set(projection) != _M4_PROJECTION_FIELDS:
        raise AssertionError("internal M4 verification projection schema drift")
    verification_root = _hex(
        verification["verification_root"], "M4 verification_root"
    )
    if sha256(_canonical(projection)).hexdigest() != verification_root:
        raise ValueError("M4 verification root does not replay from its projection")
    normalized = {**projection, "verification_root": verification_root}
    if _canonical(verification) != _canonical(normalized):
        raise ValueError("M4 verification projection does not normalize exactly")
    return normalized


def _pair_policy(control: dict[str, object], candidate: dict[str, object]) -> None:
    for axis in (
        "optimization",
        "model_revision",
        "model_digest",
        "profile_root",
        "input_distribution_root",
        "concurrency",
    ):
        if control[axis] != candidate[axis]:
            raise ValueError("paired run axes differ")
    control_policy = control["policy"]
    candidate_policy = candidate["policy"]
    assert isinstance(control_policy, dict) and isinstance(candidate_policy, dict)
    selected = _POLICY_AXIS[str(control["optimization"])]
    differences = {
        key
        for key in _POLICY_FIELDS
        if control_policy[key] != candidate_policy[key]
    }
    if differences != {selected}:
        raise ValueError("paired policies must differ only on the selected optimization")
    expected_control, expected_candidate = _POLICY_ENABLEMENT[
        str(control["optimization"])
    ]
    if (
        control_policy[selected] != expected_control
        or candidate_policy[selected] != expected_candidate
    ):
        raise ValueError("selected optimization enablement direction is invalid")


def _plan(
    value: object,
    *,
    receipt_plan_root: object,
    expected_plan_root: str,
    expected_m4_verification_root: str,
    expected_commit_sha: str,
    expected_runner_root: str,
) -> tuple[dict[str, object], str]:
    plan = _fields(value, _PLAN_FIELDS, "capture plan")
    if (
        plan["schema"] != "pih.deepseek.optimization-capture-plan.v2"
        or plan["capture_scope"] != "non_authorizing_raw_capture"
        or plan["analysis_abi"] != "paired_integer_replay_v1"
        or plan["support_state"] != "hardware_evidence_open"
    ):
        raise ValueError("capture plan is not the required M5 v2 authority")
    repetitions = plan["repetitions"]
    if type(repetitions) is not int or not 5 <= repetitions <= 100:
        raise ValueError("capture plan repetitions must be in [5, 100]")
    workloads = plan["workload_instance_roots"]
    if type(workloads) is not list or len(workloads) != repetitions:
        raise ValueError("workload roots differ from the repetition count")
    normalized_workloads = [
        _hex(item, f"workload_instance_roots[{index}]")
        for index, item in enumerate(workloads)
    ]
    if len(normalized_workloads) != len(set(normalized_workloads)):
        raise ValueError("workload instance roots must be unique")
    expected_order = [
        ["control", "candidate"] if ordinal % 2 else ["candidate", "control"]
        for ordinal in range(1, repetitions + 1)
    ]
    if plan["run_order"] != expected_order:
        raise ValueError("capture plan run order differs from the frozen schedule")
    control = _run_spec(plan["control"], "control")
    candidate = _run_spec(plan["candidate"], "candidate")
    _pair_policy(control, candidate)
    baselines = plan["baselines"]
    if type(baselines) is not list or len(baselines) != 2:
        raise ValueError("capture plan requires exactly two baselines")
    normalized_baselines = [_baseline(item) for item in baselines]
    if [item["name"] for item in normalized_baselines] != ["sglang", "vllm"]:
        raise ValueError("capture plan baselines must be canonical sglang/vllm order")
    environment = _environment(plan["environment"])
    hardware = _hardware(plan["hardware"])
    m4_verification = _m4_verification(plan["m4_verification"])
    control_policy = control["policy"]
    assert isinstance(control_policy, dict)
    expected_profile = (
        "rtx4090d" if control_policy["architecture"] == "sm89" else "h100-pcie"
    )
    if hardware["hardware_profile"] != expected_profile:
        raise ValueError("policy architecture and hardware profile differ")
    devices = hardware["devices"]
    assert isinstance(devices, list)
    if control_policy["world_size"] != len(devices):
        raise ValueError("policy world_size and hardware topology differ")
    process = environment["process"]
    assert isinstance(process, dict)
    if "CUDA_VISIBLE_DEVICES" in process:
        raise ValueError(
            "process CUDA_VISIBLE_DEVICES must remain unset for physical ordinals"
        )
    if process.get("NVIDIA_VISIBLE_DEVICES", "all") != "all":
        raise ValueError("process NVIDIA_VISIBLE_DEVICES must remain unset or all")
    m4_root = _hex(plan["m4_verification_root"], "m4_verification_root")
    if m4_verification["verification_root"] != m4_root:
        raise ValueError("M4 verification projection and plan root differ")
    m4_joins = (
        (
            "hardware profile",
            m4_verification["hardware_profile"],
            hardware["hardware_profile"],
        ),
        ("devices", m4_verification["devices"], hardware["devices"]),
        (
            "GPU inventory",
            m4_verification["gpu_inventory"],
            hardware["gpu_inventory"],
        ),
        (
            "model digest",
            m4_verification["model_digest"],
            control["model_digest"],
        ),
        ("commit", m4_verification["commit_sha"], environment["commit"]),
        ("driver", m4_verification["driver_version"], environment["driver"]),
        ("CUDA", m4_verification["cuda_version"], environment["cuda"]),
        ("NCCL", m4_verification["nccl_version"], environment["nccl"]),
        (
            "world sizes",
            m4_verification["world_sizes"],
            list(range(1, len(devices) + 1)),
        ),
    )
    for axis, observed, expected in m4_joins:
        if observed != expected:
            raise ValueError(f"M4 verification {axis} differs from capture plan")
    runner_root = _hex(plan["runner_root"], "runner_root")
    working_directory = _text(
        plan["working_directory"], "working_directory", 16_384
    )
    if not Path(working_directory).is_absolute():
        raise ValueError("working_directory must be absolute")
    timeout = plan["per_command_timeout_seconds"]
    if (
        type(timeout) is not int
        or not 1 <= timeout <= 7200
        or timeout * (2 * repetitions + 2) > 86_400
    ):
        raise ValueError("per-command timeout exceeds the 24-hour suite envelope")
    normalized: dict[str, object] = {
        "schema": "pih.deepseek.optimization-capture-plan.v2",
        "capture_scope": "non_authorizing_raw_capture",
        "analysis_abi": "paired_integer_replay_v1",
        "support_state": "hardware_evidence_open",
        "repetitions": repetitions,
        "workload_instance_roots": normalized_workloads,
        "run_order": expected_order,
        "control": control,
        "candidate": candidate,
        "baselines": normalized_baselines,
        "environment": environment,
        "hardware": hardware,
        "m4_verification": m4_verification,
        "m4_verification_root": m4_root,
        "runner_root": runner_root,
        "working_directory": working_directory,
        "per_command_timeout_seconds": timeout,
    }
    if _canonical(plan) != _canonical(normalized):
        raise ValueError("capture plan does not normalize exactly")
    observed_plan_root = _typed_json_root(
        "pih:deepseek-m5-capture-plan:v2", normalized
    )
    if _hex(receipt_plan_root, "receipt plan_root") != observed_plan_root:
        raise ValueError("capture plan root does not replay")
    if observed_plan_root != expected_plan_root:
        raise ValueError("capture plan differs from expected plan root")
    if m4_root != expected_m4_verification_root:
        raise ValueError("capture plan differs from expected M4 verification root")
    if runner_root != expected_runner_root:
        raise ValueError("capture plan differs from expected runner root")
    if environment["commit"] != expected_commit_sha:
        raise ValueError("capture plan differs from expected commit SHA")
    return normalized, observed_plan_root


def _schedule(plan: dict[str, object]) -> list[dict[str, object]]:
    schedule: list[dict[str, object]] = []
    ordinal = 1
    order = plan["run_order"]
    assert isinstance(order, list)
    for pair_ordinal, roles in enumerate(order, 1):
        assert isinstance(roles, list)
        for order_slot, role in enumerate(roles):
            schedule.append(
                {
                    "execution_ordinal": ordinal,
                    "role": role,
                    "pair_ordinal": pair_ordinal,
                    "order_slot": order_slot,
                }
            )
            ordinal += 1
    for role in ("vllm", "sglang"):
        schedule.append(
            {
                "execution_ordinal": ordinal,
                "role": role,
                "pair_ordinal": 0,
                "order_slot": 0,
            }
        )
        ordinal += 1
    return schedule


def _invocation_id(
    plan_root: str,
    *,
    execution_ordinal: int,
    role: str,
    pair_ordinal: int,
    order_slot: int,
) -> str:
    return _typed_json_root(
        "pih:deepseek-m5-invocation:v2",
        {
            "plan_root": plan_root,
            "execution_ordinal": _u32(
                execution_ordinal, "execution_ordinal", positive=True
            ),
            "role": _text(role, "execution role", 32),
            "pair_ordinal": _u32(pair_ordinal, "pair_ordinal"),
            "order_slot": _u32(order_slot, "order_slot"),
        },
    )


def _request(value: object, index: int) -> dict[str, object]:
    request = _fields(value, _REQUEST_FIELDS, "raw request observation")
    if _u32(request["ordinal"], "request ordinal") != index:
        raise ValueError("raw request ordinals must be contiguous and ordered")
    prompt_tokens = _u32(
        request["prompt_tokens"], "prompt_tokens", positive=True
    )
    output_tokens = _u32(request["output_tokens"], "output_tokens")
    input_root = _hex(request["input_root"], "request input_root")
    outcome = request["outcome"]
    if outcome not in {"completed", "timeout", "rejected", "error"}:
        raise ValueError("raw request outcome is unknown")
    intervals = request["itl_ns"]
    if type(intervals) is not list:
        raise ValueError("raw request ITL samples must be an array")
    if outcome == "completed":
        if output_tokens == 0:
            raise ValueError("completed request must contain output tokens")
        output_root: object = _hex(request["output_root"], "request output_root")
        prefill: object = str(
            _decimal(request["prefill_ns"], "prefill_ns", positive=True)
        )
        decode: object = str(
            _decimal(request["decode_ns"], "decode_ns", positive=True)
        )
        ttft: object = str(_decimal(request["ttft_ns"], "ttft_ns", positive=True))
        if len(intervals) != max(0, output_tokens - 1):
            raise ValueError("raw request ITL count differs from output tokens")
        normalized_itl = [
            str(_decimal(item, "itl_ns", positive=True)) for item in intervals
        ]
    else:
        if (
            request["output_root"] is not None
            or output_tokens != 0
            or request["prefill_ns"] is not None
            or request["decode_ns"] is not None
            or request["ttft_ns"] is not None
            or intervals
        ):
            raise ValueError("failed request must retain the canonical empty result")
        output_root = None
        prefill = None
        decode = None
        ttft = None
        normalized_itl = []
    return {
        "ordinal": index,
        "input_root": input_root,
        "output_root": output_root,
        "outcome": outcome,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "prefill_ns": prefill,
        "decode_ns": decode,
        "ttft_ns": ttft,
        "itl_ns": normalized_itl,
    }


def _observation(
    value: object,
    *,
    expected_schema: str,
    expected_role: str,
    expected_repetition: int,
    expected_invocation_id: str,
    expected_workload_root: str,
    expected_baseline_version: str | None,
) -> dict[str, object]:
    expected_fields = _OBSERVATION_FIELDS
    if expected_baseline_version is not None:
        expected_fields |= frozenset({"baseline_version"})
    observation = _fields(value, expected_fields, "raw observation")
    if (
        observation["schema"] != expected_schema
        or observation["role"] != expected_role
        or observation["invocation_id"] != expected_invocation_id
        or observation["workload_instance_root"] != expected_workload_root
        or observation["clock_abi"] != "monotonic_ns_v1"
        or type(observation["repetition"]) is not int
        or observation["repetition"] != expected_repetition
        or (
            expected_baseline_version is not None
            and observation["baseline_version"] != expected_baseline_version
        )
    ):
        raise ValueError("raw observation identity differs from the capture plan")
    requests = observation["requests"]
    if type(requests) is not list or not 1 <= len(requests) <= _MAX_REQUESTS:
        raise ValueError("raw observation request count is invalid")
    normalized_requests = [
        _request(item, index) for index, item in enumerate(requests)
    ]
    if sum(len(item["itl_ns"]) for item in normalized_requests) > _MAX_INTERVALS:
        raise ValueError("raw observation ITL sample count exceeds its bound")
    normalized: dict[str, object] = {
        "schema": expected_schema,
        "invocation_id": expected_invocation_id,
        "role": expected_role,
        "repetition": expected_repetition,
        "workload_instance_root": _hex(
            observation["workload_instance_root"], "workload_instance_root"
        ),
        "clock_abi": "monotonic_ns_v1",
        "requests": normalized_requests,
    }
    for key in ("gpu_bytes_samples", "host_bytes_samples"):
        samples = observation[key]
        if type(samples) is not list or not 1 <= len(samples) <= 1_000_000:
            raise ValueError(f"raw observation {key} count is invalid")
        normalized[key] = [str(_decimal(item, key)) for item in samples]
    if expected_baseline_version is not None:
        normalized["baseline_version"] = _text(
            observation["baseline_version"], "baseline version", 128
        )
    if _canonical(observation) != _canonical(normalized):
        raise ValueError("raw observation does not normalize exactly")
    return normalized


def _gpu_sample(value: object, inventory: list[str]) -> list[str]:
    if type(value) is not list or len(value) != len(inventory):
        raise ValueError("live GPU sample does not match the topology")
    normalized: list[str] = []
    for row, expected_identity in zip(value, inventory, strict=True):
        if type(row) is not str or not row or len(row) > 320:
            raise ValueError("live GPU sample row is invalid")
        columns = row.split(", ")
        if len(columns) != 6 or row != ", ".join(columns):
            raise ValueError("live GPU sample row schema is invalid")
        if ", ".join(columns[:5]) != expected_identity:
            raise ValueError("live GPU identity drifted from the capture plan")
        used = columns[5]
        if (
            not used.isascii()
            or not used.isdecimal()
            or (len(used) > 1 and used.startswith("0"))
        ):
            raise ValueError("live GPU memory sample is invalid")
        if int(used) > int(columns[4]):
            raise ValueError("live GPU used memory exceeds total memory")
        normalized.append(row)
    return normalized


def _capture(
    value: object,
    *,
    plan: dict[str, object],
    plan_root: str,
    scheduled: dict[str, object],
) -> dict[str, object]:
    capture = _fields(value, _CAPTURE_FIELDS, "command capture")
    for key in ("execution_ordinal", "pair_ordinal", "order_slot"):
        if type(capture[key]) is not int or capture[key] != scheduled[key]:
            raise ValueError("command capture schedule differs")
    if type(capture["role"]) is not str or capture["role"] != scheduled["role"]:
        raise ValueError("command capture schedule differs")
    role = str(scheduled["role"])
    pair_ordinal = int(scheduled["pair_ordinal"])
    order_slot = int(scheduled["order_slot"])
    execution_ordinal = int(scheduled["execution_ordinal"])
    expected_invocation = _invocation_id(
        plan_root,
        execution_ordinal=execution_ordinal,
        role=role,
        pair_ordinal=pair_ordinal,
        order_slot=order_slot,
    )
    workloads = plan["workload_instance_roots"]
    assert isinstance(workloads, list)
    if role in {"control", "candidate"}:
        spec = plan[role]
        assert isinstance(spec, dict)
        workload_root = str(workloads[pair_ordinal - 1])
        template = spec["command_template"]
        assert isinstance(template, list)
        expected_argv = [
            str(pair_ordinal)
            if item == "{repetition}"
            else workload_root
            if item == "{workload_instance_root}"
            else item
            for item in template
        ]
        expected_files = spec["command_files"]
        expected_schema = "pih.deepseek.m5.raw-run-observation.v2"
        baseline_version = None
    else:
        baselines = plan["baselines"]
        assert isinstance(baselines, list)
        spec = next(item for item in baselines if item["name"] == role)
        workload_root = str(workloads[0])
        expected_argv = spec["command"]
        expected_files = spec["command_files"]
        expected_schema = "pih.deepseek.m5.raw-baseline-observation.v2"
        baseline_version = str(spec["version"])
    if _command(capture["argv"], "captured argv") != expected_argv:
        raise ValueError("captured argv differs from the capture plan")
    before = _descriptors(capture["command_files_before"], "pre-execution files")
    after = _descriptors(capture["command_files_after"], "post-execution files")
    if before != expected_files or after != expected_files:
        raise ValueError("command file identity drifted during execution")
    started = _decimal(capture["started_monotonic_ns"], "started_monotonic_ns")
    finished = _decimal(
        capture["finished_monotonic_ns"], "finished_monotonic_ns", positive=True
    )
    if finished <= started:
        raise ValueError("command capture interval is invalid")
    if type(capture["exit_code"]) is not int or capture["exit_code"] != 0:
        raise ValueError("command capture has a nonzero or inexact exit code")
    observation = _observation(
        capture["observation"],
        expected_schema=expected_schema,
        expected_role=role,
        expected_repetition=pair_ordinal,
        expected_invocation_id=expected_invocation,
        expected_workload_root=workload_root,
        expected_baseline_version=baseline_version,
    )
    observation_root = _typed_json_root(
        "pih:deepseek-m5-raw-observation:v2", observation
    )
    if _hex(capture["observation_root"], "observation root") != observation_root:
        raise ValueError("raw observation root does not replay")
    if role in {"vllm", "sglang"}:
        baseline_requests = observation["requests"]
        assert isinstance(baseline_requests, list)
        if len(baseline_requests) != 1 or baseline_requests[0]["outcome"] != "completed":
            raise ValueError("baseline probe must observe one completed request")
    stdout = _canonical(observation) + b"\n"
    stdout_bytes = capture["stdout_bytes"]
    if (
        type(stdout_bytes) is not int
        or stdout_bytes != len(stdout)
        or not 0 < len(stdout) <= _MAX_STDOUT_BYTES
        or _hex(capture["stdout_sha256"], "stdout sha256")
        != sha256(stdout).hexdigest()
    ):
        raise ValueError("captured stdout descriptor does not replay")
    encoded_stderr = capture["stderr_base64"]
    if type(encoded_stderr) is not str or not encoded_stderr.isascii():
        raise ValueError("captured stderr must use canonical base64")
    try:
        stderr = base64.b64decode(encoded_stderr, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("captured stderr is not canonical base64") from error
    if base64.b64encode(stderr).decode("ascii") != encoded_stderr:
        raise ValueError("captured stderr is not canonical base64")
    stderr_bytes = capture["stderr_bytes"]
    if (
        type(stderr_bytes) is not int
        or stderr_bytes != len(stderr)
        or len(stderr) > _MAX_STDERR_BYTES
        or _hex(capture["stderr_sha256"], "stderr sha256")
        != sha256(stderr).hexdigest()
    ):
        raise ValueError("captured stderr descriptor does not replay")
    hardware = plan["hardware"]
    assert isinstance(hardware, dict)
    inventory = hardware["gpu_inventory"]
    assert isinstance(inventory, list)
    gpu_pre = _gpu_sample(capture["gpu_pre"], inventory)
    gpu_post = _gpu_sample(capture["gpu_post"], inventory)
    if gpu_pre != gpu_post:
        raise ValueError("GPU memory baseline was not restored after the command")
    return {
        "execution_ordinal": execution_ordinal,
        "role": role,
        "pair_ordinal": pair_ordinal,
        "order_slot": order_slot,
        "argv": expected_argv,
        "command_files_before": before,
        "command_files_after": after,
        "started_monotonic_ns": str(started),
        "finished_monotonic_ns": str(finished),
        "exit_code": 0,
        "stdout_bytes": len(stdout),
        "stdout_sha256": sha256(stdout).hexdigest(),
        "stderr_bytes": len(stderr),
        "stderr_sha256": sha256(stderr).hexdigest(),
        "stderr_base64": encoded_stderr,
        "gpu_pre": gpu_pre,
        "gpu_post": gpu_post,
        "observation": observation,
        "observation_root": observation_root,
    }


def _metrics(observation: dict[str, object]) -> dict[str, object]:
    requests = observation["requests"]
    assert isinstance(requests, list)
    completed = [item for item in requests if item["outcome"] == "completed"]
    prefill_tokens = sum(int(item["prompt_tokens"]) for item in completed)
    decode_tokens = sum(int(item["output_tokens"]) for item in completed)
    prefill_ns = sum(int(str(item["prefill_ns"])) for item in completed)
    decode_ns = sum(int(str(item["decode_ns"])) for item in completed)
    ttft_ns = sum(int(str(item["ttft_ns"])) for item in completed)
    intervals = [
        int(str(interval)) for item in completed for interval in item["itl_ns"]
    ]
    gpu_samples = [int(str(item)) for item in observation["gpu_bytes_samples"]]
    host_samples = [int(str(item)) for item in observation["host_bytes_samples"]]
    for total, name in (
        (prefill_tokens, "prefill token total"),
        (decode_tokens, "decode token total"),
        (prefill_ns, "prefill time total"),
        (decode_ns, "decode time total"),
        (ttft_ns, "TTFT total"),
        (sum(intervals), "ITL total"),
    ):
        if total > _MAX_U64:
            raise ValueError(f"{name} exceeds uint64")
    return {
        "request_count": len(requests),
        "completed_count": len(completed),
        "prefill_tokens": str(prefill_tokens),
        "prefill_elapsed_ns": str(prefill_ns),
        "decode_tokens": str(decode_tokens),
        "decode_elapsed_ns": str(decode_ns),
        "ttft_total_ns": str(ttft_ns),
        "ttft_count": len(completed),
        "itl_total_ns": str(sum(intervals)),
        "itl_count": len(intervals),
        "peak_gpu_bytes": str(max(gpu_samples)),
        "peak_host_bytes": str(max(host_samples)),
    }


def _ratio_compare(
    left_numerator: int,
    left_denominator: int,
    right_numerator: int,
    right_denominator: int,
) -> int:
    left = left_numerator * right_denominator
    right = right_numerator * left_denominator
    return (left > right) - (left < right)


def _derive(
    captures: Sequence[dict[str, object]],
) -> tuple[str, str, dict[str, object]]:
    by_role: dict[str, list[dict[str, object]]] = {"control": [], "candidate": []}
    for capture in captures:
        role = capture["role"]
        if role in by_role:
            by_role[str(role)].append(capture)
    if len(by_role["control"]) != len(by_role["candidate"]):
        raise ValueError("paired capture counts differ")
    control_metrics: list[dict[str, object]] = []
    candidate_metrics: list[dict[str, object]] = []
    correctness = True
    sufficient = True
    no_regression = True
    strictly_better = False
    for control_capture, candidate_capture in zip(
        by_role["control"], by_role["candidate"], strict=True
    ):
        control_observation = control_capture["observation"]
        candidate_observation = candidate_capture["observation"]
        assert isinstance(control_observation, dict)
        assert isinstance(candidate_observation, dict)
        control_requests = control_observation["requests"]
        candidate_requests = candidate_observation["requests"]
        assert isinstance(control_requests, list)
        assert isinstance(candidate_requests, list)
        if len(control_requests) != len(candidate_requests):
            correctness = False
        else:
            for left, right in zip(control_requests, candidate_requests, strict=True):
                if (
                    left["ordinal"] != right["ordinal"]
                    or left["input_root"] != right["input_root"]
                    or left["prompt_tokens"] != right["prompt_tokens"]
                    or left["output_tokens"] != right["output_tokens"]
                    or left["outcome"] != "completed"
                    or right["outcome"] != "completed"
                    or left["output_root"] != right["output_root"]
                ):
                    correctness = False
        left_metrics = _metrics(control_observation)
        right_metrics = _metrics(candidate_observation)
        control_metrics.append(left_metrics)
        candidate_metrics.append(right_metrics)
        required = (
            int(left_metrics["completed_count"]),
            int(right_metrics["completed_count"]),
            int(left_metrics["prefill_elapsed_ns"]),
            int(right_metrics["prefill_elapsed_ns"]),
            int(left_metrics["decode_elapsed_ns"]),
            int(right_metrics["decode_elapsed_ns"]),
            int(left_metrics["ttft_count"]),
            int(right_metrics["ttft_count"]),
            int(left_metrics["itl_count"]),
            int(right_metrics["itl_count"]),
        )
        if any(item == 0 for item in required):
            sufficient = False
            continue
        comparisons = (
            _ratio_compare(
                int(right_metrics["prefill_tokens"]),
                int(right_metrics["prefill_elapsed_ns"]),
                int(left_metrics["prefill_tokens"]),
                int(left_metrics["prefill_elapsed_ns"]),
            ),
            _ratio_compare(
                int(right_metrics["decode_tokens"]),
                int(right_metrics["decode_elapsed_ns"]),
                int(left_metrics["decode_tokens"]),
                int(left_metrics["decode_elapsed_ns"]),
            ),
            -_ratio_compare(
                int(right_metrics["ttft_total_ns"]),
                int(right_metrics["ttft_count"]),
                int(left_metrics["ttft_total_ns"]),
                int(left_metrics["ttft_count"]),
            ),
            -_ratio_compare(
                int(right_metrics["itl_total_ns"]),
                int(right_metrics["itl_count"]),
                int(left_metrics["itl_total_ns"]),
                int(left_metrics["itl_count"]),
            ),
        )
        if any(item < 0 for item in comparisons):
            no_regression = False
        if any(item > 0 for item in comparisons):
            strictly_better = True
    correctness_state = "replayed_match" if correctness else "failed"
    if not correctness:
        performance_state = "not_evaluated"
    elif not sufficient:
        performance_state = "insufficient"
    elif no_regression and strictly_better:
        performance_state = "observed_better"
    else:
        performance_state = "observed_not_better"
    def checked_sum(items: Sequence[dict[str, object]], key: str) -> int:
        total = sum(int(item[key]) for item in items)
        if total > _MAX_U64:
            raise ValueError(f"aggregate {key} exceeds uint64")
        return total

    aggregate: dict[str, object] = {}
    for role, metrics in (
        ("control", control_metrics),
        ("candidate", candidate_metrics),
    ):
        aggregate[role] = {
            "request_count": sum(int(item["request_count"]) for item in metrics),
            "completed_count": sum(
                int(item["completed_count"]) for item in metrics
            ),
            "prefill_tokens": str(checked_sum(metrics, "prefill_tokens")),
            "prefill_elapsed_ns": str(checked_sum(metrics, "prefill_elapsed_ns")),
            "decode_tokens": str(checked_sum(metrics, "decode_tokens")),
            "decode_elapsed_ns": str(checked_sum(metrics, "decode_elapsed_ns")),
            "ttft_total_ns": str(checked_sum(metrics, "ttft_total_ns")),
            "ttft_count": sum(int(item["ttft_count"]) for item in metrics),
            "itl_total_ns": str(checked_sum(metrics, "itl_total_ns")),
            "itl_count": sum(int(item["itl_count"]) for item in metrics),
            "peak_gpu_bytes": str(
                max(int(item["peak_gpu_bytes"]) for item in metrics)
            ),
            "peak_host_bytes": str(
                max(int(item["peak_host_bytes"]) for item in metrics)
            ),
        }
    return correctness_state, performance_state, {
        "per_repetition": {
            "control": control_metrics,
            "candidate": candidate_metrics,
        },
        "aggregate": aggregate,
    }


def _lexical_absolute(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("receipt must be an absolute path")
    for candidate in (path, *path.parents):
        is_junction = getattr(candidate, "is_junction", lambda: False)
        try:
            if candidate.is_symlink() or is_junction():
                raise ValueError("receipt path must not traverse a symlink or junction")
        except OSError as error:
            raise ValueError("receipt path identity cannot be inspected") from error
    return path


def _stat_identity(value: object) -> tuple[int, int, object, object]:
    return (
        int(getattr(value, "st_size")),
        int(getattr(value, "st_mtime_ns")),
        getattr(value, "st_ino", None),
        getattr(value, "st_dev", None),
    )


def _read_stable_receipt(path: Path) -> bytes:
    source_path = _lexical_absolute(path)
    try:
        lexical_before = source_path.stat()
        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor = os.open(source_path, flags)
    except OSError as error:
        raise ValueError("receipt must be an absolute regular non-symlink file") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("receipt must be an absolute regular non-symlink file")
        if _stat_identity(lexical_before) != _stat_identity(before):
            raise ValueError("receipt changed while it was opened")
        if not 0 < before.st_size <= _MAX_RECEIPT_BYTES:
            raise ValueError("receipt exceeds its byte ceiling")
        chunks: list[bytes] = []
        total = 0
        while total <= _MAX_RECEIPT_BYTES:
            remaining = _MAX_RECEIPT_BYTES + 1 - total
            chunk = os.read(descriptor, min(64 << 10, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        source = b"".join(chunks)
        if len(source) > _MAX_RECEIPT_BYTES:
            raise ValueError("receipt exceeds its byte ceiling")
        after = os.fstat(descriptor)
        try:
            lexical_after = source_path.stat()
        except OSError as error:
            raise ValueError("receipt changed while it was read") from error
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(before) != _stat_identity(lexical_after)
            or len(source) != before.st_size
        ):
            raise ValueError("receipt changed while it was read")
        return source
    except OSError as error:
        raise ValueError("receipt could not be read through a stable file descriptor") from error
    finally:
        os.close(descriptor)


def _parse_receipt(path: Path) -> dict[str, object]:
    source = _read_stable_receipt(path)

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"receipt contains duplicate field: {key}")
            result[key] = item
        return result

    try:
        value = json.loads(source.decode("ascii"), object_pairs_hook=unique)
        canonical = _canonical(value)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and "duplicate field" in str(error):
            raise
        raise ValueError("receipt is not canonical ASCII JSON") from error
    if type(value) is not dict or source != canonical + b"\n":
        raise ValueError("receipt is not canonical ASCII JSON")
    return value


def verify_receipt(
    path: Path,
    expected_evidence_root: str,
    expected_plan_root: str,
    expected_m4_verification_root: str,
    expected_commit_sha: str,
    expected_runner_root: str,
) -> str:
    """Replay a v2 receipt under five caller-owned authority pins."""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    expected_evidence = _hex(expected_evidence_root, "expected evidence root")
    expected_plan = _hex(expected_plan_root, "expected plan root")
    expected_m4 = _hex(
        expected_m4_verification_root, "expected M4 verification root"
    )
    expected_commit = _hex(
        expected_commit_sha, "expected commit SHA", frozenset({40, 64})
    )
    expected_runner = _hex(expected_runner_root, "expected runner root")
    receipt = _fields(_parse_receipt(path), _RECEIPT_FIELDS, "receipt")
    if receipt["schema"] != "pih.deepseek.optimization-pair.v2":
        raise ValueError("receipt is not the required optimization-pair v2 schema")
    if (
        receipt["capture_scope"] != "non_authorizing_raw_capture"
        or receipt["analysis_abi"] != "paired_integer_replay_v1"
        or receipt["support_state"] != "hardware_evidence_open"
    ):
        raise ValueError("receipt authority state differs from the M5 v2 contract")
    evidence_root = _hex(receipt["evidence_root"], "receipt evidence root")
    if evidence_root != expected_evidence:
        raise ValueError("receipt differs from expected evidence root")
    plan, plan_root = _plan(
        receipt["plan"],
        receipt_plan_root=receipt["plan_root"],
        expected_plan_root=expected_plan,
        expected_m4_verification_root=expected_m4,
        expected_commit_sha=expected_commit,
        expected_runner_root=expected_runner,
    )
    executions = receipt["executions"]
    schedule = _schedule(plan)
    if type(executions) is not list or len(executions) != len(schedule):
        raise ValueError("execution set differs from the frozen schedule")
    normalized: list[dict[str, object]] = []
    previous_finished = -1
    for value, scheduled in zip(executions, schedule, strict=True):
        capture = _capture(
            value, plan=plan, plan_root=plan_root, scheduled=scheduled
        )
        started = int(str(capture["started_monotonic_ns"]))
        if started < previous_finished:
            raise ValueError("command capture intervals overlap or reorder")
        previous_finished = int(str(capture["finished_monotonic_ns"]))
        normalized.append(capture)
    if _canonical(executions) != _canonical(normalized):
        raise ValueError("execution capture does not normalize exactly")
    correctness_state, performance_state, derived = _derive(normalized)
    if _canonical(receipt["derived_metrics"]) != _canonical(derived):
        raise ValueError("derived metrics do not replay from raw observations")
    if receipt["correctness_state"] != correctness_state:
        raise ValueError("derived correctness state does not replay")
    if receipt["performance_state"] != performance_state:
        raise ValueError("derived performance state does not replay")
    if receipt["baseline_state"] != "runnable_observed":
        raise ValueError("derived baseline state does not replay")
    payload = {key: item for key, item in receipt.items() if key != "evidence_root"}
    replayed_evidence = _typed_json_root(
        "pih:deepseek-m5-optimization-pair:v2", payload
    )
    if replayed_evidence != evidence_root:
        raise ValueError("pair evidence root does not replay")
    verification_payload = {
        "schema": "pih.deepseek.m5.optimization_evidence_verification.v1",
        "support_state": "hardware_evidence_open",
        "evidence_root": evidence_root,
        "plan_root": plan_root,
        "m4_verification_root": expected_m4,
        "commit_sha": expected_commit,
        "runner_root": expected_runner,
        "correctness_state": correctness_state,
        "performance_state": performance_state,
        "baseline_state": "runnable_observed",
    }
    return _typed_json_root(
        "pih:deepseek-m5-optimization-verification:v1",
        verification_payload,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Independently replay one DeepSeek M5 v2 optimization receipt"
    )
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--expected-evidence-root", required=True)
    parser.add_argument("--expected-plan-root", required=True)
    parser.add_argument("--expected-m4-verification-root", required=True)
    parser.add_argument("--expected-commit-sha", required=True)
    parser.add_argument("--expected-runner-root", required=True)
    parsed = parser.parse_args(arguments)
    root = verify_receipt(
        parsed.receipt.absolute(),
        parsed.expected_evidence_root,
        parsed.expected_plan_root,
        parsed.expected_m4_verification_root,
        parsed.expected_commit_sha,
        parsed.expected_runner_root,
    )
    print(
        json.dumps(
            {
                "schema": (
                    "pih.deepseek.m5.optimization_evidence_verification.v1"
                ),
                "support_state": "hardware_evidence_open",
                "verification_root": root,
            },
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
