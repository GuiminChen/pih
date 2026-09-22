#!/usr/bin/env python3
"""Compile a deterministic, non-authorizing Qwen M2 qualification bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping


_INPUT_FIELDS = {
    "schema",
    "profile_id",
    "evidence_state",
    "hardware",
    "commands",
    "artifacts",
    "metrics",
    "independent_reference",
}
_HARDWARE_FIELDS = {
    "availability",
    "reason",
    "gpu_model",
    "gpu_uuid",
    "compute_capability",
    "driver_version",
    "toolkit_version",
}
_DESCRIPTOR_FIELDS = {"role", "path", "sha256", "bytes"}
_COMMAND_FIELDS = {"role", "argv", "exit_code", "stdout", "stderr"}
_REFERENCE_FIELDS = {"identity", "sha256"}
_MAX_OBJECT_BYTES = 16 << 30
_MAX_OBJECTS = 4096
_MAX_INPUT_BYTES = 16 << 20


def _object(value: object, *, name: str, fields: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields are invalid")
    return value


def _text(value: object, *, name: str, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not 1 <= len(value.encode("utf-8")) <= maximum:
        raise ValueError(f"{name} is invalid")
    return value


def _digest(value: object, *, name: str) -> str:
    text = _text(value, name=name, maximum=64)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} digest is invalid")
    return text


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def parse_input_json(source: bytes) -> object:
    if not isinstance(source, bytes):
        raise TypeError("qualification input must be bytes")
    if not source or len(source) > _MAX_INPUT_BYTES:
        raise ValueError("qualification input is empty or exceeds its byte ceiling")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"qualification input contains duplicate key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(source.decode("utf-8"), object_pairs_hook=unique_object)
    except UnicodeDecodeError as error:
        raise ValueError("qualification input must be UTF-8") from error
    except json.JSONDecodeError as error:
        raise ValueError("qualification input JSON is invalid") from error


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sync_directory(directory: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_file(path: Path) -> None:
    with path.open("r+b") as source:
        os.fsync(source.fileno())


def _publish_directory(staging: Path, target: Path) -> None:
    target.mkdir()
    published = False
    try:
        os.replace(staging / "objects", target / "objects")
        os.replace(staging / "manifest.json", target / "manifest.json")
        _sync_directory(target)
        staging.rmdir()
        _sync_directory(target.parent)
        published = True
    finally:
        if not published:
            shutil.rmtree(target, ignore_errors=True)
            try:
                _sync_directory(target.parent)
            except OSError:
                pass


def _verify_descriptor(value: object, *, name: str) -> tuple[dict[str, object], Path]:
    row = _object(value, name=name, fields=_DESCRIPTOR_FIELDS)
    role = _text(row["role"], name=f"{name}.role", maximum=128)
    path = Path(_text(row["path"], name=f"{name}.path")).resolve(strict=True)
    expected_digest = _digest(row["sha256"], name=f"{name}.sha256")
    expected_bytes = row["bytes"]
    if type(expected_bytes) is not int or not 0 <= expected_bytes <= _MAX_OBJECT_BYTES:
        raise ValueError(f"{name}.bytes is invalid")
    if not path.is_file():
        raise ValueError(f"{name}.path is not a regular file")
    hasher = hashlib.sha256()
    observed_bytes = 0
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            observed_bytes += len(chunk)
            if observed_bytes > expected_bytes or observed_bytes > _MAX_OBJECT_BYTES:
                raise ValueError(f"{name} byte count differs")
            hasher.update(chunk)
    if observed_bytes != expected_bytes:
        raise ValueError(f"{name} byte count differs")
    if hasher.hexdigest() != expected_digest:
        raise ValueError(f"{name} digest differs")
    return {
        "role": role,
        "sha256": expected_digest,
        "bytes": str(expected_bytes),
        "media_type": "application/octet-stream",
        "object_path": f"objects/{expected_digest}",
    }, path


def _compile_descriptors(
    values: object, *, name: str
) -> tuple[list[dict[str, object]], dict[str, Path]]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a nonempty array")
    if len(values) > _MAX_OBJECTS:
        raise ValueError(f"{name} exceeds its object bound")
    compiled: list[dict[str, object]] = []
    sources: dict[str, Path] = {}
    roles: set[str] = set()
    for index, value in enumerate(values):
        descriptor, source = _verify_descriptor(value, name=f"{name}[{index}]")
        role = str(descriptor["role"])
        if role in roles:
            raise ValueError(f"{name} contains a duplicate role")
        roles.add(role)
        digest = str(descriptor["sha256"])
        sources.setdefault(digest, source)
        compiled.append(descriptor)
    compiled.sort(key=lambda row: str(row["role"]))
    return compiled, sources


def _compile_commands(
    values: object,
) -> tuple[list[dict[str, object]], dict[str, Path]]:
    if not isinstance(values, list) or not values:
        raise ValueError("commands must be a nonempty array")
    if len(values) > 256:
        raise ValueError("commands exceeds its bound")
    compiled: list[dict[str, object]] = []
    sources: dict[str, Path] = {}
    roles: set[str] = set()
    for index, value in enumerate(values):
        command = _object(value, name=f"commands[{index}]", fields=_COMMAND_FIELDS)
        role = _text(command["role"], name=f"commands[{index}].role", maximum=128)
        if role in roles:
            raise ValueError("commands contains a duplicate role")
        roles.add(role)
        argv = command["argv"]
        if not isinstance(argv, list) or not argv or len(argv) > 256:
            raise ValueError(f"commands[{index}].argv is invalid")
        normalized_argv = [
            _text(argument, name=f"commands[{index}].argv", maximum=4096)
            for argument in argv
        ]
        exit_code = command["exit_code"]
        if type(exit_code) is not int or not -(1 << 31) <= exit_code < 1 << 31:
            raise ValueError(f"commands[{index}].exit_code is invalid")
        stdout, stdout_source = _verify_descriptor(
            command["stdout"], name=f"commands[{index}].stdout"
        )
        stderr, stderr_source = _verify_descriptor(
            command["stderr"], name=f"commands[{index}].stderr"
        )
        sources[str(stdout["sha256"])] = stdout_source
        sources[str(stderr["sha256"])] = stderr_source
        compiled.append(
            {
                "role": role,
                "argv": normalized_argv,
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
            }
        )
    compiled.sort(key=lambda row: str(row["role"]))
    return compiled, sources


def _compile_hardware(value: object, evidence_state: str) -> dict[str, object]:
    hardware = _object(value, name="hardware", fields=_HARDWARE_FIELDS)
    availability = _text(hardware["availability"], name="hardware.availability")
    if evidence_state == "hardware_evidence_open" and availability != "unavailable":
        raise ValueError("hardware_evidence_open requires unavailable target hardware")
    if availability != "unavailable":
        raise ValueError("this compiler only accepts unavailable target hardware")
    result: dict[str, object] = {
        "availability": availability,
        "reason": _text(hardware["reason"], name="hardware.reason"),
        "gpu_model": _text(hardware["gpu_model"], name="hardware.gpu_model"),
        "compute_capability": _text(
            hardware["compute_capability"], name="hardware.compute_capability", maximum=16
        ),
        "toolkit_version": _text(
            hardware["toolkit_version"], name="hardware.toolkit_version", maximum=64
        ),
    }
    for field in ("gpu_uuid", "driver_version"):
        item = hardware[field]
        if item is not None:
            raise ValueError(f"hardware.{field} must be null when target hardware is unavailable")
        result[field] = None
    return result


def compile_bundle(source: object, destination: str | os.PathLike[str]) -> dict[str, object]:
    root = _object(source, name="input", fields=_INPUT_FIELDS)
    if root["schema"] != "pih.qwen_m2_qualification_input.v1":
        raise ValueError("input schema is invalid")
    if root["profile_id"] != "QW-4090-INT4":
        raise ValueError("profile_id is invalid")
    if root["evidence_state"] != "hardware_evidence_open":
        raise ValueError("evidence_state cannot promote hardware support")
    hardware = _compile_hardware(root["hardware"], str(root["evidence_state"]))
    commands, command_sources = _compile_commands(root["commands"])
    artifacts, artifact_sources = _compile_descriptors(root["artifacts"], name="artifacts")
    metrics, metric_sources = _compile_descriptors(root["metrics"], name="metrics")
    reference = _object(
        root["independent_reference"],
        name="independent_reference",
        fields=_REFERENCE_FIELDS,
    )
    independent_reference = {
        "identity": _text(reference["identity"], name="independent_reference.identity"),
        "sha256": _digest(reference["sha256"], name="independent_reference.sha256"),
    }
    manifest: dict[str, object] = {
        "schema": "pih.qwen_m2_qualification_bundle.v1",
        "profile_id": "QW-4090-INT4",
        "hardware": hardware,
        "commands": commands,
        "artifacts": artifacts,
        "metrics": metrics,
        "independent_reference": independent_reference,
        "decision": {
            "support_status": "unsupported",
            "evidence_state": "hardware_evidence_open",
            "reason": "target_hardware_unavailable",
        },
    }
    identity = hashlib.sha256(
        b"pih.qwen_m2_qualification_bundle.v1\0" + _canonical(manifest)
    ).hexdigest()
    manifest["bundle_identity"] = identity

    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(f"bundle destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        objects = staging / "objects"
        objects.mkdir()
        all_sources = command_sources | artifact_sources | metric_sources
        for digest, source_path in sorted(all_sources.items()):
            output = objects / digest
            shutil.copyfile(source_path, output)
            if _file_digest(output) != digest:
                raise RuntimeError("published evidence object digest drifted")
            _sync_file(output)
        manifest_path = staging / "manifest.json"
        with manifest_path.open("wb") as published:
            published.write(_canonical(manifest) + b"\n")
            published.flush()
            os.fsync(published.fileno())
        _publish_directory(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    with arguments.input.open("rb") as input_file:
        source = parse_input_json(input_file.read(_MAX_INPUT_BYTES + 1))
    manifest = compile_bundle(source, arguments.output)
    print(manifest["bundle_identity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
