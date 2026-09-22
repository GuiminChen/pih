#!/usr/bin/env python3
"""Run one non-authorizing DeepSeek M5 raw A/B capture plan."""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, BinaryIO


_REPOSITORY = Path(__file__).resolve().parents[1]
if str(_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY))

from tools.evidence.deepseek_optimization_evidence import (
    DeepSeekBaselineSpec,
    DeepSeekOptimizationCapturePlan,
    DeepSeekOptimizationEvidenceReceipt,
    DeepSeekOptimizationRunSpec,
    _revalidate_plan,
    execution_schedule,
    invocation_id,
    typed_json_root,
)


_MAX_INPUT_BYTES = 8 << 20
_MAX_STREAM_BYTES = 16 << 20
_MAX_CAPTURE_PAYLOAD_BYTES = _MAX_STREAM_BYTES
_MAX_GPU_SAMPLE_BYTES = 64 << 10
_MAX_RUNNER_BYTES = 16 << 20
_MAX_COMMAND_FILE_BYTES = 2 << 30
_READ_CHUNK_BYTES = 64 << 10
_TERMINATION_GRACE_SECONDS = 2.0
_GPU_SAMPLE_TIMEOUT_SECONDS = 10
_NVIDIA_SMI_QUERY_ARGS = (
    "--query-gpu=index,uuid,name,compute_cap,memory.total,memory.used",
    "--format=csv,noheader,nounits",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _lexical_absolute(path: Path, name: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    for candidate in (path, *path.parents):
        is_junction = getattr(candidate, "is_junction", lambda: False)
        if candidate.is_symlink() or is_junction():
            raise ValueError(f"{name} must not traverse a symlink or junction")
    return path


def _stat_identity(value: object) -> tuple[int, int, object, object]:
    return (
        int(getattr(value, "st_size")),
        int(getattr(value, "st_mtime_ns")),
        getattr(value, "st_ino", None),
        getattr(value, "st_dev", None),
    )


def _read_stable_file(
    path: Path,
    name: str,
    *,
    maximum_bytes: int,
    retain_source: bool,
) -> tuple[int, str, bytes | None]:
    source_path = _lexical_absolute(path, name)
    try:
        lexical_before = source_path.stat()
    except OSError as error:
        raise ValueError(
            f"{name} must be an existing regular non-symlink file"
        ) from error
    if not stat.S_ISREG(lexical_before.st_mode):
        raise ValueError(f"{name} must be an existing regular non-symlink file")
    if not 0 < lexical_before.st_size <= maximum_bytes:
        raise ValueError(f"{name} is empty or exceeds its byte ceiling")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(source_path, flags)
    except OSError as error:
        raise ValueError(
            f"{name} must be an existing regular non-symlink file"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(
                f"{name} must be an existing regular non-symlink file"
            )
        if _stat_identity(lexical_before) != _stat_identity(before):
            raise ValueError(f"{name} changed while it was opened")
        if not 0 < before.st_size <= maximum_bytes:
            raise ValueError(f"{name} is empty or exceeds its byte ceiling")
        hasher = sha256()
        chunks: list[bytes] | None = [] if retain_source else None
        total = 0
        while total <= maximum_bytes:
            remaining = maximum_bytes + 1 - total
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            hasher.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
            total += len(chunk)
        if total > maximum_bytes:
            raise ValueError(f"{name} exceeds its byte ceiling")
        after = os.fstat(descriptor)
        try:
            _lexical_absolute(source_path, name)
            lexical_after = source_path.stat()
        except (OSError, ValueError) as error:
            raise ValueError(f"{name} changed while it was read") from error
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(before) != _stat_identity(lexical_after)
            or total != before.st_size
        ):
            raise ValueError(f"{name} changed while it was read")
        source = b"".join(chunks) if chunks is not None else None
        return total, hasher.hexdigest(), source
    except OSError as error:
        raise ValueError(
            f"{name} could not be read through a stable file descriptor"
        ) from error
    finally:
        os.close(descriptor)


def _object(path: Path, name: str) -> dict[str, Any]:
    _size, _digest, source = _read_stable_file(
        path,
        name,
        maximum_bytes=_MAX_INPUT_BYTES,
        retain_source=True,
    )
    assert source is not None

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON field: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(source.decode("ascii"), object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must contain canonical ASCII JSON") from error
    if type(value) is not dict or source not in {_canonical(value), _canonical(value) + b"\n"}:
        raise ValueError(f"{name} must contain one canonical JSON object")
    return value


def _sha_file(path: Path) -> str:
    _size, digest, _retained = _read_stable_file(
        path,
        "runner source",
        maximum_bytes=_MAX_RUNNER_BYTES,
        retain_source=False,
    )
    return digest


def _file_descriptor(path: Path, name: str) -> dict[str, object]:
    source = _lexical_absolute(path, name)
    size, digest, _retained = _read_stable_file(
        source,
        name,
        maximum_bytes=_MAX_COMMAND_FILE_BYTES,
        retain_source=False,
    )
    return {"path": str(source), "bytes": size, "sha256": digest}


def _command_files(value: object, role: str) -> tuple[dict[str, object], ...]:
    if type(value) is not list or not 1 <= len(value) <= 64:
        raise ValueError(f"{role} command_files must be a nonempty path list")
    paths: list[Path] = []
    for index, item in enumerate(value):
        if type(item) is not str:
            raise ValueError(f"{role} command file path must be text")
        paths.append(Path(item))
    collected: list[dict[str, object]] = []
    for index, path in enumerate(paths):
        try:
            collected.append(_file_descriptor(path, f"{role} command file {index}"))
        except ValueError as error:
            if index == 0:
                raise ValueError(
                    f"{role} command executable must be an existing absolute file"
                ) from error
            raise
    descriptors = tuple(collected)
    if len({str(item["path"]) for item in descriptors}) != len(descriptors):
        raise ValueError(f"{role} command_files contains duplicate paths")
    return descriptors


def _run_spec(path: Path, expected_role: str) -> DeepSeekOptimizationRunSpec:
    value = _object(path, f"{expected_role} run spec")
    fields = {
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
    if set(value) != fields or value["role"] != expected_role:
        raise ValueError(f"{expected_role} run spec schema is invalid")
    command = value["command_template"]
    if type(command) is not list:
        raise ValueError(f"{expected_role} command template must be an argv list")
    files = _command_files(value["command_files"], expected_role)
    try:
        return DeepSeekOptimizationRunSpec.create(
            role=value["role"],
            optimization=value["optimization"],
            model_revision=value["model_revision"],
            model_digest=value["model_digest"],
            profile_root=value["profile_root"],
            input_distribution_root=value["input_distribution_root"],
            concurrency=value["concurrency"],
            policy=value["policy"],
            policy_root=value["policy_root"],
            command_template=tuple(command),
            command_files=files,
        )
    except (TypeError, ValueError) as error:
        if files and tuple(command) and command[0] != files[0]["path"]:
            raise ValueError(
                f"{expected_role} command executable must be the first declared file"
            ) from error
        raise


def _baseline_spec(path: Path) -> DeepSeekBaselineSpec:
    value = _object(path, "baseline spec")
    if set(value) != {"name", "version", "command", "command_files"}:
        raise ValueError("baseline spec schema is invalid")
    name = value["name"]
    if name not in {"vllm", "sglang"}:
        raise ValueError("baseline name must be vllm or sglang")
    command = value["command"]
    if type(command) is not list:
        raise ValueError(f"{name} command must be an argv list")
    files = _command_files(value["command_files"], str(name))
    try:
        return DeepSeekBaselineSpec.create(
            name=name,
            version=value["version"],
            command=tuple(command),
            command_files=files,
        )
    except (TypeError, ValueError) as error:
        if files and tuple(command) and command[0] != files[0]["path"]:
            raise ValueError(
                f"{name} command executable must be the first declared file"
            ) from error
        raise


def _runner_root() -> str:
    return _sha_file(Path(__file__).absolute())


def build_plan(
    *,
    control_run: Path,
    candidate_run: Path,
    baseline_paths: tuple[Path, Path],
    workload_plan: Path,
    environment_path: Path,
    hardware_path: Path,
    hardware_verification_path: Path,
    expected_m4_verification_root: str,
    expected_commit_sha: str,
    working_directory: Path,
    per_command_timeout_seconds: int,
) -> DeepSeekOptimizationCapturePlan:
    control = _run_spec(control_run, "control")
    candidate = _run_spec(candidate_run, "candidate")
    baselines = tuple(_baseline_spec(path) for path in baseline_paths)
    if len(baselines) != 2:
        raise ValueError("exactly two baseline specs are required")
    workload = _object(workload_plan, "workload plan")
    if set(workload) != {"repetitions", "workload_instance_roots"} or type(
        workload["workload_instance_roots"]
    ) is not list:
        raise ValueError("workload plan schema is invalid")
    environment = _object(environment_path, "environment")
    if environment.get("commit") != expected_commit_sha:
        raise ValueError("environment commit differs from expected_commit_sha")
    hardware = _object(hardware_path, "hardware")
    verification = _object(hardware_verification_path, "M4 hardware verification")
    directory = _lexical_absolute(working_directory, "working_directory")
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("working_directory must be a regular non-symlink directory")
    return DeepSeekOptimizationCapturePlan.create(
        control=control,
        candidate=candidate,
        baselines=baselines,
        repetitions=workload["repetitions"],
        workload_instance_roots=tuple(workload["workload_instance_roots"]),
        environment=environment,
        hardware=hardware,
        m4_verification=verification,
        m4_verification_root=expected_m4_verification_root,
        runner_root=_runner_root(),
        working_directory=str(directory),
        per_command_timeout_seconds=per_command_timeout_seconds,
    )


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()
    deadline = time.monotonic() + _TERMINATION_GRACE_SECONDS
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    if process.poll() is None:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
    process.wait(timeout=_TERMINATION_GRACE_SECONDS)


def _terminate_live_posix_process_group(
    process_id: int,
    *,
    platform_name: str | None = None,
    kill_group: Callable[[int, int], None] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    if (os.name if platform_name is None else platform_name) != "posix":
        return False
    sender = getattr(os, "killpg") if kill_group is None else kill_group
    try:
        sender(process_id, 0)
    except ProcessLookupError:
        return False
    try:
        sender(process_id, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = clock() + _TERMINATION_GRACE_SECONDS
    while clock() < deadline:
        try:
            sender(process_id, 0)
        except ProcessLookupError:
            break
        sleeper(0.01)
    else:
        try:
            sender(process_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return True


def _reject_and_kill_live_descendants(
    process: subprocess.Popen[bytes], role: str,
) -> None:
    if _terminate_live_posix_process_group(process.pid):
        raise RuntimeError(f"{role} command left live descendant processes")


def _drain(
    stream: BinaryIO,
    buffer: bytearray,
    overflow: threading.Event,
    maximum_bytes: int,
) -> None:
    try:
        while chunk := stream.read(_READ_CHUNK_BYTES):
            if len(buffer) + len(chunk) > maximum_bytes:
                overflow.set()
                return
            buffer.extend(chunk)
    finally:
        stream.close()


def _execute_bounded(
    *,
    argv: tuple[str, ...],
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: int,
    role: str,
    maximum_stream_bytes: int = _MAX_STREAM_BYTES,
) -> tuple[int, bytes, bytes, int, int]:
    if (
        type(maximum_stream_bytes) is not int
        or not 0 < maximum_stream_bytes <= _MAX_STREAM_BYTES
    ):
        raise ValueError("command stream byte ceiling is invalid")
    creation_flags = 0
    if os.name == "nt":
        creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    started = time.monotonic_ns()
    deadline = time.monotonic() + timeout_seconds
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=os.name == "posix",
            creationflags=creation_flags,
        )
    except OSError as error:
        raise RuntimeError(f"{role} command executable could not be launched") from error
    assert process.stdout is not None and process.stderr is not None
    stdout = bytearray()
    stderr = bytearray()
    overflow = threading.Event()
    threads = (
        threading.Thread(
            target=_drain,
            args=(process.stdout, stdout, overflow, maximum_stream_bytes),
            daemon=True,
        ),
        threading.Thread(
            target=_drain,
            args=(process.stderr, stderr, overflow, maximum_stream_bytes),
            daemon=True,
        ),
    )
    for thread in threads:
        thread.start()
    timed_out = False
    while process.poll() is None:
        if overflow.is_set():
            _terminate_process(process)
            break
        if time.monotonic() >= deadline:
            timed_out = True
            _terminate_process(process)
            break
        time.sleep(0.01)
    descendants_terminated = _terminate_live_posix_process_group(process.pid)
    for thread in threads:
        thread.join(timeout=_TERMINATION_GRACE_SECONDS)
    if any(thread.is_alive() for thread in threads):
        _terminate_process(process)
        raise RuntimeError(f"{role} command stream did not close")
    finished = time.monotonic_ns()
    if overflow.is_set():
        raise RuntimeError(f"{role} command output exceeds its byte ceiling")
    if timed_out:
        raise TimeoutError(f"{role} command timed out")
    if descendants_terminated:
        raise RuntimeError(f"{role} command left live descendant processes")
    _reject_and_kill_live_descendants(process, role)
    return process.returncode, bytes(stdout), bytes(stderr), started, finished


def _select_gpu_rows(
    source: str,
    devices: Sequence[int],
) -> tuple[str, ...]:
    by_ordinal: dict[int, str] = {}
    for raw in source.splitlines():
        row = raw.strip()
        if not row:
            continue
        fields = row.split(", ")
        if (
            len(fields) != 6
            or not fields[0].isascii()
            or not fields[0].isdecimal()
            or (len(fields[0]) > 1 and fields[0].startswith("0"))
        ):
            raise RuntimeError("nvidia-smi returned an invalid GPU row")
        ordinal = int(fields[0])
        if ordinal in by_ordinal:
            raise RuntimeError("nvidia-smi returned a duplicate GPU ordinal")
        by_ordinal[ordinal] = row
    if any(device not in by_ordinal for device in devices):
        raise RuntimeError("nvidia-smi omitted a required GPU ordinal")
    return tuple(by_ordinal[device] for device in devices)


def _live_gpu_sample(
    hardware: Mapping[str, object],
    *,
    executor: Callable[..., tuple[int, bytes, bytes, int, int]] = _execute_bounded,
) -> tuple[str, ...]:
    environment = dict(os.environ)
    _validate_ambient_gpu_namespace(environment)
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi command executable is unavailable")
    returncode, stdout, stderr, _started, _finished = executor(
        argv=(executable, *_NVIDIA_SMI_QUERY_ARGS),
        cwd=Path.cwd().absolute(),
        environment=environment,
        timeout_seconds=_GPU_SAMPLE_TIMEOUT_SECONDS,
        role="nvidia-smi",
        maximum_stream_bytes=_MAX_GPU_SAMPLE_BYTES,
    )
    if len(stdout) > _MAX_GPU_SAMPLE_BYTES or len(stderr) > _MAX_GPU_SAMPLE_BYTES:
        raise RuntimeError("nvidia-smi output exceeds its byte ceiling")
    if returncode != 0:
        raise RuntimeError("nvidia-smi GPU identity query failed")
    try:
        source = stdout.decode("ascii")
    except UnicodeDecodeError as error:
        raise RuntimeError("nvidia-smi output is not canonical ASCII") from error
    devices = hardware["devices"]
    assert isinstance(devices, tuple)
    return _select_gpu_rows(source, tuple(int(item) for item in devices))


def _validate_ambient_gpu_namespace(
    environment: Mapping[str, str] | None = None,
) -> None:
    ambient = os.environ if environment is None else environment
    if "CUDA_VISIBLE_DEVICES" in ambient:
        raise RuntimeError("ambient CUDA_VISIBLE_DEVICES must be unset")
    if ambient.get("NVIDIA_VISIBLE_DEVICES", "all") != "all":
        raise RuntimeError(
            "ambient NVIDIA_VISIBLE_DEVICES must be unset or exactly all"
        )


def _validate_live_gpu_sample(
    sample: tuple[str, ...],
    hardware: Mapping[str, object],
    phase: str,
) -> None:
    inventory = hardware["gpu_inventory"]
    assert isinstance(inventory, tuple)
    if type(sample) is not tuple or len(sample) != len(inventory):
        raise RuntimeError(f"{phase} GPU sample does not match the topology")
    for row, expected in zip(sample, inventory, strict=True):
        if type(row) is not str:
            raise RuntimeError(f"{phase} GPU sample row is invalid")
        fields = row.split(", ")
        if (
            len(fields) != 6
            or row != ", ".join(fields)
            or ", ".join(fields[:5]) != expected
        ):
            raise RuntimeError(f"{phase} GPU identity drifted from the capture plan")
        used = fields[5]
        if (
            not used.isascii()
            or not used.isdecimal()
            or (len(used) > 1 and used.startswith("0"))
            or int(used) > int(fields[4])
        ):
            raise RuntimeError(f"{phase} GPU memory sample is invalid")


def _parse_observation(source: bytes, role: str) -> dict[str, object]:
    if not source or len(source) > _MAX_STREAM_BYTES:
        raise RuntimeError(f"{role} raw observation exceeds its byte ceiling")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"{role} raw observation contains duplicate field")
            result[key] = value
        return result

    try:
        value = json.loads(source.decode("ascii"), object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{role} raw observation is not canonical ASCII JSON") from error
    if type(value) is not dict or source != _canonical(value) + b"\n":
        raise RuntimeError(f"{role} raw observation has noncanonical or trailing bytes")
    return value


def _descriptors_for_role(
    plan: DeepSeekOptimizationCapturePlan,
    role: str,
) -> tuple[Mapping[str, object], ...]:
    if role == "control":
        return plan.control.command_files
    if role == "candidate":
        return plan.candidate.command_files
    return next(item.command_files for item in plan.baselines if item.name == role)


def _argv_for_schedule(
    plan: DeepSeekOptimizationCapturePlan,
    schedule: Mapping[str, object],
) -> tuple[str, ...]:
    role = str(schedule["role"])
    pair = int(schedule["pair_ordinal"])
    if role in {"control", "candidate"}:
        spec = plan.control if role == "control" else plan.candidate
        workloads = plan.payload["workload_instance_roots"]
        assert isinstance(workloads, tuple)
        return spec.render_command(pair, str(workloads[pair - 1]))
    return next(item.command for item in plan.baselines if item.name == role)


def _rehash_declared_files(
    descriptors: Sequence[Mapping[str, object]],
    role: str,
) -> tuple[dict[str, object], ...]:
    observed = tuple(
        _file_descriptor(Path(str(item["path"])), f"{role} command file {index}")
        for index, item in enumerate(descriptors)
    )
    expected = tuple(dict(item) for item in descriptors)
    if observed != expected:
        raise RuntimeError(f"{role} command file identity drifted")
    return observed


def run_pair(
    plan: DeepSeekOptimizationCapturePlan,
    *,
    gpu_sampler: Callable[[Mapping[str, object]], tuple[str, ...]] = _live_gpu_sample,
    system: str | None = None,
    executor: Callable[..., tuple[int, bytes, bytes, int, int]] = _execute_bounded,
) -> DeepSeekOptimizationEvidenceReceipt:
    plan = _revalidate_plan(plan)
    if (platform.system() if system is None else system) != "Linux":
        raise RuntimeError("DeepSeek M5 target capture requires Linux")
    _validate_ambient_gpu_namespace()
    environment = plan.payload["environment"]
    hardware = plan.payload["hardware"]
    assert isinstance(environment, Mapping)
    assert isinstance(hardware, Mapping)
    process_environment = environment["process"]
    assert isinstance(process_environment, Mapping)
    cwd = Path(str(plan.payload["working_directory"]))
    timeout = int(plan.payload["per_command_timeout_seconds"])
    captures: list[dict[str, object]] = []
    capture_payload_bytes = 2
    for schedule in execution_schedule(plan):
        role = str(schedule["role"])
        pair = int(schedule["pair_ordinal"])
        order_slot = int(schedule["order_slot"])
        execution_ordinal = int(schedule["execution_ordinal"])
        command_files = _descriptors_for_role(plan, role)
        before = _rehash_declared_files(command_files, role)
        gpu_pre = gpu_sampler(hardware)
        _validate_live_gpu_sample(gpu_pre, hardware, f"{role} pre-execution")
        argv = _argv_for_schedule(plan, schedule)
        invocation = invocation_id(
            plan.plan_root,
            execution_ordinal=execution_ordinal,
            role=role,
            pair_ordinal=pair,
            order_slot=order_slot,
        )
        child_environment = {
            str(key): str(value) for key, value in process_environment.items()
        }
        child_environment.update(
            {
                "PIH_M5_INVOCATION_ID": invocation,
                "PIH_M5_ROLE": role,
                "PIH_M5_REPETITION": str(pair),
                "PIH_M5_PLAN_ROOT": plan.plan_root,
            }
        )
        returncode, stdout, stderr, started, finished = executor(
            argv=argv,
            cwd=cwd,
            environment=child_environment,
            timeout_seconds=timeout,
            role=role,
        )
        if returncode != 0:
            raise RuntimeError(f"{role} command failed with exit code {returncode}")
        observation = _parse_observation(stdout, role)
        after = _rehash_declared_files(command_files, role)
        gpu_post = gpu_sampler(hardware)
        _validate_live_gpu_sample(gpu_post, hardware, f"{role} post-execution")
        if gpu_pre != gpu_post:
            raise RuntimeError(f"{role} GPU memory baseline was not restored")
        encoded_stderr_bytes = 4 * ((len(stderr) + 2) // 3)
        if encoded_stderr_bytes > _MAX_CAPTURE_PAYLOAD_BYTES - capture_payload_bytes:
            raise RuntimeError(
                "cumulative capture payload exceeds its byte ceiling"
            )
        capture = {
            "execution_ordinal": execution_ordinal,
            "role": role,
            "pair_ordinal": pair,
            "order_slot": order_slot,
            "argv": list(argv),
            "command_files_before": list(before),
            "command_files_after": list(after),
            "started_monotonic_ns": str(started),
            "finished_monotonic_ns": str(finished),
            "exit_code": returncode,
            "stdout_bytes": len(stdout),
            "stdout_sha256": sha256(stdout).hexdigest(),
            "stderr_bytes": len(stderr),
            "stderr_sha256": sha256(stderr).hexdigest(),
            "stderr_base64": base64.b64encode(stderr).decode("ascii"),
            "gpu_pre": list(gpu_pre),
            "gpu_post": list(gpu_post),
            "observation": observation,
            "observation_root": typed_json_root(
                "pih:deepseek-m5-raw-observation:v2", observation
            ),
        }
        increment = len(_canonical(capture)) + (1 if captures else 0)
        if increment > _MAX_CAPTURE_PAYLOAD_BYTES - capture_payload_bytes:
            raise RuntimeError(
                "cumulative capture payload exceeds its byte ceiling"
            )
        capture_payload_bytes += increment
        captures.append(capture)
    return DeepSeekOptimizationEvidenceReceipt.create(
        plan=plan, captures=tuple(captures)
    )


def _sync_directory(directory: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _preflight_output(output: Path) -> Path:
    destination = _lexical_absolute(output, "output")
    if destination.is_symlink() or destination.exists():
        raise FileExistsError("output must be a new non-symlink path")
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise ValueError("output parent must be a regular non-symlink directory")
    return destination


def _publication_payload(
    receipt: DeepSeekOptimizationEvidenceReceipt,
    *,
    expected_plan_root: str,
    expected_m4_verification_root: str,
    expected_commit_sha: str,
    expected_runner_root: str,
) -> bytes:
    payload = receipt.to_payload()
    replayed_evidence_root = typed_json_root(
        "pih:deepseek-m5-optimization-pair:v2", payload
    )
    if replayed_evidence_root != receipt.evidence_root:
        raise ValueError("receipt evidence root or payload does not replay")
    plan = payload.get("plan")
    if type(plan) is not dict:
        raise ValueError("receipt plan payload is invalid")
    replayed_plan_root = typed_json_root(
        "pih:deepseek-m5-capture-plan:v2", plan
    )
    if payload.get("plan_root") != replayed_plan_root:
        raise ValueError("receipt plan root or payload does not replay")
    if replayed_plan_root != expected_plan_root:
        raise ValueError("receipt plan root differs from the external pin")
    if plan.get("m4_verification_root") != expected_m4_verification_root:
        raise ValueError(
            "receipt M4 verification root differs from the external pin"
        )
    environment = plan.get("environment")
    if type(environment) is not dict:
        raise ValueError("receipt environment payload is invalid")
    if environment.get("commit") != expected_commit_sha:
        raise ValueError("receipt commit differs from the external pin")
    if plan.get("runner_root") != expected_runner_root:
        raise ValueError("receipt runner root differs from the external pin")
    payload["evidence_root"] = receipt.evidence_root
    encoded = _canonical(payload) + b"\n"
    if len(encoded) > _MAX_STREAM_BYTES:
        raise ValueError("optimization receipt exceeds its byte ceiling")
    return encoded


def publish_receipt(
    receipt: DeepSeekOptimizationEvidenceReceipt,
    output: Path,
    *,
    verifier: Callable[..., str],
    expected_plan_root: str,
    expected_m4_verification_root: str,
    expected_commit_sha: str,
    expected_runner_root: str,
) -> str:
    if not isinstance(receipt, DeepSeekOptimizationEvidenceReceipt):
        raise TypeError("receipt must be DeepSeek optimization evidence")
    destination = _preflight_output(output)
    encoded = _publication_payload(
        receipt,
        expected_plan_root=expected_plan_root,
        expected_m4_verification_root=expected_m4_verification_root,
        expected_commit_sha=expected_commit_sha,
        expected_runner_root=expected_runner_root,
    )
    temporary: Path | None = None
    verification_root: str | None = None
    published = False
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as sink:
            temporary = Path(sink.name)
            sink.write(encoded)
            sink.flush()
            os.fsync(sink.fileno())
        verification_root = verifier(
            path=temporary.absolute(),
            expected_evidence_root=receipt.evidence_root,
            expected_plan_root=expected_plan_root,
            expected_m4_verification_root=expected_m4_verification_root,
            expected_commit_sha=expected_commit_sha,
            expected_runner_root=expected_runner_root,
        )
        if (
            type(verification_root) is not str
            or len(verification_root) != 64
            or any(character not in "0123456789abcdef" for character in verification_root)
        ):
            raise RuntimeError("independent M5 verifier returned an invalid root")
        os.link(temporary, destination)
        published = True
        _sync_directory(destination.parent)
    except BaseException:
        if published:
            destination.unlink(missing_ok=True)
            try:
                _sync_directory(destination.parent)
            except OSError:
                pass
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    assert verification_root is not None
    return verification_root


def _load_independent_verifier() -> Callable[..., str]:
    try:
        from verify_deepseek_m5_optimization_evidence import verify_receipt
    except ImportError as error:
        raise RuntimeError("independent M5 verifier is unavailable") from error
    return verify_receipt


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a non-authorizing DeepSeek M5 raw A/B capture"
    )
    parser.add_argument("--run-hardware", action="store_true")
    parser.add_argument("--print-plan-roots", action="store_true")
    parser.add_argument("--control-run", required=True, type=Path)
    parser.add_argument("--candidate-run", required=True, type=Path)
    parser.add_argument("--baseline", required=True, action="append", type=Path)
    parser.add_argument("--workload-plan", required=True, type=Path)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--hardware", required=True, type=Path)
    parser.add_argument("--hardware-verification", required=True, type=Path)
    parser.add_argument("--expected-m4-verification-root", required=True)
    parser.add_argument("--expected-commit-sha", required=True)
    parser.add_argument("--expected-plan-root")
    parser.add_argument("--expected-runner-root")
    parser.add_argument("--working-directory", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--output", type=Path)
    parsed = parser.parse_args(arguments)
    if parsed.print_plan_roots and parsed.run_hardware:
        parser.error("--print-plan-roots and --run-hardware are mutually exclusive")
    if not parsed.print_plan_roots and not parsed.run_hardware:
        parser.error(
            "--run-hardware is required; use --print-plan-roots for read-only planning"
        )
    if len(parsed.baseline) != 2:
        parser.error("exactly two --baseline specs are required")
    try:
        plan = build_plan(
            control_run=parsed.control_run,
            candidate_run=parsed.candidate_run,
            baseline_paths=tuple(parsed.baseline),
            workload_plan=parsed.workload_plan,
            environment_path=parsed.environment,
            hardware_path=parsed.hardware,
            hardware_verification_path=parsed.hardware_verification,
            expected_m4_verification_root=parsed.expected_m4_verification_root,
            expected_commit_sha=parsed.expected_commit_sha,
            working_directory=parsed.working_directory,
            per_command_timeout_seconds=parsed.timeout_seconds,
        )
        if parsed.print_plan_roots:
            print(
                json.dumps(
                    {
                        "schema": "pih.deepseek.m5.capture-plan-roots.v2",
                        "plan_root": plan.plan_root,
                        "runner_root": plan.payload["runner_root"],
                        "m4_verification_root": parsed.expected_m4_verification_root,
                        "commit_sha": parsed.expected_commit_sha,
                        "support_state": "hardware_evidence_open",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        if (
            parsed.output is None
            or parsed.expected_plan_root is None
            or parsed.expected_runner_root is None
        ):
            raise ValueError(
                "hardware capture requires --output and external plan/runner roots"
            )
        if plan.plan_root != parsed.expected_plan_root:
            raise ValueError("capture plan root differs from --expected-plan-root")
        if plan.payload["runner_root"] != parsed.expected_runner_root:
            raise ValueError("runner root differs from --expected-runner-root")
        output = _preflight_output(parsed.output)
        receipt = run_pair(plan)
        verification_root = publish_receipt(
            receipt,
            output,
            verifier=_load_independent_verifier(),
            expected_plan_root=parsed.expected_plan_root,
            expected_m4_verification_root=parsed.expected_m4_verification_root,
            expected_commit_sha=parsed.expected_commit_sha,
            expected_runner_root=parsed.expected_runner_root,
        )
    except (
        FileExistsError,
        OSError,
        RuntimeError,
        TimeoutError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "schema": "pih.deepseek.m5.capture-publication.v2",
                "evidence_root": receipt.evidence_root,
                "verification_root": verification_root,
                "support_state": "hardware_evidence_open",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
